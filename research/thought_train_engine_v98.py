"""Finite, recoverable thought-only micro/Adam engine; no execution owner.

Full language endpoint frozen. A checkpoint boundary has either completed
micros or a READY group; partially executed micro graphs are never replayed.
Any micro exception restores preceding scaled gradients and records ABORTED.
Optimizer exceptions restore thought/Adam/scaler/gradients before mutation.
"""
from dataclasses import asdict
import copy,math,random
import numpy as np
import torch
from research.memory_thought_diffusion_v94 import prepare_thought_optimizer
from research.thought_only_update_v94 import ThoughtOnlyUpdate
from research.thought_language_objective_v96 import ThoughtLanguageObjective


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch=torch.get_rng_state().clone(),
        CUDA=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(s):
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(s['torch'])
    if s['CUDA']:torch.cuda.set_rng_state_all(s['CUDA'])


class ThoughtTrainEngine:
    def __init__(self,thought,codec,codec_SHA,orders,*,batch=8,lr=1e-4,source_metadata=None):
        if not orders or batch<1 or not math.isfinite(lr) or lr<=0:raise ValueError('finite nonempty exact order/batch/LR')
        self.thought,self.codec,self.codec_SHA=thought,codec,codec_SHA;self.orders=list(orders);self.batch=batch;self.base_lr=lr
        self.device=next(thought.parameters()).device.type;self.metadata=copy.deepcopy(source_metadata or {})
        boundary,self.optimizer=prepare_thought_optimizer(codec,thought,lr)
        self.update=ThoughtOnlyUpdate(boundary,self.optimizer);self.objective=ThoughtLanguageObjective(thought,codec,self.update,codec_SHA)
        self.scaler=self._scaler();self.runtime=dict(phase='IDLE',cursor=0,attempts=0,updates=0,AMP_skips=0,consecutive_skips=0,
            pending_ids=[],pending_losses=[],history=[],exposure_bytes=0,calls=dict(denoise=0,readout=0),aborted=None)
        self.thought.train();self.optimizer.zero_grad(set_to_none=True)

    def _scaler(self):return torch.amp.GradScaler(self.device,init_scale=1024.,growth_interval=16384,enabled=True)

    def _gradients(self):return {n:None if p.grad is None else p.grad.detach().clone() for n,p in self.thought.named_parameters()}

    def _put_gradients(self,values):
        if set(values)!=set(dict(self.thought.named_parameters())):raise ValueError('exact thought gradient ownership')
        for n,p in self.thought.named_parameters():p.grad=None if values[n] is None else values[n].to(p.device).clone()

    def begin(self):
        r=self.runtime
        if r['phase']!='IDLE' or r['cursor']>=len(self.orders):raise ValueError('one next finite group, no rewind/extraepoch')
        r['attempts']+=1;r['phase']='MICROS';r['pending_ids']=[];r['pending_losses']=[]
        total=math.ceil(len(self.orders)/self.batch);factor=.1+.9*.5*(1+math.cos(math.pi*(r['attempts']-1)/total))
        for g in self.optimizer.param_groups:g['lr']=self.base_lr*factor
        self.optimizer.zero_grad(set_to_none=True)

    def group_size(self):
        start=(self.runtime['attempts']-1)*self.batch
        return min(self.batch,len(self.orders)-start)

    def micro(self,identity,source,packet,charge=lambda kind:None):
        r=self.runtime
        if r['phase']!='MICROS' or r['cursor']>=len(self.orders) or identity!=self.orders[r['cursor']]:raise ValueError('exact next finite source')
        gradients=self._gradients();calls_before=dict(r['calls'])
        def count(kind):
            if kind not in r['calls']:raise ValueError('thought/D calls only; source E cache reused')
            charge(kind);r['calls'][kind]+=1
        try:
            shape=packet['semantic_target'].shape
            # Every used RNG is preserved. No gold/noise construction reaches
            # target-free inference; this is the training-only VP corruption.
            t=.001+.998*torch.rand(1,device=self.device);noise=torch.randn(shape,device=self.device)
            with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
                loss,metrics=self.objective(source,packet,t,noise,count)
            if float(loss.detach())>1000:raise FloatingPointError('fixed gross numeric loss bound1000')
            self.scaler.scale(loss/self.group_size()).backward()
        except BaseException as e:
            self._put_gradients(gradients)
            r['phase']='ABORTED';r['aborted']=dict(source_id=identity,error=repr(e),calls_delta={k:r['calls'][k]-calls_before[k] for k in r['calls']},
                no_graph_replay=True,previous_complete_micro_gradients_restored=True)
            # The row was attempted; its calls/RNG/exposure remain consumed.
            r['cursor']+=1;r['exposure_bytes']+=int((packet['labels']<256).sum());raise
        r['pending_ids'].append(identity);r['pending_losses'].append(metrics);r['cursor']+=1;r['exposure_bytes']+=metrics['active_bytes']
        if len(r['pending_ids'])==self.group_size():r['phase']='READY'
        self.update.assert_fast();return metrics

    def finish(self,*,force_overflow=False):
        r=self.runtime
        if r['phase']!='READY':raise ValueError('whole finite group before Adam')
        self.update.assert_fast();snapshot=self.payload(audit=False)
        try:
            if force_overflow:
                # CPU contract only; actual worker never enables injection.
                next(p for p in self.thought.parameters() if p.grad is not None).grad.flatten()[0]=float('inf')
            self.scaler.unscale_(self.optimizer)
            gradients=[p.grad for p in self.thought.parameters() if p.grad is not None]
            finite=all(torch.isfinite(g).all() for g in gradients)
            if finite:
                norm=torch.nn.utils.clip_grad_norm_(self.thought.parameters(),1.,error_if_nonfinite=True)
                if not torch.isfinite(norm):raise FloatingPointError('finite pre-step norm, no overflowed clipping')
            else:norm=torch.tensor(float('inf'))
            scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update();skipped=self.scaler.get_scale()<scale
            if skipped:r['AMP_skips']+=1;r['consecutive_skips']+=1
            else:
                if not finite:raise FloatingPointError('nonfinite gradients were not skipped')
                r['updates']+=1;r['consecutive_skips']=0
            self.update.assert_fast()
        except BaseException:
            self.restore(snapshot,audit=False);raise
        values=r['pending_losses'];means={k:sum(x[k] for x in values)/len(values) for k in ('latent','byte_CE','EOS_CE')}
        r['history'].append(dict(attempt=r['attempts'],cursor=r['cursor'],update=r['updates'],skip=skipped,LR=self.optimizer.param_groups[0]['lr'],
            source_ids=list(r['pending_ids']),rows=len(values),loss=means,grad_norm=None if skipped else float(norm)))
        r['pending_ids']=[];r['pending_losses']=[];r['phase']='DONE' if r['cursor']==len(self.orders) else 'IDLE'
        self.optimizer.zero_grad(set_to_none=True)
        if r['consecutive_skips']>=3:raise FloatingPointError('three consecutive AMP skips; consumed finite rows, no rewind')
        return r['history'][-1]

    def payload(self,*,audit=True):
        if audit:self.update.audit_endpoint()
        self.update.assert_fast()
        return cpu_tree(dict(schema='thought_only_fullstate_v98',thought_config=asdict(self.thought.c),thought=self.thought.state_dict(),
            optimizer=self.optimizer.state_dict(),optimizer_names=list(dict(self.thought.named_parameters())),scaler=self.scaler.state_dict(),
            gradients=self._gradients(),ALL_RNG=rng_state(),runtime=self.runtime,orders=self.orders,batch=self.batch,base_lr=self.base_lr,
            codec_SHA=self.codec_SHA,frozen_endpoint_digest=self.update.boundary.endpoint_digest,source_metadata=self.metadata))

    def restore(self,p,*,audit=True):
        if (p['schema']!='thought_only_fullstate_v98' or p['thought_config']!=asdict(self.thought.c) or p['orders']!=self.orders or p['batch']!=self.batch
                or p['codec_SHA']!=self.codec_SHA or p['frozen_endpoint_digest']!=self.update.boundary.endpoint_digest
                or p['source_metadata']!=self.metadata or p['optimizer_names']!=list(dict(self.thought.named_parameters()))):raise ValueError('complete matching source/endpoint/config/order/ownership')
        self.thought.load_state_dict(p['thought'],strict=True)
        # Same-device optimizer.load_state_dict may retain the input Adam tensor
        # storages (including CPU step counters). Future steps must NOT mutate
        # the saved/checkpoint dictionary that supplied the restore.
        self.optimizer.load_state_dict(cpu_tree(p['optimizer']))
        self.scaler=self._scaler();self.scaler.load_state_dict(p['scaler'])
        # READY can resume directly at unscale_, without another scale(loss).
        # Public scale() materializes restored scale/growth on the right device;
        # no model call, RNG draw, new gradient, or private optimizer-stage edit.
        self.scaler.scale(torch.zeros((),device=self.device))
        self._put_gradients(p['gradients']);self.runtime=copy.deepcopy(p['runtime'])
        self.base_lr=p['base_lr'];restore_rng(p['ALL_RNG']);self.thought.train();self.update.assert_fast()
        if audit:self.update.audit_endpoint()
