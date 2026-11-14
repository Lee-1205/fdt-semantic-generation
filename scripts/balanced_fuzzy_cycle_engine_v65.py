"""Live language encoder -> fuzzy Commit -> prefix-free language decoder."""
import copy,random
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts
from research.fuzzy_roundtrip_v64 import EXACT_POLICY,enable_cycle_training
import text_generation_support_v1 as support

from research.fuzzy_cycle_objective_v65 import training_weights,OBJECTIVE

SCHEMA='fdt_live_fuzzy_roundtrip_v65_fullstate'


class CycleEngine:
    def __init__(self,model,rows,orders,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.rows=rows;self.orders=orders;self.budget=budget;self.plan_sha=plan_sha;self.device=device
        self.names=enable_cycle_training(self.model);named=dict(self.model.named_parameters())
        lookup,self.objective_report=training_weights(rows);self.symbol_weights=torch.tensor(lookup,device=device,dtype=torch.float32)
        groups=[]
        for prefix,lr in (('encoder.core.language.blocks.',1e-6),('encoder.core.',3e-6),('readout.',1e-4)):
            ids=[n for n in self.names if n.startswith(prefix) and not any(n in g['names'] for g in groups)]
            if ids:groups.append(dict(names=ids,lr=lr))
        assert sorted(sum([g['names'] for g in groups],[]))==sorted(self.names)
        self.optimizer=torch.optim.AdamW([dict(params=[named[n] for n in g['names']],lr=g['lr']) for g in groups],weight_decay=.01)
        self.optimizer_names=[g['names'] for g in groups]
        self.scaler=torch.amp.GradScaler(device,init_scale=1024.)
        self.attempts=self.updates=self.cursor=self.skips=0;self.history=[];self.phase='initial';self.evaluations={};self.inactive_cuda_rng=[]
        self.exposure=dict(rows=0,target_UTF8_bytes=0,active_symbols=0)
        self.frozen_sha=support.state_sha({n:p for n,p in self.model.named_parameters() if not p.requires_grad})
    def step(self,force_overflow=False):
        self.budget.check()
        if self.attempts>=len(self.orders):raise StopIteration('new finite cycle ended')
        ids=self.orders[self.attempts];assert len(ids)==1
        row=self.rows[ids[0]];raw=row['prompt'].encode();x=pack_prompts([row['prompt']],len(raw),self.device)
        self.attempts+=1;self.cursor+=1;self.exposure['rows']+=1;self.exposure['target_UTF8_bytes']+=len(raw);self.exposure['active_symbols']+=len(raw)+1
        self.model.train();self.optimizer.zero_grad(set_to_none=True);self.budget.charge('encode');self.budget.charge('readout')
        if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            logits,commit,episode=self.model(x,source_ids=(row['id'],),source_versions=(self.budget.data_sha,),model_epoch=self.plan_sha+':'+str(self.attempts))
            y=torch.full(logits.shape[:2],257,dtype=torch.long,device=self.device);flat=y.flatten();flat[:len(raw)]=torch.tensor(list(raw),device=self.device);flat[len(raw)]=256
            active=y!=257
            ce=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),y,ignore_index=257,reduction='none')
            # Uniform byte objective plus explicit EOS learning. Neither target
            # bytes nor desired output length is passed to the language reader.
            byte_loss=(ce*self.symbol_weights[y]).sum()/active.sum();eos_loss=ce[y==256].mean();loss=byte_loss+.1*eos_loss
        if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('cycle loss stop')
        self.scaler.scale(loss).backward()
        named=dict(self.model.named_parameters())
        if force_overflow:named[self.names[0]].grad.fill_(float('inf'))
        self.scaler.unscale_(self.optimizer)
        if self.device=='cuda' and torch.cuda.max_memory_allocated()>12*1024**3:raise MemoryError('fixed GPU12GiB cycle cap')
        if any(named[n].grad is None for n in self.names):raise ValueError('missing live encoder/decoder gradient')
        if any(p.grad is not None for n,p in named.items() if not p.requires_grad):raise ValueError('frozen reasoning or memory gradient')
        norm=torch.nn.utils.clip_grad_norm_([named[n] for n in self.names],1.)
        scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update();skip=self.scaler.get_scale()<scale
        self.skips+=int(skip);self.updates+=int(not skip)
        record=dict(attempt=self.attempts,updates=self.updates,rows=ids,bytes=len(raw),loss=float(loss.detach()),weighted_byte_CE=float(byte_loss.detach()),byte_CE=float(ce.sum().detach()/active.sum()),EOS_CE=float(eos_loss.detach()),AMP_skip=skip,scale=self.scaler.get_scale(),grad_norm=float(norm) if torch.isfinite(norm) else None,Exact_entries=0,GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0)
        self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
        if sum(h['AMP_skip'] for h in self.history[-16:])>=3 or self.skips>=8:raise FloatingPointError('persistent AMP skip')
        if not skip and not torch.isfinite(norm):raise FloatingPointError('nonfinite applied cycle update')
        if record['GPU_peak_bytes']>12*1024**3:raise MemoryError('fixed GPU12GiB cycle cap after Adam')
        return record
    def verify_frozen(self):
        if support.state_sha({n:p for n,p in self.model.named_parameters() if not p.requires_grad})!=self.frozen_sha:raise ValueError('frozen reasoning/prior/source weights changed')
    def export(self):
        return dict(schema=SCHEMA,objective=OBJECTIVE,symbol_weights=self.symbol_weights.detach().cpu(),objective_report=self.objective_report,plan_sha256=self.plan_sha,exact_policy=EXACT_POLICY,parent_reference=self.budget.parent_reference,
            model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},core=self.model.encoder.c.__dict__,input_capacity=self.model.encoder.input_capacity,
            optimizer=self.optimizer.state_dict(),optimizer_names=self.optimizer_names,scaler=self.scaler.state_dict(),orders=self.orders,
            attempts=self.attempts,updates=self.updates,cursor=self.cursor,skips=self.skips,history=self.history,exposure=self.exposure,phase=self.phase,
            evaluations=self.evaluations,calls=self.budget.calls,frozen_sha=self.frozen_sha,gradients={},
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else self.inactive_cuda_rng,
            python_rng=random.getstate(),numpy_rng=numpy_state(),denoise_calls=0,Exact_entries=0,copy_bytes=0)
    def restore(self,state):
        if (state['objective']!=OBJECTIVE or not torch.equal(state['symbol_weights'].cpu(),self.symbol_weights.cpu()) or state['objective_report']!=self.objective_report or state['schema']!=SCHEMA or state['plan_sha256']!=self.plan_sha or state['exact_policy']!=EXACT_POLICY or state['parent_reference']!=self.budget.parent_reference
            or state['orders']!=self.orders or state['optimizer_names']!=self.optimizer_names or state['input_capacity']!=self.model.encoder.input_capacity
            or state['cursor']!=sum(map(len,self.orders[:state['attempts']])) or [h['rows'] for h in state['history']]!=self.orders[:state['attempts']] or state['gradients']):raise ValueError('cycle recovery contract')
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        for key in ('attempts','updates','cursor','skips','history','exposure','phase','evaluations','frozen_sha'):setattr(self,key,copy.deepcopy(state[key]))
        self.budget.calls=copy.deepcopy(state['calls']);self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        torch.set_rng_state(state['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);n=state['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        self.verify_frozen()
