"""Finite loss-only teacher phase, then fresh fixed-upstream source each step."""
import copy,hashlib,random
from pathlib import Path
import numpy as np
import torch
from frozen_verbalizer_engine_v72 import FrozenEngine
from research.critical_retention_objective_v75 import OBJECTIVE,COEFFICIENTS,decoder_losses
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from research.fuzzy_factual_objective_v70 import critical_mask
from research.text_diffusion_v1 import pack_prompts
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support

SCHEMA='fdt_live_critical_retention_v75_fullstate'

class RetentionEngine(FrozenEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.inflight=None;self.teacher_GPU_peak_bytes=0
        for k in ('critical_CE_coefficient','critical_masks_TRAIN_only','anchor_coefficient'):self.objective_report.pop(k,None)
        self.objective_report.update(objective=OBJECTIVE,coefficients=COEFFICIENTS,decoder_LR=1e-5,
            critical_groups_independently_normalized=True,teacher_cache='loss-only; no feature cache; live frozen source reencoded for head')
        self.optimizer.param_groups[0]['lr']=1e-5
        self.exposure.update(head_encoder_rows=0,head_encoder_UTF8_bytes=0)
    def cache_next(self):
        if self.inflight is not None:raise RuntimeError('interrupted V75 phase; no automatic replay')
        if self.attempts or self.cache_cursor>=len(self.rows):raise StopIteration('finite teacher phase')
        i=self.cache_cursor;row=self.rows[i];self.inflight=dict(stage='teacher',row_index=i)
        f,t,z=self.encode_packet(row,self.budget.critical_targets[row['id']])
        if self.device=='cuda':self.teacher_GPU_peak_bytes=max(self.teacher_GPU_peak_bytes,torch.cuda.max_memory_allocated())
        t.update(schema='V75_TRAIN_loss_only_teacher',embedding_sha=self.embedding_sha,teacher_sha=self.teacher_sha,source_id=row['id'],source_SHA=hashlib.sha256(row['prompt'].encode()).hexdigest())
        folder=self.budget.output/'teacher_targets';folder.mkdir(exist_ok=True,parents=True);file=folder/f'{i:04d}.pt'
        sha=save_tensor(file,t,self.budget);self.cache_refs.append(dict(targets=str(file),targets_SHA=sha,cache_key=t['cache_key'],source_id=row['id']))
        self.cache_cursor+=1;self.exposure['encoder_rows']+=1;self.exposure['teacher_rows']+=1;self.exposure['encoder_UTF8_bytes']+=t['source_bytes'];self.inflight=None
        if self.cache_cursor==len(self.rows):
            if support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('teacher changed before phase boundary')
            self.phase='head_training'
    def packet(self,index,verify=True):
        ref=self.cache_refs[index]
        if set(ref)!={'targets','targets_SHA','cache_key','source_id'} or (verify and support.sha(ref['targets'])!=ref['targets_SHA']):raise ValueError('V75 teacher cache reference')
        t=torch.load(ref['targets'],map_location='cpu',weights_only=True);row=self.rows[index];b=row['prompt'].encode();slots=(len(b)+15)//16
        expected_keys={'schema','labels','critical_mask','teacher_gold_probability','teacher_correct','source_bytes','cache_key','embedding_sha','teacher_sha','source_id','source_SHA'}
        if (set(t)!=expected_keys or t['schema']!='V75_TRAIN_loss_only_teacher' or t['cache_key']!=self.source_key(row) or t['embedding_sha']!=self.embedding_sha
            or t['teacher_sha']!=self.teacher_sha or t['source_id']!=row['id'] or ref['source_id']!=row['id'] or t['source_SHA']!=hashlib.sha256(b).hexdigest()
            or t['labels'].shape!=(slots+1,16) or t['source_bytes']!=len(b)):raise ValueError('loss-only packet source epoch')
        y=t['labels'];expected=torch.full_like(y,257);expected.flatten()[:len(b)]=torch.tensor(list(b));expected.flatten()[len(b)]=256
        if (not torch.equal(y,expected) or any(t[k].shape!=y.shape for k in ('critical_mask','teacher_correct','teacher_gold_probability'))
            or not torch.equal(critical_mask(y,self.budget.critical_targets[row['id']],len(b)),t['critical_mask'])):raise ValueError('loss-only TRAIN labels/mask')
        return t
    @torch.no_grad()
    def extract_live(self,row):
        self.budget.charge('encode');self.model.eval();x=pack_prompts([row['prompt']],len(row['prompt'].encode()),self.device)
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            commit,ep=self.model.extract(x,source_ids=(row['id'],),source_versions=(self.budget.data_sha,),model_epoch=self.embedding_sha)
        if ep.memories.exact.valid.any():raise ValueError('Exact must be NULL')
        return commit
    def step(self,force_overflow=False):
        if self.inflight is not None:raise RuntimeError('interrupted V75 phase; no automatic replay')
        self.budget.check()
        if self.cache_cursor!=len(self.rows) or self.attempts>=len(self.orders):raise StopIteration('finite teacher/head ordering')
        ids=self.orders[self.attempts];assert len(ids)==1;t=self.packet(ids[0]);row=self.rows[ids[0]]
        self.attempts+=1;self.cursor+=1;self.exposure['head_rows']+=1;self.exposure['head_UTF8_bytes']+=t['source_bytes'];self.exposure['head_symbols']+=t['source_bytes']+1;self.exposure['head_critical_positions']+=int(t['critical_mask'].sum())
        self.inflight=dict(stage='head_source_requested',consumed_attempt=self.attempts,rows=ids)
        try:
            if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
            c=self.extract_live(row);self.exposure['head_encoder_rows']+=1;self.exposure['head_encoder_UTF8_bytes']+=t['source_bytes'];self.inflight['stage']='head_readout_requested'
            self.model.readout.readout.train();self.optimizer.zero_grad(set_to_none=True);self.budget.charge('readout')
            with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
                z=self.model.verbalize(c);loss,parts=decoder_losses(z,t['labels'].to(self.device),self.symbol_weights,t['critical_mask'].to(self.device),t['teacher_gold_probability'].to(self.device),t['teacher_correct'].to(self.device))
            if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('bounded V75 loss')
            self.inflight['stage']='head_backward';self.scaler.scale(loss).backward();named=dict(self.model.named_parameters())
            if force_overflow:named[self.names[0]].grad.fill_(float('inf'))
            self.scaler.unscale_(self.optimizer)
            if any(named[n].grad is None for n in self.names) or any(v.grad is not None for n,v in named.items() if n not in self.names):raise ValueError('V75 seven final-readout gradient ownership')
            norm=torch.nn.utils.clip_grad_norm_([named[n] for n in self.names],1.);old_scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update();skip=self.scaler.get_scale()<old_scale
            self.skips+=int(skip);self.updates+=int(not skip)
            record=dict(attempt=self.attempts,updates=self.updates,rows=ids,bytes=t['source_bytes'],loss=float(loss.detach()),AMP_skip=skip,scale=self.scaler.get_scale(),
                grad_norm=float(norm) if torch.isfinite(norm) else None,GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0,Exact_entries=0)
            record.update({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()});self.history.append(record);self.optimizer.zero_grad(set_to_none=True);self.inflight=None
            if record['GPU_peak_bytes']>12*1024**3 or self.skips>=8 or sum(h['AMP_skip'] for h in self.history[-16:])>=3 or (not skip and not torch.isfinite(norm)):raise FloatingPointError('V75 AMP/numeric/GPU stop')
            return record
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
    def export(self):
        s=super().export();s.update(schema=SCHEMA,objective=OBJECTIVE,inflight=copy.deepcopy(self.inflight),long_block_work=getattr(self.budget,'block_work',0),teacher_GPU_peak_bytes=self.teacher_GPU_peak_bytes);return s
    def restore(self,s):
        pending_head=s['inflight'] is not None and s['inflight']['stage'].startswith('head_')
        if (s['schema']!=SCHEMA or s['objective']!=OBJECTIVE or s['objective_report']!=self.objective_report or not torch.equal(s['symbol_weights'],self.symbol_weights.cpu())
            or s['plan_sha256']!=self.plan_sha or s['exact_policy']!=EXACT_POLICY or s['orders']!=self.orders or s['optimizer_names']!=self.optimizer_names or s['parent_reference']!=self.budget.parent_reference
            or s['input_capacity']!=self.model.encoder.input_capacity or s['embedding_sha']!=self.embedding_sha or s['teacher_sha']!=self.teacher_sha or s['gradients']
            or s['cache_cursor']!=len(s['cache_refs']) or s['cache_cursor']>len(self.rows) or s['cursor']!=s['attempts'] or [h['rows'] for h in s['history']]!=self.orders[:len(s['history'])]
            or len(s['history'])!=s['attempts']-int(pending_head) or (s['attempts'] and s['cache_cursor']!=len(self.rows)) or s['new_dev_refs']):raise ValueError('V75 full phase recovery identity')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for k in ('cache_cursor','cache_refs','new_dev_refs','attempts','updates','cursor','skips','phase','history','exposure','evaluations','inflight','teacher_GPU_peak_bytes'):setattr(self,k,copy.deepcopy(s[k]))
        self.budget.calls=copy.deepcopy(s['calls']);self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng']);self.budget.block_work=s['long_block_work'];self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('initial teacher mismatch')
        for i in range(self.cache_cursor):self.packet(i)
        torch.set_rng_state(s['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
