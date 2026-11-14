"""Five live source microbatches at one parameter state, one clip/Adam update.

Language repair only: seven readout tensors. Future thought-only training is a
different phase. Teacher margins/labels are loss-only; features are never saved.
"""
import copy,hashlib,random
from pathlib import Path
import numpy as np
import torch
from update_balance_engine_v77 import RetentionEngine as PreviousEngine
from research.competing_margin_objective_v78 import OBJECTIVE,decoder_losses,gold_margin,frozen_objective_report,GROUP_ROW_WEIGHTS
from research.fuzzy_factual_objective_v70 import critical_mask
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support

SCHEMA='fdt_grouped_competing_margin_v78_fullstate'


class MarginEngine(PreviousEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.objective_report=frozen_objective_report(self.objective_report,self.budget.frozen_symbol_profile,len(self.orders))
        if not self.orders or any(len(group)!=5 for group in self.orders):raise ValueError('V78 five-row finite groups')

    def cache_next(self):
        if self.inflight is not None:raise RuntimeError('interrupted V78 teacher, no automatic replay')
        if self.attempts or self.cache_cursor>=len(self.rows):raise StopIteration('finite teacher phase')
        i=self.cache_cursor;row=self.rows[i];self.inflight=dict(stage='teacher',row_index=i)
        _,t,z=self.encode_packet(row,self.budget.critical_targets[row['id']])
        margin,competitor=gold_margin(z,t['labels'])
        t.update(schema='V78_TRAIN_loss_only_teacher_margin',teacher_margin=margin.cpu(),teacher_competitor=competitor.to(torch.int16).cpu(),
            embedding_sha=self.embedding_sha,teacher_sha=self.teacher_sha,source_id=row['id'],source_SHA=hashlib.sha256(row['prompt'].encode()).hexdigest())
        if self.device=='cuda':self.teacher_GPU_peak_bytes=max(self.teacher_GPU_peak_bytes,torch.cuda.max_memory_allocated())
        directory=self.budget.output/'teacher_targets';directory.mkdir(exist_ok=True,parents=True);f=directory/f'{i:04d}.pt';sha=save_tensor(f,t,self.budget)
        self.cache_refs.append(dict(targets=str(f),targets_SHA=sha,cache_key=t['cache_key'],source_id=row['id']))
        self.cache_cursor+=1;self.exposure['encoder_rows']+=1;self.exposure['teacher_rows']+=1;self.exposure['encoder_UTF8_bytes']+=t['source_bytes'];self.inflight=None
        if self.cache_cursor==len(self.rows):
            if support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('teacher changed during phase')
            self.phase='head_training'

    def packet(self,index,verify=True):
        ref=self.cache_refs[index]
        if set(ref)!={'targets','targets_SHA','cache_key','source_id'} or (verify and support.sha(ref['targets'])!=ref['targets_SHA']):raise ValueError('V78 teacher reference')
        t=torch.load(ref['targets'],map_location='cpu',weights_only=True);row=self.rows[index];b=row['prompt'].encode();y=t['labels']
        keys={'schema','labels','critical_mask','teacher_gold_probability','teacher_correct','source_bytes','cache_key','embedding_sha','teacher_sha','source_id','source_SHA','teacher_margin','teacher_competitor'}
        if (set(t)!=keys or t['schema']!='V78_TRAIN_loss_only_teacher_margin' or t['cache_key']!=self.source_key(row) or t['embedding_sha']!=self.embedding_sha
            or t['teacher_sha']!=self.teacher_sha or t['source_id']!=row['id'] or ref['source_id']!=row['id'] or t['source_SHA']!=hashlib.sha256(b).hexdigest()
            or y.shape!=((len(b)+15)//16+1,16) or t['source_bytes']!=len(b)):raise ValueError('V78 loss-only source epoch')
        expected=torch.full_like(y,257);expected.flatten()[:len(b)]=torch.tensor(list(b));expected.flatten()[len(b)]=256
        if not torch.equal(y,expected) or any(t[k].shape!=y.shape for k in ('critical_mask','teacher_correct','teacher_gold_probability','teacher_margin','teacher_competitor')):raise ValueError('V78 target/margin shape')
        if not torch.equal(critical_mask(y,self.budget.critical_targets[row['id']],len(b)),t['critical_mask']):raise ValueError('TRAIN mask differs')
        if (not torch.isfinite(t['teacher_margin']).all() or not torch.isfinite(t['teacher_gold_probability']).all()
            or ((t['teacher_gold_probability']<0)|(t['teacher_gold_probability']>1)).any()
            or t['teacher_competitor'].dtype!=torch.int16 or ((t['teacher_competitor']<0)|(t['teacher_competitor']>257)).any()
            or (t['teacher_competitor'].long()==y).any() or ((t['teacher_margin']<0)&t['teacher_correct']).any() or ((t['teacher_margin']>0)&~t['teacher_correct']).any()):raise ValueError('V78 malformed teacher competitors')
        return t

    def step(self,force_overflow=False):
        if self.inflight is not None:raise RuntimeError('interrupted V78 group, no automatic replay')
        self.budget.check()
        if self.cache_cursor!=len(self.rows) or self.attempts>=len(self.orders):raise StopIteration('finite teacher/group ordering')
        ids=self.orders[self.attempts];packets=[self.packet(i) for i in ids]
        self.attempts+=1;self.cursor+=1
        for t in packets:
            self.exposure['head_rows']+=1;self.exposure['head_UTF8_bytes']+=t['source_bytes'];self.exposure['head_symbols']+=t['source_bytes']+1;self.exposure['head_critical_positions']+=int(t['critical_mask'].sum())
        self.inflight=dict(stage='head_group_requested',consumed_attempt=self.attempts,rows=ids,completed_micros=[])
        self.optimizer.zero_grad(set_to_none=True);micros=[]
        try:
            if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
            for micro,(index,t,weight) in enumerate(zip(ids,packets,GROUP_ROW_WEIGHTS)):
                self.inflight.update(stage='head_micro_source_requested',micro_index=micro)
                c=self.extract_live(self.rows[index]);self.exposure['head_encoder_rows']+=1;self.exposure['head_encoder_UTF8_bytes']+=t['source_bytes']
                self.model.readout.readout.train();self.budget.charge('readout');self.inflight['stage']='head_micro_backward'
                with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
                    z=self.model.verbalize(c)
                    loss,parts=decoder_losses(z,t['labels'].to(self.device),self.symbol_weights,t['critical_mask'].to(self.device),
                        t['teacher_gold_probability'].to(self.device),t['teacher_correct'].to(self.device),t['teacher_margin'].to(self.device))
                if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('bounded V78 micro loss')
                self.scaler.scale(weight*loss).backward()
                record=dict(row=index,bytes=t['source_bytes'],loss=float(loss.detach()),weight=weight)
                record.update({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()});micros.append(record)
                self.inflight['completed_micros']=copy.deepcopy(micros)
                del z,c,loss
            named=dict(self.model.named_parameters())
            if force_overflow:named[self.names[0]].grad.fill_(float('inf'))
            self.inflight['stage']='head_group_unscale_clip_once';self.scaler.unscale_(self.optimizer)
            if any(named[n].grad is None for n in self.names) or any(v.grad is not None for n,v in named.items() if n not in self.names):raise ValueError('V78 seven-readout-only gradients')
            norm=torch.nn.utils.clip_grad_norm_([named[n] for n in self.names],1.);scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update();skip=self.scaler.get_scale()<scale
            self.skips+=int(skip);self.updates+=int(not skip)
            record=dict(attempt=self.attempts,updates=self.updates,rows=ids,micros=micros,bytes=sum(t['source_bytes'] for t in packets),
                loss=sum(v['weight']*v['loss'] for v in micros),AMP_skip=skip,scale=self.scaler.get_scale(),grad_norm=float(norm) if torch.isfinite(norm) else None,
                GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0,clip_and_Adam_calls=1,Exact_entries=0)
            self.history.append(record);self.optimizer.zero_grad(set_to_none=True);self.inflight=None
            if record['GPU_peak_bytes']>12*1024**3 or self.skips>=8 or sum(h['AMP_skip'] for h in self.history[-16:])>=3 or (not skip and not torch.isfinite(norm)):raise FloatingPointError('V78 AMP/numeric/GPU stop')
            return record
        except BaseException:
            # Preserve consumed group/micro/call/exposure state, never replay it.
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def export(self):
        state=super().export();state.update(schema=SCHEMA,objective=OBJECTIVE);return state

    def restore(self,s):
        pending=s['inflight'] is not None and s['inflight']['stage'].startswith('head_')
        if (s['schema']!=SCHEMA or s['objective']!=OBJECTIVE or s['objective_report']!=self.objective_report or not torch.equal(s['symbol_weights'],self.symbol_weights.cpu())
            or s['plan_sha256']!=self.plan_sha or s['exact_policy']!=EXACT_POLICY or s['orders']!=self.orders or s['optimizer_names']!=self.optimizer_names or s['parent_reference']!=self.budget.parent_reference
            or s['input_capacity']!=self.model.encoder.input_capacity or s['embedding_sha']!=self.embedding_sha or s['teacher_sha']!=self.teacher_sha or s['gradients']
            or s['cache_cursor']!=len(s['cache_refs']) or s['cache_cursor']>len(self.rows) or s['cursor']!=s['attempts'] or [h['rows'] for h in s['history']]!=self.orders[:len(s['history'])]
            or len(s['history'])!=s['attempts']-int(pending) or (s['attempts'] and s['cache_cursor']!=len(self.rows)) or s['new_dev_refs']):raise ValueError('V78 exact phase/group recovery identity')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for key in ('cache_cursor','cache_refs','new_dev_refs','attempts','updates','cursor','skips','phase','history','exposure','evaluations','inflight','teacher_GPU_peak_bytes'):setattr(self,key,copy.deepcopy(s[key]))
        self.budget.calls=copy.deepcopy(s['calls']);self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng']);self.budget.block_work=s['long_block_work'];self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('initial teacher mismatch')
        for i in range(self.cache_cursor):self.packet(i)
        torch.set_rng_state(s['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
