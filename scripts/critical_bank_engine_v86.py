"""Finite full-FP32 teacher/head engine with TRAIN weak-critical references.

No runnable worker/plan or automatic restart. A new owner must bound every
encoder/readout/reference call, clock, output and STOP. All old budgets closed.
"""
import copy,hashlib,random
from pathlib import Path
import numpy as np
import torch
from fdt_local.runner import numpy_state
from frozen_verbalizer_engine_v72 import FrozenEngine
from research.competing_margin_objective_v78 import decoder_losses,gold_margin,GROUP_ROW_WEIGHTS
from research.last_linear_objective_v79 import freeze_to_last_linear,LAST_LINEAR_NAMES
from research.actual_update_retention_v80 import bounded_adam_proposal
from research.critical_reference_bank_v85 import CriticalReferenceBank
from research.critical_reference_device_v86 import DeviceReference
from research.readout_precision_policy_v81 import POLICY
from research.fuzzy_factual_objective_v70 import critical_mask
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support

SCHEMA='fdt_fullFP32_critical_reference_engine_v86'
OBJECTIVE='V78_loss_fullFP32_lastLinear2+TRAIN_broad_top1+all_weak_critical_margin_guard'

class CriticalBankEngine(FrozenEngine):
    def __init__(self,model,rows,orders,budget,plan_sha,device='cuda'):
        super().__init__(model,rows,orders,budget,plan_sha,device)
        self.storage_delta_names=list(self.names);self.names=freeze_to_last_linear(self.model);self.optimizer_names=[self.names]
        named=dict(self.model.named_parameters());self.optimizer=torch.optim.AdamW([named[n] for n in self.names],lr=1e-5,weight_decay=.01)
        self.symbol_weights=torch.tensor(budget.frozen_symbol_profile['weights'],dtype=torch.float32,device=device)
        if self.symbol_weights.shape!=(258,) or self.symbol_weights[257]!=0 or not torch.isfinite(self.symbol_weights).all():raise ValueError('immutable source symbol profile')
        if not orders or any(len(group)!=5 or any(type(i)!=int or not 0<=i<len(rows) for i in group) for group in orders):raise ValueError('finite five-source groups')
        self.frozen_prefix_SHA=support.state_sha({k:v for k,v in self.model.state_dict().items() if k in self.storage_delta_names and k not in self.names})
        self.frozen_except_last2_SHA=support.state_sha({k:v for k,v in self.model.state_dict().items() if k not in self.names})
        self.objective_report=dict(objective=OBJECTIVE,precision_policy=POLICY,LR=1e-5,weight_decay=.01,group_weights=list(GROUP_ROW_WEIGHTS),max_attempts=len(orders),
            fresh2Adam=True,serialized_override_tensors=7,trainable_tensors=2,critical_floor_fraction=.5,critical_floor_cap=1.,weak_critical_margin_max=1.,bank_patch_cap=8192,
            broad_reference='fixed TRAIN-only source indices',frontier='ALL TRAIN weak critical; not DEV; no label/span NN feature',old_source_profile=copy.deepcopy(budget.frozen_symbol_profile))
        sources=[dict(id=r['id'],SHA=hashlib.sha256(r['prompt'].encode()).hexdigest()) for r in rows]
        self.bank=CriticalReferenceBank(sources,budget.anchor_indices,embedding_SHA=self.embedding_sha,prefix_SHA=self.frozen_prefix_SHA,parent_SHA=budget.parent_SHA)
        self.rejections=0;self.inflight=None;self.teacher_GPU_peak_bytes=0;self.reference_patch_work=0;self.bank_ready=False;self.reference_final=None;self._device_reference=None
        self.exposure.update(head_encoder_rows=0,head_encoder_UTF8_bytes=0)

    def verify_frozen(self):
        if support.state_sha({k:v for k,v in self.model.state_dict().items() if k not in self.names})!=self.frozen_except_last2_SHA:raise ValueError('frozen codec and readout prefix changed')

    @torch.no_grad()
    def cache_next(self):
        if self.inflight is not None:raise RuntimeError('interrupted teacher: exact state retained, automatic replay forbidden')
        if self.attempts or self.cache_cursor>=len(self.rows):raise StopIteration('one finite teacher pass')
        i=self.cache_cursor;row=self.rows[i];self.inflight=dict(stage='teacher_requested',row_index=i);captured=[]
        hook=self.model.readout.readout.net[2].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu()))
        try:_,t,z=super().encode_packet(row,self.budget.critical_targets[row['id']])
        finally:hook.remove()
        if z.dtype!=torch.float32 or len(captured)!=1 or captured[0].dtype!=torch.float32 or captured[0].shape!=(len(t['labels']),512):raise ValueError('actual teacher/prefix must execute fixed fullFP32 readout')
        y=t['labels'];valid=y!=257;margin,competitor=gold_margin(z,y);t['teacher_correct']&=valid;margin=margin.masked_fill(~valid,0.)
        t.update(schema='V86_loss_only_fullFP32_teacher',embedding_sha=self.embedding_sha,teacher_sha=self.teacher_sha,source_id=row['id'],source_SHA=hashlib.sha256(row['prompt'].encode()).hexdigest(),
            teacher_margin=margin,teacher_competitor=competitor.to(torch.int16),precision_policy=POLICY)
        self.inflight['stage']='teacher_packet_write_requested';directory=self.budget.output/'teacher_targets';directory.mkdir(exist_ok=True,parents=True);f=directory/f'{i:04d}.pt';sha=save_tensor(f,t,self.budget)
        self.cache_refs.append(dict(targets=str(f),targets_SHA=sha,cache_key=t['cache_key'],source_id=row['id']));self.inflight['stage']='teacher_packet_saved'
        feature=dict(hidden=captured[0],source_id=row['id'],source_SHA=t['source_SHA'],embedding_SHA=self.embedding_sha,prefix_SHA=self.frozen_prefix_SHA,parent_SHA=self.budget.parent_SHA,precision_policy=POLICY)
        target={k:t[k] for k in ('source_id','source_SHA','source_bytes','labels','critical_mask','teacher_correct','teacher_margin','precision_policy')}
        self.bank.ingest(i,feature,target);self.inflight['stage']='teacher_bank_ingested'
        self.cache_cursor+=1;self.exposure['encoder_rows']+=1;self.exposure['teacher_rows']+=1;self.exposure['encoder_UTF8_bytes']+=t['source_bytes'];self.inflight=None
        if self.device=='cuda':self.teacher_GPU_peak_bytes=max(self.teacher_GPU_peak_bytes,torch.cuda.max_memory_allocated())
        if self.cache_cursor==len(self.rows):
            if support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('teacher changed within finite phase')
            self.phase='reference_initialization'

    def packet(self,index,verify=True):
        ref=self.cache_refs[index];row=self.rows[index]
        if set(ref)!={'targets','targets_SHA','cache_key','source_id'} or (verify and support.sha(ref['targets'])!=ref['targets_SHA']):raise ValueError('fresh loss-only teacher SHA')
        t=torch.load(ref['targets'],map_location='cpu',weights_only=True);raw=row['prompt'].encode();y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:len(raw)]=torch.tensor(list(raw));expected[len(raw)]=256
        keys={'schema','labels','critical_mask','teacher_gold_probability','teacher_correct','source_bytes','cache_key','embedding_sha','teacher_sha','source_id','source_SHA','teacher_margin','teacher_competitor','precision_policy'}
        if (set(t)!=keys or t['schema']!='V86_loss_only_fullFP32_teacher' or t['source_id']!=row['id'] or t['source_SHA']!=hashlib.sha256(raw).hexdigest()
            or t['source_bytes']!=len(raw) or t['embedding_sha']!=self.embedding_sha or t['teacher_sha']!=self.teacher_sha or t['precision_policy']!=POLICY
            or ref['cache_key']!=self.source_key(row) or ref['cache_key']!=t['cache_key'] or not torch.equal(y.flatten(),expected)
            or not torch.equal(t['critical_mask'],critical_mask(y,self.budget.critical_targets[row['id']],len(raw)))):raise ValueError('teacher source/precision/target identity')
        if any(t[k].shape!=y.shape for k in ('critical_mask','teacher_gold_probability','teacher_correct','teacher_margin','teacher_competitor')) or not torch.isfinite(t['teacher_margin']).all():raise ValueError('teacher shapes/numeric')
        return t

    @torch.no_grad()
    def initialize_reference(self):
        if self.inflight is not None or self.bank_ready or self.cache_cursor!=len(self.rows) or self.attempts:raise ValueError('one completed TRAIN teacher before reference')
        self.inflight=dict(stage='reference_requested');features,_=self.bank.components();self.budget.charge('reference_linear');self.reference_patch_work+=len(features['hidden'])
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=False):z=self.model.readout.readout.net[2](features['hidden'].to(self.device)).reshape(-1,16,258)
        receipt=self.bank.calibrate(z.cpu());self._device_reference=DeviceReference(features,self.bank.calibration,self.device);self.bank_ready=True;self.inflight=None;self.phase='head_training';return receipt

    @torch.no_grad()
    def reference_decision(self):
        if not self.bank_ready:raise ValueError('no actual fixed reference')
        if self._device_reference is None:
            features,_=self.bank.components();self._device_reference=DeviceReference(features,self.bank.calibration,self.device)
        self.budget.charge('reference_linear');self.reference_patch_work+=len(self._device_reference.hidden)
        if self.reference_patch_work>8192*(4*len(self.orders)+2):raise ValueError('fixed reference projection work cap')
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=False):z=self.model.readout.readout.net[2](self._device_reference.hidden).reshape(-1,16,258)
        return self._device_reference.decision(z)

    @torch.no_grad()
    def extract_live(self,row):
        from research.text_diffusion_v1 import pack_prompts
        self.budget.charge('encode');self.model.eval();x=pack_prompts([row['prompt']],len(row['prompt'].encode()),self.device)
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):c,ep=self.model.extract(x,source_ids=(row['id'],),source_versions=(self.budget.data_sha,),model_epoch=self.embedding_sha)
        if ep.memories.exact.valid.any():raise ValueError('Exact must remain NULL')
        return c

    def step(self,force_overflow=False):
        if self.inflight is not None:raise RuntimeError('interrupted consumed group, no automatic replay')
        if not self.bank_ready or self.cache_cursor!=len(self.rows) or self.attempts>=len(self.orders):raise StopIteration('complete fixed teacher/reference then finite groups')
        self.budget.check();ids=self.orders[self.attempts];packets=[self.packet(i) for i in ids];self.attempts+=1;self.cursor+=1
        for t in packets:
            self.exposure['head_rows']+=1;self.exposure['head_UTF8_bytes']+=t['source_bytes'];self.exposure['head_symbols']+=t['source_bytes']+1;self.exposure['head_critical_positions']+=int(t['critical_mask'].sum())
        self.inflight=dict(stage='head_group_requested',consumed_attempt=self.attempts,rows=ids,completed_micros=[]);self.optimizer.zero_grad(set_to_none=True);micros=[]
        try:
            if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
            for j,(index,t,weight) in enumerate(zip(ids,packets,GROUP_ROW_WEIGHTS)):
                self.inflight.update(stage='head_micro_source_requested',micro_index=j);c=self.extract_live(self.rows[index]);self.exposure['head_encoder_rows']+=1;self.exposure['head_encoder_UTF8_bytes']+=t['source_bytes']
                self.budget.charge('readout');self.inflight['stage']='head_micro_backward'
                z=self.model.verbalize(c)
                if z.dtype!=torch.float32:raise ValueError('head backward must use genuine fullFP32 output path')
                loss,parts=decoder_losses(z,t['labels'].to(self.device),self.symbol_weights,t['critical_mask'].to(self.device),t['teacher_gold_probability'].to(self.device),t['teacher_correct'].to(self.device),t['teacher_margin'].to(self.device))
                if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('bounded numeric micro loss')
                self.scaler.scale(weight*loss).backward();record=dict(row=index,bytes=t['source_bytes'],weight=weight,loss=float(loss.detach()))
                record.update({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()});micros.append(record);self.inflight['completed_micros']=copy.deepcopy(micros);del z,c,loss
            named=dict(self.model.named_parameters())
            if force_overflow:named[self.names[0]].grad.fill_(float('inf'))
            self.inflight['stage']='head_unscale_clip_once';self.scaler.unscale_(self.optimizer)
            if any(named[n].grad is None for n in self.names) or any(v.grad is not None for n,v in named.items() if n not in self.names):raise ValueError('lastLinear2-only gradient ownership')
            norm=torch.nn.utils.clip_grad_norm_([named[n] for n in self.names],1.);self.inflight['stage']='head_actual_delta_reference'
            receipt=bounded_adam_proposal(self.optimizer,{n:named[n] for n in self.names},self.scaler,self.reference_decision,guard=self.budget.check)
            skip=receipt['AMP_skip'];self.skips+=int(skip);self.updates+=int(receipt['accepted']);self.rejections+=int(not skip and not receipt['accepted'])
            self.history.append(dict(attempt=self.attempts,updates=self.updates,rows=ids,micros=micros,AMP_skip=skip,rejected=not skip and not receipt['accepted'],
                update_receipt=receipt,scale=self.scaler.get_scale(),grad_norm=float(norm) if torch.isfinite(norm) else None,loss=sum(v['weight']*v['loss'] for v in micros),clip_and_Adam_calls=1,
                GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0,Exact_entries=0))
            self.optimizer.zero_grad(set_to_none=True);self.inflight=None
            if self.skips>=8 or sum(h['AMP_skip'] for h in self.history[-16:])>=3 or (not skip and not torch.isfinite(norm)) or (self.device=='cuda' and torch.cuda.max_memory_allocated()>12*1024**3):raise FloatingPointError('AMP/numeric/GPU stop')
            return self.history[-1]
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True);raise

    def export(self):
        s=super().export();s.update(schema=SCHEMA,objective=OBJECTIVE,objective_report=self.objective_report,precision_policy=POLICY,storage_delta_names=self.storage_delta_names,
            frozen_prefix_SHA=self.frozen_prefix_SHA,frozen_head_prefix_SHA=self.frozen_prefix_SHA,frozen_except_last2_SHA=self.frozen_except_last2_SHA,bank_state=self.bank.export(),bank_ready=self.bank_ready,reference_final=self.reference_final,
            reference_patch_work=self.reference_patch_work,rejections=self.rejections,inflight=copy.deepcopy(self.inflight),teacher_GPU_peak_bytes=self.teacher_GPU_peak_bytes,long_block_work=getattr(self.budget,'block_work',0))
        return s

    def restore(self,s):
        pending=s['inflight'] is not None and s['inflight']['stage'].startswith('head_');teacher_pending=s['inflight'] is not None and s['inflight']['stage'].startswith('teacher_')
        if (s['schema']!=SCHEMA or s['objective']!=OBJECTIVE or s['objective_report']!=self.objective_report or s['precision_policy']!=POLICY or s['plan_sha256']!=self.plan_sha
            or s['exact_policy']!=EXACT_POLICY or s['parent_reference']!=self.budget.parent_reference or s['orders']!=self.orders or s['optimizer_names']!=self.optimizer_names
            or not torch.equal(s['symbol_weights'],self.symbol_weights.cpu()) or s['embedding_sha']!=self.embedding_sha or s['teacher_sha']!=self.teacher_sha
            or s['frozen_prefix_SHA']!=self.frozen_prefix_SHA or s['frozen_head_prefix_SHA']!=self.frozen_prefix_SHA or s['frozen_except_last2_SHA']!=self.frozen_except_last2_SHA or s['storage_delta_names']!=self.storage_delta_names
            or s['input_capacity']!=self.model.encoder.input_capacity or s['gradients'] or s['new_dev_refs']):raise ValueError('full engine/model/precision/source/group identity')
        if (s['cursor']!=s['attempts'] or not 0<=s['attempts']<=len(self.orders) or len(s['history'])!=s['attempts']-int(pending)
            or [h['rows'] for h in s['history']]!=self.orders[:len(s['history'])] or s['updates']+s['skips']+s['rejections']!=len(s['history'])
            or not 0<=s['cache_cursor']<=len(self.rows) or len(s['cache_refs']) not in ({s['cache_cursor'],s['cache_cursor']+1} if teacher_pending else {s['cache_cursor']})
            or (s['attempts'] and not s['bank_ready']) or (s['bank_ready'] and s['cache_cursor']!=len(self.rows))):raise ValueError('finite/inflight phase cursor ledger')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for key in ('cache_cursor','cache_refs','new_dev_refs','attempts','updates','cursor','skips','phase','history','exposure','evaluations','inflight','teacher_GPU_peak_bytes','rejections','reference_patch_work','bank_ready','reference_final'):setattr(self,key,copy.deepcopy(s[key]))
        self.bank.restore(s['bank_state']);self._device_reference=None
        if self.bank.cursor not in ({self.cache_cursor,self.cache_cursor+1} if teacher_pending else {self.cache_cursor}) or self.bank_ready!=(self.bank.calibration is not None):raise ValueError('bank/teacher/reference phase mismatch')
        self.budget.calls=copy.deepcopy(s['calls']);self.budget.block_work=s['long_block_work'];self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng']);self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('fixed initial teacher changed')
        for i in range(len(self.cache_refs)):self.packet(i)
        torch.set_rng_state(s['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
