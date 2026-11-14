"""Five live source microbatches at one parameter state, one clip/Adam update.

Language repair only: seven readout tensors. Future thought-only training is a
different phase. Teacher margins/labels are loss-only; features are never saved.
"""
import copy,hashlib,random
from research.actual_update_retention_v80 import bounded_adam_proposal,TrainDecision
from pathlib import Path
import numpy as np
import torch
from update_balance_engine_v77 import RetentionEngine as PreviousEngine
from research.update_retention_objective_v80 import OBJECTIVE,decoder_losses,gold_margin,frozen_objective_report,GROUP_ROW_WEIGHTS,freeze_to_last_linear
from research.fuzzy_factual_objective_v70 import critical_mask
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support

SCHEMA='fdt_guarded_readout_v80_fullstate'


class MarginEngine(PreviousEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.storage_delta_names=list(self.names)
        self.names=freeze_to_last_linear(self.model);self.optimizer_names=[self.names]
        named=dict(self.model.named_parameters());self.optimizer=torch.optim.AdamW([named[n] for n in self.names],lr=1e-5,weight_decay=.01)
        self.frozen_head_prefix_SHA=support.state_sha({k:v for k,v in self.model.state_dict().items() if k in self.storage_delta_names and k not in self.names})
        self.frozen_except_last2_SHA=support.state_sha({k:v for k,v in self.model.state_dict().items() if k not in self.names})
        self.objective_report=frozen_objective_report(self.objective_report,self.budget.frozen_symbol_profile,len(self.orders))
        if not self.orders or any(len(group)!=5 for group in self.orders):raise ValueError('V80 five-row finite groups')
        self.anchor_indices=list(self.budget.anchor_indices);self.anchor_refs=[];self.bank_ref=None;self.bank_summary=None;self.rejections=0;self.reference_patch_work=0;self._bank=None;self.reference_final_ref=None

    def verify_frozen(self):
        if support.state_sha({k:v for k,v in self.model.state_dict().items() if k not in self.names})!=self.frozen_except_last2_SHA:raise ValueError('V80 frozen features/encoder/memories changed')
        if support.state_sha({k:v for k,v in self.model.state_dict().items() if k not in self.storage_delta_names})!=self.embedding_sha:raise ValueError('V80 Commit embedding identity changed')

    def cache_next(self):
        if self.inflight is not None:raise RuntimeError('interrupted V80 teacher, no automatic replay')
        if self.attempts or self.cache_cursor>=len(self.rows):raise StopIteration('finite teacher phase')
        i=self.cache_cursor;row=self.rows[i];self.inflight=dict(stage='teacher',row_index=i)
        captured=[]
        handle=self.model.readout.readout.net[2].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu())) if i in self.anchor_indices else None
        try:_,t,z=self.encode_packet(row,self.budget.critical_targets[row['id']])
        finally:
            if handle is not None:handle.remove()
        margin,competitor=gold_margin(z,t['labels'])
        t.update(schema='V80_TRAIN_loss_only_teacher_margin',teacher_margin=margin.cpu(),teacher_competitor=competitor.to(torch.int16).cpu(),
            embedding_sha=self.embedding_sha,teacher_sha=self.teacher_sha,source_id=row['id'],source_SHA=hashlib.sha256(row['prompt'].encode()).hexdigest())
        if self.device=='cuda':self.teacher_GPU_peak_bytes=max(self.teacher_GPU_peak_bytes,torch.cuda.max_memory_allocated())
        directory=self.budget.output/'teacher_targets';directory.mkdir(exist_ok=True,parents=True);f=directory/f'{i:04d}.pt';sha=save_tensor(f,t,self.budget)
        self.cache_refs.append(dict(targets=str(f),targets_SHA=sha,cache_key=t['cache_key'],source_id=row['id']))
        if i in self.anchor_indices:
            if len(captured)!=1 or captured[0].shape!=(len(t['labels']),self.model.readout.readout.net[2].in_features):raise ValueError('frozen prefix feature hook')
            feature=dict(schema='V80_TRAIN_FROZEN_PREFIX_ONLY',hidden=captured[0],source_id=row['id'],cache_key=t['cache_key'],embedding_sha=self.embedding_sha,frozen_prefix_sha=self.frozen_head_prefix_SHA)
            directory=self.budget.output/'TRAIN_reference';directory.mkdir(exist_ok=True);f=directory/f'feature_{i:04d}.pt';fsha=save_tensor(f,feature,self.budget)
            self.anchor_refs.append(dict(index=i,features=str(f),SHA=fsha,targets=str(self.cache_refs[-1]['targets']),targets_SHA=self.cache_refs[-1]['targets_SHA']))
        self.cache_cursor+=1;self.exposure['encoder_rows']+=1;self.exposure['teacher_rows']+=1;self.exposure['encoder_UTF8_bytes']+=t['source_bytes'];self.inflight=None
        if self.cache_cursor==len(self.rows):
            if support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('teacher changed during phase')
            self.phase='head_training'

    def packet(self,index,verify=True):
        ref=self.cache_refs[index]
        if set(ref)!={'targets','targets_SHA','cache_key','source_id'} or (verify and support.sha(ref['targets'])!=ref['targets_SHA']):raise ValueError('V80 teacher reference')
        t=torch.load(ref['targets'],map_location='cpu',weights_only=True);row=self.rows[index];b=row['prompt'].encode();y=t['labels']
        keys={'schema','labels','critical_mask','teacher_gold_probability','teacher_correct','source_bytes','cache_key','embedding_sha','teacher_sha','source_id','source_SHA','teacher_margin','teacher_competitor'}
        if (set(t)!=keys or t['schema']!='V80_TRAIN_loss_only_teacher_margin' or t['cache_key']!=self.source_key(row) or t['embedding_sha']!=self.embedding_sha
            or t['teacher_sha']!=self.teacher_sha or t['source_id']!=row['id'] or ref['source_id']!=row['id'] or t['source_SHA']!=hashlib.sha256(b).hexdigest()
            or y.shape!=((len(b)+15)//16+1,16) or t['source_bytes']!=len(b)):raise ValueError('V80 loss-only source epoch')
        expected=torch.full_like(y,257);expected.flatten()[:len(b)]=torch.tensor(list(b));expected.flatten()[len(b)]=256
        if not torch.equal(y,expected) or any(t[k].shape!=y.shape for k in ('critical_mask','teacher_correct','teacher_gold_probability','teacher_margin','teacher_competitor')):raise ValueError('V80 target/margin shape')
        if not torch.equal(critical_mask(y,self.budget.critical_targets[row['id']],len(b)),t['critical_mask']):raise ValueError('TRAIN mask differs')
        if (not torch.isfinite(t['teacher_margin']).all() or not torch.isfinite(t['teacher_gold_probability']).all()
            or ((t['teacher_gold_probability']<0)|(t['teacher_gold_probability']>1)).any()
            or t['teacher_competitor'].dtype!=torch.int16 or ((t['teacher_competitor']<0)|(t['teacher_competitor']>257)).any()
            or (t['teacher_competitor'].long()==y).any() or ((t['teacher_margin']<0)&t['teacher_correct']).any() or ((t['teacher_margin']>0)&~t['teacher_correct']).any()):raise ValueError('V80 malformed teacher competitors')
        return t

    def step(self,force_overflow=False):
        if self.inflight is not None:raise RuntimeError('interrupted V80 group, no automatic replay')
        self.budget.check()
        if self.bank_ref is None:raise ValueError('TRAIN reference bank not initialized')
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
                self.model.readout.readout.net[2].train();self.budget.charge('readout');self.inflight['stage']='head_micro_backward'
                with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
                    z=self.model.verbalize(c)
                    loss,parts=decoder_losses(z,t['labels'].to(self.device),self.symbol_weights,t['critical_mask'].to(self.device),
                        t['teacher_gold_probability'].to(self.device),t['teacher_correct'].to(self.device),t['teacher_margin'].to(self.device))
                if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('bounded V80 micro loss')
                self.scaler.scale(weight*loss).backward()
                record=dict(row=index,bytes=t['source_bytes'],loss=float(loss.detach()),weight=weight)
                record.update({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()});micros.append(record)
                self.inflight['completed_micros']=copy.deepcopy(micros)
                del z,c,loss
            named=dict(self.model.named_parameters())
            if force_overflow:named[self.names[0]].grad.fill_(float('inf'))
            self.inflight['stage']='head_group_unscale_clip_once';self.scaler.unscale_(self.optimizer)
            if any(named[n].grad is None for n in self.names) or any(v.grad is not None for n,v in named.items() if n not in self.names):raise ValueError('V80 two-lastlinear-only gradients')
            norm=torch.nn.utils.clip_grad_norm_([named[n] for n in self.names],1.)
            self.inflight['stage']='head_actual_Adam_proposal_reference'
            receipt=bounded_adam_proposal(self.optimizer,{n:named[n] for n in self.names},self.scaler,self.reference_decision,guard=self.budget.check)
            skip=receipt['AMP_skip'];self.skips+=int(skip);self.updates+=int(receipt['accepted']);self.rejections+=int(not skip and not receipt['accepted'])
            record=dict(attempt=self.attempts,updates=self.updates,rows=ids,micros=micros,bytes=sum(t['source_bytes'] for t in packets),
                loss=sum(v['weight']*v['loss'] for v in micros),AMP_skip=skip,rejected=(not skip and not receipt['accepted']),update_receipt=receipt,scale=self.scaler.get_scale(),grad_norm=float(norm) if torch.isfinite(norm) else None,
                GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0,clip_and_Adam_calls=1,Exact_entries=0)
            self.history.append(record);self.optimizer.zero_grad(set_to_none=True);self.inflight=None
            if record['GPU_peak_bytes']>12*1024**3 or self.skips>=8 or sum(h['AMP_skip'] for h in self.history[-16:])>=3 or (not skip and not torch.isfinite(norm)):raise FloatingPointError('V80 AMP/numeric/GPU stop')
            return record
        except BaseException:
            # Preserve consumed group/micro/call/exposure state, never replay it.
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def export(self):
        state=super().export();state.update(anchor_indices=self.anchor_indices,anchor_refs=copy.deepcopy(self.anchor_refs),bank_ref=copy.deepcopy(self.bank_ref),bank_summary=copy.deepcopy(self.bank_summary),reference_final_ref=copy.deepcopy(self.reference_final_ref),rejections=self.rejections,reference_patch_work=self.reference_patch_work,schema=SCHEMA,objective=OBJECTIVE,storage_delta_names=self.storage_delta_names,frozen_head_prefix_SHA=self.frozen_head_prefix_SHA,frozen_except_last2_SHA=self.frozen_except_last2_SHA);return state

    def restore(self,s):
        if s['anchor_indices']!=self.anchor_indices:raise ValueError('TRAIN reference selection changed')
        self.anchor_refs=copy.deepcopy(s['anchor_refs']);self.bank_ref=copy.deepcopy(s['bank_ref']);self.bank_summary=copy.deepcopy(s['bank_summary']);self.reference_final_ref=copy.deepcopy(s['reference_final_ref']);self.rejections=s['rejections'];self.reference_patch_work=s['reference_patch_work'];self._bank=None
        if not 0<=len(self.anchor_refs)<=len(self.anchor_indices) or {r['index'] for r in self.anchor_refs}!={i for i in self.anchor_indices if i<s['cache_cursor']}:raise ValueError('partial TRAIN feature cursor')
        if s['attempts'] and self.bank_ref is None:raise ValueError('missing TRAIN reference bank')
        if s['storage_delta_names']!=self.storage_delta_names or s['frozen_head_prefix_SHA']!=self.frozen_head_prefix_SHA or s['frozen_except_last2_SHA']!=self.frozen_except_last2_SHA:raise ValueError('V80 frozen prefix/storage scope identity')
        pending=s['inflight'] is not None and s['inflight']['stage'].startswith('head_')
        if (s['schema']!=SCHEMA or s['objective']!=OBJECTIVE or s['objective_report']!=self.objective_report or not torch.equal(s['symbol_weights'],self.symbol_weights.cpu())
            or s['plan_sha256']!=self.plan_sha or s['exact_policy']!=EXACT_POLICY or s['orders']!=self.orders or s['optimizer_names']!=self.optimizer_names or s['parent_reference']!=self.budget.parent_reference
            or s['input_capacity']!=self.model.encoder.input_capacity or s['embedding_sha']!=self.embedding_sha or s['teacher_sha']!=self.teacher_sha or s['gradients']
            or s['cache_cursor']!=len(s['cache_refs']) or s['cache_cursor']>len(self.rows) or s['cursor']!=s['attempts'] or [h['rows'] for h in s['history']]!=self.orders[:len(s['history'])]
            or len(s['history'])!=s['attempts']-int(pending) or (s['attempts'] and s['cache_cursor']!=len(self.rows)) or s['new_dev_refs']):raise ValueError('V80 exact phase/group recovery identity')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for key in ('cache_cursor','cache_refs','new_dev_refs','attempts','updates','cursor','skips','phase','history','exposure','evaluations','inflight','teacher_GPU_peak_bytes'):setattr(self,key,copy.deepcopy(s[key]))
        self.budget.calls=copy.deepcopy(s['calls']);self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng']);self.budget.block_work=s['long_block_work'];self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('initial teacher mismatch')
        for i in range(self.cache_cursor):self.packet(i)
        for r in self.anchor_refs:self._feature(r)
        if self.bank_ref is not None:self.load_bank()
        if self.updates+self.skips+self.rejections!=len(self.history):raise ValueError('accepted/rejected/skip ledger')
        torch.set_rng_state(s['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))


    def _feature(self,ref):
        if support.sha(ref['features'])!=ref['SHA'] or support.sha(ref['targets'])!=ref['targets_SHA']:raise ValueError('TRAIN reference file changed')
        feature=torch.load(ref['features'],map_location='cpu',weights_only=True);i=ref['index'];t=self.packet(i)
        if set(feature)!={'schema','hidden','source_id','cache_key','embedding_sha','frozen_prefix_sha'} or feature['schema']!='V80_TRAIN_FROZEN_PREFIX_ONLY':raise ValueError('TRAIN hidden feature schema')
        if feature['source_id']!=self.rows[i]['id'] or feature['cache_key']!=t['cache_key'] or feature['embedding_sha']!=self.embedding_sha or feature['frozen_prefix_sha']!=self.frozen_head_prefix_SHA:raise ValueError('TRAIN hidden epoch')
        h=feature['hidden']
        if h.shape!=(len(t['labels']),self.model.readout.readout.net[2].in_features) or not torch.isfinite(h).all():raise ValueError('TRAIN hidden shape/numeric')
        return h,t

    @torch.no_grad()
    def initialize_reference_bank(self):
        if self.inflight is not None or self.attempts or self.cache_cursor!=len(self.rows) or self.bank_ref is not None:raise RuntimeError('one complete TRAIN reference initialization')
        self.inflight=dict(stage='reference_bank_build_requested');mapping={r['index']:r for r in self.anchor_refs}
        if set(mapping)!=set(self.anchor_indices):raise ValueError('all selected TRAIN sources required')
        pieces=[self._feature(mapping[i]) for i in self.anchor_indices]
        h=torch.cat([x[0] for x in pieces]);y=torch.cat([x[1]['labels'] for x in pieces]);critical=torch.cat([x[1]['critical_mask'] for x in pieces]);rowcorrect=torch.cat([x[1]['teacher_correct'] for x in pieces])
        if len(h)!=self.budget.anchor_hidden_rows:raise ValueError('fixed selected TRAIN hidden row budget')
        self.budget.charge('reference_linear');self.reference_patch_work+=len(h)
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            z=self.model.readout.readout.net[2](h.to(self.device)).reshape(len(h),16,258)
        initial_argmax=z.argmax(-1).cpu();correct=(initial_argmax==y)&(y!=257)
        if not torch.isfinite(z).all():raise FloatingPointError('initial TRAIN reference')
        directory=self.budget.output/'TRAIN_reference';directory.mkdir(exist_ok=True)
        f=directory/'bank.features.pt';g=directory/'bank.targets.pt'
        fp=dict(schema='V80_TRAIN_REFERENCE_FEATURES',hidden=h,source_indices=self.anchor_indices,embedding_sha=self.embedding_sha,frozen_prefix_sha=self.frozen_head_prefix_SHA)
        tp=dict(schema='V80_TRAIN_REFERENCE_LOSS_ONLY',labels=y,critical_mask=critical,initial_correct=correct,initial_argmax=initial_argmax.to(torch.int16),source_indices=self.anchor_indices,initial_teacher_sha=self.teacher_sha)
        fs=save_tensor(f,fp,self.budget);gs=save_tensor(g,tp,self.budget)
        self.bank_ref=dict(features=str(f),features_SHA=fs,targets=str(g),targets_SHA=gs)
        self.bank_summary=dict(sources=len(self.anchor_indices),hidden_rows=len(h),hidden_value_bytes=h.numel()*h.element_size(),protected_positions=int(correct.sum()),
            protected_critical=int((correct&critical).sum()),protected_noncritical_EOS=int((correct&~critical).sum()),rowwise_teacher_correct_flag_disagreements=int(((rowcorrect!=correct)&(y!=257)).sum()),
            all_source_conditions_are_TRAIN_only=True,labels_used_as_feature=False,not_all_wrong_teacher_argmax_or_rowwise_logit_parity_verified=True)
        self.inflight=None;self.load_bank()

    def load_bank(self):
        if self.bank_ref is None:raise ValueError('no TRAIN bank')
        r=self.bank_ref
        if support.sha(r['features'])!=r['features_SHA'] or support.sha(r['targets'])!=r['targets_SHA']:raise ValueError('TRAIN bank SHA')
        f=torch.load(r['features'],map_location='cpu',weights_only=True);t=torch.load(r['targets'],map_location='cpu',weights_only=True)
        if set(f)!={'schema','hidden','source_indices','embedding_sha','frozen_prefix_sha'} or set(t)!={'schema','labels','critical_mask','initial_correct','initial_argmax','source_indices','initial_teacher_sha'}:raise ValueError('features and loss-only targets separated')
        if f['source_indices']!=self.anchor_indices or t['source_indices']!=self.anchor_indices or f['embedding_sha']!=self.embedding_sha or f['frozen_prefix_sha']!=self.frozen_head_prefix_SHA or t['initial_teacher_sha']!=self.teacher_sha:raise ValueError('TRAIN bank identity')
        h=f['hidden'];y=t['labels'];mask=t['initial_correct'];critical=t['critical_mask']
        pieces=[self._feature({r['index']:r for r in self.anchor_refs}[i]) for i in self.anchor_indices]
        if not torch.equal(h,torch.cat([v[0] for v in pieces])) or not torch.equal(y,torch.cat([v[1]['labels'] for v in pieces])) or not torch.equal(critical,torch.cat([v[1]['critical_mask'] for v in pieces])):raise ValueError('TRAIN bank source values changed')
        if not torch.equal(mask,(t['initial_argmax'].long()==y)&(y!=257)):raise ValueError('stored initial TRAIN reference mask')
        if len(h)!=self.budget.anchor_hidden_rows or mask.shape!=y.shape or mask.dtype!=torch.bool or (mask&(y==257)).any():raise ValueError('TRAIN bank mask shape')
        self._bank=(h.to(self.device),y.to(self.device),critical.to(self.device),mask.to(self.device))

    @torch.no_grad()
    def reference_decision(self):
        if self._bank is None:self.load_bank()
        h,y,critical,mask=self._bank;self.budget.charge('reference_linear');self.reference_patch_work+=len(h)
        if self.reference_patch_work>self.budget.anchor_hidden_rows*(4*len(self.orders)+2):raise ValueError('fixed reference patch work cap')
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            z=self.model.readout.readout.net[2](h).reshape(len(h),16,258)
        if not torch.isfinite(z).all():raise FloatingPointError('TRAIN candidate reference logits')
        flipped=(z.argmax(-1)!=y)&mask
        c=int((flipped&critical).sum());n=int((flipped&~critical).sum())
        return TrainDecision(c+n==0,dict(new_critical_flips=c,new_noncritical_EOS_flips=n,protected_positions=int(mask.sum()),reference_hidden_rows=len(h)))


    @torch.no_grad()
    def finalize_reference_bank(self):
        if self.inflight is not None or self.reference_final_ref is not None or self.attempts!=len(self.orders):raise ValueError('one terminal TRAIN reference record')
        h,y,critical,mask=self._bank;self.budget.charge('reference_linear');self.reference_patch_work+=len(h)
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            z=self.model.readout.readout.net[2](h).reshape(len(h),16,258)
        pred=z.argmax(-1);flips=int(((pred!=y)&mask).sum())
        if flips or not torch.isfinite(z).all():raise FloatingPointError('final TRAIN reference preservation')
        packet=dict(schema='V80_FINAL_TRAIN_REFERENCE_PREDICTIONS',argmax=pred.to(torch.int16).cpu(),features_SHA=self.bank_ref['features_SHA'],targets_SHA=self.bank_ref['targets_SHA'],initial_protected_flips=flips)
        file=self.budget.output/'TRAIN_reference/final_predictions.pt';sha=save_tensor(file,packet,self.budget)
        self.reference_final_ref=dict(path=str(file),SHA=sha,initial_protected_flips=flips)
