"""Finite source-cache/teacher phase then seven-tensor decoder learning.

The encoder and role binding are immutable. Cache features are bound memory
values; labels and teacher probabilities are in a separate loss-only packet.
"""
import copy,gc,hashlib,random
from pathlib import Path
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from research.fuzzy_factual_objective_v70 import critical_mask,training_weights
from research.frozen_verbalizer_objective_v72 import freeze_extraction_and_binding,decoder_losses,OBJECTIVE
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support

SCHEMA='fdt_frozen_verbalizer_v72_fullstate'

class FrozenEngine:
    def __init__(self,model,rows,orders,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.rows=rows;self.orders=orders;self.budget=budget;self.plan_sha=plan_sha;self.device=device
        self.names=freeze_extraction_and_binding(model);named=dict(model.named_parameters());self.optimizer_names=[self.names]
        self.optimizer=torch.optim.AdamW([named[n] for n in self.names],lr=3e-5,weight_decay=.01)
        self.scaler=torch.amp.GradScaler(device,init_scale=1024.)
        lookup,self.objective_report=training_weights(rows);self.objective_report.update(objective=OBJECTIVE,anchor_coefficient=.05,decoder_LR=3e-5)
        self.symbol_weights=torch.tensor(lookup,device=device,dtype=torch.float32)
        self.embedding_sha=support.state_sha({k:v for k,v in model.state_dict().items() if k not in self.names})
        self.teacher_sha=support.state_sha(model.readout.readout.state_dict());self.cache_cursor=0;self.cache_refs=[]
        self.new_dev_refs=[]
        self.attempts=self.updates=self.skips=self.cursor=0;self.phase='cache';self.history=[];self.evaluations={};self.inactive_cuda_rng=[]
        self.exposure=dict(encoder_rows=0,teacher_rows=0,encoder_UTF8_bytes=0,head_rows=0,head_UTF8_bytes=0,head_symbols=0,head_critical_positions=0)
        if any(r['id'] not in budget.critical_targets for r in rows):raise ValueError('missing separated TRAIN targets')
    def verify_frozen(self):
        current=support.state_sha({k:v for k,v in self.model.state_dict().items() if k not in self.names})
        if current!=self.embedding_sha:raise ValueError('encoder/binding/prior/reasoner changed')
    def source_key(self,row):
        return support.state_sha((self.embedding_sha,row['id'],self.budget.data_sha,hashlib.sha256(row['prompt'].encode()).hexdigest()))
    @torch.no_grad()
    def encode_packet(self,row,spans):
        self.budget.check();self.model.eval();b=row['prompt'].encode();x=pack_prompts([row['prompt']],len(b),self.device)
        self.budget.charge('encode');self.budget.charge('readout')
        if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            c,ep=self.model.extract(x,source_ids=(row['id'],),source_versions=(self.budget.data_sha,),model_epoch=self.embedding_sha)
            z=self.model.verbalize(c)
        if ep.memories.exact.valid.any() or not torch.isfinite(z).all():raise ValueError('numeric/Exact cache contract')
        y=torch.full(z.shape[:2],257,device=self.device,dtype=torch.long);y.flatten()[:len(b)]=torch.tensor(list(b),device=self.device);y.flatten()[len(b)]=256
        mask=critical_mask(y,spans,len(b));key=self.source_key(row)
        f=dict(schema='bound_memory_cache_v72',semantic=c.reader_tensor().cpu(),memory_revisions=c.memory_revisions,
            embedding_sha=self.embedding_sha,source_id=row['id'],source_version=self.budget.data_sha,source_SHA=hashlib.sha256(b).hexdigest(),cache_key=key)
        t=dict(schema='TRAIN_teacher_targets_v72',labels=y.cpu(),critical_mask=mask.cpu(),
            teacher_gold_probability=z.float().softmax(-1).gather(-1,y[...,None]).squeeze(-1).cpu(),teacher_correct=(z.argmax(-1)==y).cpu(),
            source_bytes=len(b),cache_key=key)
        if self.device=='cuda' and torch.cuda.max_memory_allocated()>12*1024**3:raise MemoryError('GPU12GiB extraction cap')
        return f,t,z.float().cpu()
    def cache_next(self):
        if self.attempts or self.cache_cursor>=len(self.rows):raise StopIteration('finite cache phase')
        row=self.rows[self.cache_cursor];f,t,_=self.encode_packet(row,self.budget.critical_targets[row['id']])
        directory=self.budget.output/'cache';directory.mkdir(exist_ok=True,parents=True)
        fp=directory/f'{self.cache_cursor:04d}.features.pt';tp=directory/f'{self.cache_cursor:04d}.targets.pt'
        fsha=save_tensor(fp,f,self.budget);tsha=save_tensor(tp,t,self.budget)
        self.cache_refs.append(dict(index=self.cache_cursor,id=row['id'],features=str(fp),targets=str(tp),features_SHA=fsha,targets_SHA=tsha,cache_key=f['cache_key'],
            GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0))
        self.cache_cursor+=1;self.exposure['encoder_rows']+=1;self.exposure['teacher_rows']+=1;self.exposure['encoder_UTF8_bytes']+=t['source_bytes']
        if self.cache_cursor==len(self.rows):
            self.verify_frozen()
            if support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('teacher changed during cache phase')
            self.phase='head_training'
    def packet(self,index,verify=True):
        ref=self.cache_refs[index]
        if verify and (support.sha(ref['features'])!=ref['features_SHA'] or support.sha(ref['targets'])!=ref['targets_SHA']):raise ValueError('cache changed')
        f=torch.load(ref['features'],map_location='cpu',weights_only=True);t=torch.load(ref['targets'],map_location='cpu',weights_only=True)
        if f['embedding_sha']!=self.embedding_sha or f['cache_key']!=self.source_key(self.rows[index]) or f['cache_key']!=t['cache_key'] or ref['cache_key']!=f['cache_key']:raise ValueError('cache embedding/source epoch')
        if set(f)!={'schema','semantic','memory_revisions','embedding_sha','source_id','source_version','source_SHA','cache_key'}:raise ValueError('source/target leakage into memory features')
        row=self.rows[index];n=len(row['prompt'].encode());slots=(n+15)//16
        if (f['source_id']!=row['id'] or f['source_version']!=self.budget.data_sha or f['source_SHA']!=hashlib.sha256(row['prompt'].encode()).hexdigest()
            or f['semantic'].shape!=(1,slots+2,self.model.encoder.c.z_dim) or t['source_bytes']!=n or t['labels'].shape!=(slots+1,16)
            or any(t[k].shape!=t['labels'].shape for k in ('critical_mask','teacher_gold_probability','teacher_correct'))):raise ValueError('cache source/shape provenance')
        return f,t
    def step(self,force_overflow=False):
        self.budget.check()
        if self.cache_cursor!=len(self.rows) or self.attempts>=len(self.orders):raise StopIteration('cache incomplete or finite epoch ended')
        if getattr(self.budget,'eval_rows',[]) and len(self.new_dev_refs)!=len(self.budget.eval_rows):raise ValueError('missing initial DEV cache references')
        ids=self.orders[self.attempts];assert len(ids)==1;f,t=self.packet(ids[0]);self.model.eval();self.model.readout.readout.train()
        c=DifferentiableCodecCommit.from_semantic(f['semantic'].to(self.device),f['memory_revisions'],self.embedding_sha)
        y=t['labels'].to(self.device);mask=t['critical_mask'].to(self.device);p0=t['teacher_gold_probability'].to(self.device);correct=t['teacher_correct'].to(self.device)
        self.attempts+=1;self.cursor+=1;self.exposure['head_rows']+=1;self.exposure['head_UTF8_bytes']+=t['source_bytes'];self.exposure['head_symbols']+=t['source_bytes']+1;self.exposure['head_critical_positions']+=int(mask.sum())
        self.optimizer.zero_grad(set_to_none=True);self.budget.charge('readout')
        if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            z=self.model.verbalize(c);loss,parts=decoder_losses(z,y,self.symbol_weights,mask,p0,correct)
        if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('bounded head loss')
        self.scaler.scale(loss).backward();named=dict(self.model.named_parameters())
        if force_overflow:named[self.names[0]].grad.fill_(float('inf'))
        self.scaler.unscale_(self.optimizer)
        if any(named[n].grad is None for n in self.names) or any(v.grad is not None for n,v in named.items() if n not in self.names):raise ValueError('exact seven readout gradient ownership')
        norm=torch.nn.utils.clip_grad_norm_([named[n] for n in self.names],1.)
        scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update();skip=self.scaler.get_scale()<scale
        self.skips+=int(skip);self.updates+=int(not skip)
        record=dict(attempt=self.attempts,updates=self.updates,rows=ids,bytes=t['source_bytes'],loss=float(loss.detach()),AMP_skip=skip,scale=self.scaler.get_scale(),
            grad_norm=float(norm) if torch.isfinite(norm) else None,GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0,Exact_entries=0)
        record.update({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()});self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
        if record['GPU_peak_bytes']>12*1024**3:raise MemoryError('GPU12GiB head cap')
        if self.skips>=8 or sum(h['AMP_skip'] for h in self.history[-16:])>=3 or (not skip and not torch.isfinite(norm)):raise FloatingPointError('AMP/numeric head stop')
        return record
    def export(self):
        return dict(schema=SCHEMA,objective=OBJECTIVE,objective_report=self.objective_report,symbol_weights=self.symbol_weights.cpu(),plan_sha256=self.plan_sha,exact_policy=EXACT_POLICY,
            parent_reference=self.budget.parent_reference,model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},core=self.model.encoder.c.__dict__,input_capacity=self.model.encoder.input_capacity,
            optimizer=self.optimizer.state_dict(),optimizer_names=self.optimizer_names,scaler=self.scaler.state_dict(),orders=self.orders,embedding_sha=self.embedding_sha,teacher_sha=self.teacher_sha,
            cache_cursor=self.cache_cursor,cache_refs=copy.deepcopy(self.cache_refs),new_dev_refs=copy.deepcopy(self.new_dev_refs),attempts=self.attempts,updates=self.updates,cursor=self.cursor,skips=self.skips,phase=self.phase,
            history=self.history,exposure=self.exposure,evaluations=self.evaluations,calls=self.budget.calls,gradients={},
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else self.inactive_cuda_rng,python_rng=random.getstate(),numpy_rng=numpy_state(),
            Exact_entries=0,copy_bytes=0,denoise_calls=0)
    def restore(self,s):
        if (s['schema']!=SCHEMA or s['objective']!=OBJECTIVE or s['objective_report']!=self.objective_report or not torch.equal(s['symbol_weights'],self.symbol_weights.cpu())
            or s['plan_sha256']!=self.plan_sha or s['exact_policy']!=EXACT_POLICY or s['orders']!=self.orders or s['optimizer_names']!=self.optimizer_names or s['parent_reference']!=self.budget.parent_reference
            or s['input_capacity']!=self.model.encoder.input_capacity or s['embedding_sha']!=self.embedding_sha or s['teacher_sha']!=self.teacher_sha or s['gradients']
            or s['cache_cursor']!=len(s['cache_refs']) or s['cache_cursor']>len(self.rows) or s['cursor']!=s['attempts'] or [h['rows'] for h in s['history']]!=self.orders[:s['attempts']]):raise ValueError('full phase recovery identity')
        if s['attempts'] and s['cache_cursor']!=len(self.rows):raise ValueError('head phase without complete cache')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for key in ('cache_cursor','cache_refs','new_dev_refs','attempts','updates','cursor','skips','phase','history','exposure','evaluations'):setattr(self,key,copy.deepcopy(s[key]))
        self.budget.calls=copy.deepcopy(s['calls']);self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng'])
        self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('restored initial teacher mismatch')
        for index in range(self.cache_cursor):self.packet(index)
        for index,ref in enumerate(self.new_dev_refs):
            if support.sha(ref['features'])!=ref['features_SHA']:raise ValueError('DEV cache changed')
            f=torch.load(ref['features'],map_location='cpu',weights_only=True)
            if f['embedding_sha']!=self.embedding_sha or f['cache_key']!=self.source_key(self.budget.eval_rows[index]):raise ValueError('DEV cache source epoch')
        torch.set_rng_state(s['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
