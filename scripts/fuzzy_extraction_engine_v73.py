"""Initial loss-only teacher packets, then live source extraction updates."""
import copy,hashlib,random
from pathlib import Path
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from research.fuzzy_extraction_v73 import enable_fixed_readout_extraction
from research.frozen_verbalizer_objective_v72 import decoder_losses,OBJECTIVE as LOSS
from research.fuzzy_factual_objective_v70 import critical_mask,training_weights
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support
SCHEMA='fdt_live_fuzzy_extraction_v73_fullstate'
OBJECTIVE='fixed_V72_final_readout7_live_upstream23:'+LOSS

class ExtractionEngine:
    def __init__(self,model,rows,orders,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.rows=rows;self.orders=orders;self.budget=budget;self.plan_sha=plan_sha;self.device=device
        self.names=enable_fixed_readout_extraction(model);named=dict(model.named_parameters());groups=[]
        for prefix,lr in (('encoder.core.language.blocks.',1e-6),('encoder.core.',3e-6),('readout.role_projection.',3e-6)):
            names=[n for n in self.names if n.startswith(prefix) and not any(n in g['names'] for g in groups)]
            if names:groups.append(dict(names=names,lr=lr))
        assert sorted(sum([g['names'] for g in groups],[]))==sorted(self.names)
        self.optimizer_names=[g['names'] for g in groups]
        self.optimizer=torch.optim.AdamW([dict(params=[named[n] for n in g['names']],lr=g['lr']) for g in groups],weight_decay=.01)
        self.scaler=torch.amp.GradScaler(device,init_scale=1024.)
        lookup,self.objective_report=training_weights(rows);self.objective_report.update(objective=OBJECTIVE,anchor_coefficient=.05,LR_groups=[g['lr'] for g in groups])
        self.symbol_weights=torch.tensor(lookup,device=device,dtype=torch.float32)
        self.frozen_sha=support.state_sha({k:v for k,v in model.state_dict().items() if k not in self.names});self.teacher_model_sha=support.state_sha(model.state_dict())
        self.teacher_cursor=0;self.teacher_refs=[];self.attempts=self.updates=self.cursor=self.skips=0;self.history=[];self.evaluations={};self.phase='initial_teacher';self.inactive_cuda_rng=[];self.inflight=None
        self.exposure=dict(teacher_rows=0,teacher_UTF8_bytes=0,live_rows=0,live_UTF8_bytes=0,live_symbols=0,live_critical_positions=0)
    def verify_frozen(self):
        if support.state_sha({k:v for k,v in self.model.state_dict().items() if k not in self.names})!=self.frozen_sha:raise ValueError('fixed readout/prior/reasoner changed')
    def teacher_key(self,row):
        return support.state_sha((self.teacher_model_sha,row['id'],self.budget.data_sha,hashlib.sha256(row['prompt'].encode()).hexdigest()))
    def target(self,z,row):
        b=row['prompt'].encode();y=torch.full(z.shape[:2],257,dtype=torch.long,device=z.device)
        y.flatten()[:len(b)]=torch.tensor(list(b),device=z.device);y.flatten()[len(b)]=256;return y
    @torch.no_grad()
    def teacher_next(self):
        if self.attempts or self.teacher_cursor>=len(self.rows):raise StopIteration('finite immutable teacher phase')
        row=self.rows[self.teacher_cursor];self.budget.check();self.model.eval();b=row['prompt'].encode();x=pack_prompts([row['prompt']],len(b),self.device)
        self.budget.charge('encode');self.budget.charge('readout')
        if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            z,c,ep=self.model(x,source_ids=(row['id'],),source_versions=(self.budget.data_sha,),model_epoch=self.teacher_model_sha)
        if ep.memories.exact.valid.any() or not torch.isfinite(z).all():raise ValueError('teacher numeric/Exact')
        y=self.target(z,row);mask=critical_mask(y,self.budget.critical_targets[row['id']],len(b));key=self.teacher_key(row)
        # This is a loss packet only. It contains no reusable Commit or source H.
        packet=dict(schema='TRAIN_loss_teacher_v73',labels=y.cpu(),critical_mask=mask.cpu(),teacher_gold_probability=z.float().softmax(-1).gather(-1,y[...,None]).squeeze(-1).cpu(),
            teacher_correct=(z.argmax(-1)==y).cpu(),source_id=row['id'],source_SHA=hashlib.sha256(b).hexdigest(),source_bytes=len(b),source_version=self.budget.data_sha,teacher_model_SHA=self.teacher_model_sha,teacher_key=key)
        folder=self.budget.output/'teacher_targets';folder.mkdir(parents=True,exist_ok=True);file=folder/f'{self.teacher_cursor:04d}.pt';digest=save_tensor(file,packet,self.budget)
        peak=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0
        if peak>12*1024**3:raise MemoryError('teacherGPU12GiB')
        self.teacher_refs.append(dict(index=self.teacher_cursor,file=str(file),SHA=digest,key=key,GPU_peak_bytes=peak));self.teacher_cursor+=1
        self.exposure['teacher_rows']+=1;self.exposure['teacher_UTF8_bytes']+=len(b)
        if self.teacher_cursor==len(self.rows):
            if support.state_sha(self.model.state_dict())!=self.teacher_model_sha:raise ValueError('teacher weights changed')
            self.phase='teacher_complete'
    def packet(self,index):
        ref=self.teacher_refs[index];row=self.rows[index]
        if support.sha(ref['file'])!=ref['SHA']:raise ValueError('teacher packet changed')
        s=torch.load(ref['file'],map_location='cpu',weights_only=True);n=len(row['prompt'].encode());shape=((n+15)//16+1,16)
        if (set(s)!={'schema','labels','critical_mask','teacher_gold_probability','teacher_correct','source_id','source_SHA','source_bytes','source_version','teacher_model_SHA','teacher_key'}
            or s['teacher_key']!=ref['key'] or s['teacher_key']!=self.teacher_key(row) or s['source_id']!=row['id'] or s['source_SHA']!=hashlib.sha256(row['prompt'].encode()).hexdigest()
            or s['source_bytes']!=n or s['source_version']!=self.budget.data_sha or s['teacher_model_SHA']!=self.teacher_model_sha
            or any(s[k].shape!=shape for k in ('labels','critical_mask','teacher_gold_probability','teacher_correct'))):raise ValueError('teacher packet source/epoch/shape')
        return s
    def live_loss(self,index):
        row=self.rows[index];t=self.packet(index);b=row['prompt'].encode();x=pack_prompts([row['prompt']],len(b),self.device)
        self.model.train();self.model.readout.readout.eval();self.budget.charge('encode');self.budget.charge('readout')
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            z,c,ep=self.model(x,source_ids=(row['id'],),source_versions=(self.budget.data_sha,),model_epoch=self.plan_sha+':live:'+str(self.attempts))
            y=t['labels'].to(self.device)
            if not torch.equal(y,self.target(z,row)) or ep.memories.exact.valid.any():raise ValueError('live labels/source/Exact')
            loss,parts=decoder_losses(z,y,self.symbol_weights,t['critical_mask'].to(self.device),t['teacher_gold_probability'].to(self.device),t['teacher_correct'].to(self.device))
        return loss,parts,t
    def step(self,force_overflow=False):
        self.budget.check()
        if self.inflight is not None:raise RuntimeError('interrupted attempt preserved; no automatic replay/update')
        if self.teacher_cursor!=len(self.rows) or self.attempts>=len(self.orders):raise StopIteration('teacher incomplete or epoch closed')
        if getattr(self.budget,'baseline_required',False) and 'new_baseline' not in self.evaluations:raise ValueError('new held baseline not committed')
        ids=self.orders[self.attempts];assert len(ids)==1;self.attempts+=1;self.cursor+=1;self.optimizer.zero_grad(set_to_none=True)
        self.inflight=dict(attempt=self.attempts,rows=ids,stage='source_started',source_bytes=len(self.rows[ids[0]]['prompt'].encode()),exposure_accounted=False)
        if self.device=='cuda':torch.cuda.reset_peak_memory_stats()
        loss,parts,t=self.live_loss(ids[0]);n=t['source_bytes'];self.exposure['live_rows']+=1;self.exposure['live_UTF8_bytes']+=n;self.exposure['live_symbols']+=n+1;self.exposure['live_critical_positions']+=int(t['critical_mask'].sum())
        self.inflight.update(stage='loss_computed',exposure_accounted=True)
        if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('live bounded loss')
        self.scaler.scale(loss).backward();named=dict(self.model.named_parameters())
        if force_overflow:named[self.names[0]].grad.fill_(float('inf'))
        self.scaler.unscale_(self.optimizer)
        if any(named[n].grad is None for n in self.names) or any(v.grad is not None for n,v in named.items() if n not in self.names):raise ValueError('exact upstream23 gradient ownership')
        norm=torch.nn.utils.clip_grad_norm_([named[n] for n in self.names],1.);scale=self.scaler.get_scale();self.inflight['stage']='optimizer_attempted';self.scaler.step(self.optimizer);self.scaler.update();skip=self.scaler.get_scale()<scale
        self.skips+=int(skip);self.updates+=int(not skip)
        record=dict(attempt=self.attempts,updates=self.updates,rows=ids,bytes=n,loss=float(loss.detach()),AMP_skip=skip,scale=self.scaler.get_scale(),grad_norm=float(norm) if torch.isfinite(norm) else None,
            GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else 0,Exact_entries=0)
        record.update({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()});self.history.append(record);self.inflight=None;self.optimizer.zero_grad(set_to_none=True)
        if record['GPU_peak_bytes']>12*1024**3:raise MemoryError('liveGPU12GiB')
        if self.skips>=8 or sum(h['AMP_skip'] for h in self.history[-16:])>=3 or (not skip and not torch.isfinite(norm)):raise FloatingPointError('AMP/numeric stop')
        return record
    def export(self):
        return dict(schema=SCHEMA,objective=OBJECTIVE,objective_report=self.objective_report,symbol_weights=self.symbol_weights.cpu(),plan_sha256=self.plan_sha,exact_policy=EXACT_POLICY,
            parent_reference=self.budget.parent_reference,model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},core=self.model.encoder.c.__dict__,input_capacity=self.model.encoder.input_capacity,
            optimizer=self.optimizer.state_dict(),optimizer_names=self.optimizer_names,scaler=self.scaler.state_dict(),orders=self.orders,frozen_sha=self.frozen_sha,teacher_model_sha=self.teacher_model_sha,
            teacher_cursor=self.teacher_cursor,teacher_refs=copy.deepcopy(self.teacher_refs),attempts=self.attempts,updates=self.updates,cursor=self.cursor,skips=self.skips,phase=self.phase,history=self.history,
            exposure=self.exposure,evaluations=self.evaluations,calls=self.budget.calls,long_block_work=getattr(self.budget,'block_work',0),inflight=copy.deepcopy(self.inflight),gradients={},torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else self.inactive_cuda_rng,python_rng=random.getstate(),numpy_rng=numpy_state(),Exact_entries=0,copy_bytes=0,denoise_calls=0)
    def restore(self,s):
        if (s['schema']!=SCHEMA or s['objective']!=OBJECTIVE or s['objective_report']!=self.objective_report or not torch.equal(s['symbol_weights'],self.symbol_weights.cpu())
            or s['plan_sha256']!=self.plan_sha or s['orders']!=self.orders or s['optimizer_names']!=self.optimizer_names or s['parent_reference']!=self.budget.parent_reference
            or s['frozen_sha']!=self.frozen_sha or s['teacher_model_sha']!=self.teacher_model_sha or s['input_capacity']!=self.model.encoder.input_capacity or s['gradients']
            or s['teacher_cursor']!=len(s['teacher_refs']) or s['teacher_cursor']>len(self.rows) or s['cursor']!=s['attempts'] or [h['rows'] for h in s['history']]!=self.orders[:len(s['history'])]
            or len(s['history'])!=s['attempts']-int(s['inflight'] is not None)
            or (s['attempts'] and s['teacher_cursor']!=len(self.rows))):raise ValueError('actual phase restore identity')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for k in ('teacher_cursor','teacher_refs','attempts','updates','cursor','skips','phase','history','exposure','evaluations','inflight'):setattr(self,k,copy.deepcopy(s[k]))
        self.budget.calls=copy.deepcopy(s['calls']);self.budget.block_work=s['long_block_work'];self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng']);self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.state_dict())!=self.teacher_model_sha:raise ValueError('teacher phase model drift')
        for i in range(self.teacher_cursor):self.packet(i)
        torch.set_rng_state(s['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
