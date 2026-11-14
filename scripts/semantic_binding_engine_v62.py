"""Fresh semantic-extraction optimizer; closed legacy trajectories are references."""
import copy,random
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.semantic_binding_v59 import MemoryTokens
from research.semantic_binding_objective_v62 import training_loss,OBJECTIVE
import text_generation_support_v1 as support

SCHEMA='fdt_semantic_binding_v62_fullstate'

def select_packet(features,ids,device):
    keys=('mu','contextual_values','exact_keys','exact_values','valid','starts','ends')
    tensors={k:features[k].index_select(0,ids).to(device) for k in keys}
    revisions=tuple(features['source_revisions'][i] for i in ids.tolist())
    return MemoryTokens(**tensors,source_revisions=revisions,source_epoch=features['source_epoch'])

class SemanticEngine:
    def __init__(self,model,features,targets,orders,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.features=features;self.targets=targets;self.orders=orders;self.budget=budget;self.plan_sha=plan_sha;self.device=device
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4,weight_decay=.01)
        self.scaler=torch.amp.GradScaler('cuda' if device=='cuda' else 'cpu',init_scale=1024.)
        self.prior_sha=support.state_sha({'prior':self.model.persistent_prior})
        self.attempts=self.updates=self.cursor=self.skips=0;self.history=[];self.cache_refs={};self.inactive_cuda_rng=[]
        self.baseline_reference=getattr(budget,'baseline_reference',None)
        self.phase='before_source_extraction';self.extract_cursor={'train':0,'dev':0};self.extract_shards={'train':[],'dev':[]}
        self.exposure=dict(row_selections=0,cached_source_UTF8_bytes=0)

    def step(self,force_overflow=False):
        self.budget.check()
        if self.attempts>=len(self.orders):raise StopIteration('finite semantic epochs ended')
        order=self.orders[self.attempts];ids=torch.tensor(order);packet=select_packet(self.features,ids,self.device)
        targets={k:self.targets[k][ids].to(self.device) for k in ('spans','relation','polarity','quantity')}
        meaning_ids=self.targets['meaning_ids'][ids].to(self.device)
        self.attempts+=1;self.cursor+=len(ids);self.exposure['row_selections']+=len(ids)
        self.exposure['cached_source_UTF8_bytes']+=int(packet.valid.sum())
        self.optimizer.zero_grad(set_to_none=True);self.model.train();self.budget.charge('bind')
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            commit,raw=self.model(packet,self.plan_sha+':'+str(self.attempts))
            loss,parts=training_loss(commit,raw,targets,meaning_ids)
        if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('semantic objective stop')
        self.scaler.scale(loss).backward()
        if force_overflow:next(self.model.parameters()).grad.fill_(float('inf'))
        self.scaler.unscale_(self.optimizer)
        if any(v.grad is None for v in self.model.parameters()):raise ValueError('missing semantic encoder gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
        skip=self.scaler.get_scale()<scale;self.skips+=int(skip);self.updates+=int(not skip)
        record=dict(attempt=self.attempts,updates=self.updates,rows=order,loss=float(loss.detach()),
            losses={k:float(v.detach()) for k,v in parts.items()},AMP_skip=skip,scale=self.scaler.get_scale(),
            grad_norm=float(norm) if torch.isfinite(norm) else None)
        self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
        if sum(h['AMP_skip'] for h in self.history[-16:])>=3 or self.skips>=8:raise FloatingPointError('persistent semantic AMP stop')
        if not skip and not torch.isfinite(norm):raise FloatingPointError('nonfinite applied semantic gradient')
        return record

    def export(self):
        return dict(schema=SCHEMA,plan_sha256=self.plan_sha,parent_reference=self.budget.parent_reference,baseline_reference=self.baseline_reference,
            binder={k:v.detach().cpu() for k,v in self.model.state_dict().items()},prior_sha256=self.prior_sha,
            source_epoch=self.model.source_epoch,binder_hidden=self.model.project.out_features,
            objective_contract=OBJECTIVE,ontology=['give','open','contain'],graph_contract='coupled_spans_fourrole_memory_readout',optimizer=self.optimizer.state_dict(),scaler=self.scaler.state_dict(),
            orders=self.orders,attempts=self.attempts,updates=self.updates,cursor=self.cursor,skips=self.skips,
            history=self.history,exposure=self.exposure,cache_refs=self.cache_refs,calls=self.budget.calls,gradients={},
            phase=self.phase,extract_cursor=self.extract_cursor,extract_shards=self.extract_shards,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else self.inactive_cuda_rng,
            python_rng=random.getstate(),numpy_rng=numpy_state(),old_source_and_language_optimizer_updates=0)

    def restore(self,state,verify_files=True):
        if (state['baseline_reference']!=self.baseline_reference or state['objective_contract']!=OBJECTIVE or state['schema']!=SCHEMA or state['plan_sha256']!=self.plan_sha or state['source_epoch']!=self.model.source_epoch
            or state['graph_contract']!='coupled_spans_fourrole_memory_readout' or state['binder_hidden']!=self.model.project.out_features or state['ontology']!=['give','open','contain']
            or state['parent_reference']!=self.budget.parent_reference or state['orders']!=self.orders or state['gradients']
            or state['prior_sha256']!=self.prior_sha or support.state_sha({'prior':state['binder']['persistent_prior']})!=self.prior_sha
            or state['cursor']!=sum(map(len,self.orders[:state['attempts']]))
            or [h['rows'] for h in state['history']]!=self.orders[:state['attempts']]):raise ValueError('semantic recovery contract')
        if verify_files:
            refs={**state['cache_refs'],self.budget.parent_reference['path']:self.budget.parent_reference['sha256']}
            if 'codec_path' in self.budget.parent_reference:
                refs[self.budget.parent_reference['codec_path']]=self.budget.parent_reference['codec_sha256']
            if 'binder_path' in self.budget.parent_reference:
                refs[self.budget.parent_reference['binder_path']]=self.budget.parent_reference['binder_sha256']
            for name,digest in refs.items():
                if support.sha(name)!=digest:raise ValueError('source/cache reference changed')
        self.model.load_state_dict(state['binder'],strict=True);self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        for key in ('attempts','updates','cursor','skips','history','exposure','cache_refs','phase','extract_cursor','extract_shards'):setattr(self,key,copy.deepcopy(state[key]))
        self.budget.calls=copy.deepcopy(state['calls']);self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        torch.set_rng_state(state['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);n=state['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
