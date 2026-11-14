"""Fresh language-readout Adam; frozen parent and immutable Commit cache are references."""
import copy,random
import numpy as np
import torch
from fdt_local.runner import numpy_state
import text_generation_support_v1 as support

SCHEMA='fdt_frozen_commit_readout_v54_fullstate'

def make_orders(rows,epochs=6,batch=256):
    active=[];offset=0
    for row in rows:
        length=len(row['prompt'].encode())
        active.extend(range(offset,offset+(length+16)//16))
        offset+=(length+15)//16+1
    orders=[]
    for epoch in range(epochs):
        shuffled=list(active);random.Random(54111+epoch).shuffle(shuffled)
        orders.extend([shuffled[i:i+batch] for i in range(0,len(shuffled),batch)])
    return orders,active

class ReadoutEngine:
    def __init__(self,head,features,targets,orders,budget,plan_sha,device='cuda'):
        self.head=head.to(device);self.features=features;self.targets=targets;self.orders=orders
        self.budget=budget;self.plan_sha=plan_sha;self.device=device
        self.optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
        self.scaler=torch.amp.GradScaler('cuda' if device=='cuda' else 'cpu',init_scale=1024.)
        self.attempts=self.updates=self.cursor=self.skips=0;self.history=[];self.cache_refs={}
        self.exposure=dict(selected_patches=0,active_symbols=0,target_bytes=0,EOS=0)
        self.inactive_cuda_rng=[]
        self.phase='before_extraction';self.extract_cursor={'train':0,'dev':0};self.extract_shards={'train':[],'dev':[]}

    def step(self,force_overflow=False):
        self.budget.check()
        if self.attempts>=len(self.orders):raise StopIteration('finite probe epochs ended')
        indices=self.orders[self.attempts];ids=torch.tensor(indices)
        self.attempts+=1;self.cursor+=len(ids);self.exposure['selected_patches']+=len(ids)
        f=self.features;local=f['local'].index_select(0,ids).to(self.device)
        global_views=f['global'].index_select(0,f['doc_index'][ids]).to(self.device)
        terminal=f['terminal'][ids].to(self.device);target=self.targets[ids].to(self.device)
        active=target!=257;EOS=target==256
        self.exposure['active_symbols']+=int(active.sum());self.exposure['EOS']+=int(EOS.sum())
        self.exposure['target_bytes']+=int((target<256).sum())
        self.optimizer.zero_grad(set_to_none=True);self.head.train()
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            logits=self.head(local,global_views,terminal).float()
            ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),target.masked_fill(~active,-100),ignore_index=-100,reduction='none')
            byte_CE=ce.sum()/active.sum()
            EOS_CE=ce[EOS].mean() if EOS.any() else ce.sum()*0
            loss=byte_CE+.1*EOS_CE
        if not torch.isfinite(loss) or float(loss.detach())>1000:raise FloatingPointError('readout loss stop')
        self.budget.charge('readout');self.scaler.scale(loss).backward()
        if force_overflow:next(self.head.parameters()).grad.fill_(float('inf'))
        self.scaler.unscale_(self.optimizer)
        if any(parameter.grad is None for parameter in self.head.parameters()):raise ValueError('missing readout gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.head.parameters(),1.)
        before=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
        skip=self.scaler.get_scale()<before;self.skips+=int(skip);self.updates+=int(not skip)
        record=dict(attempt=self.attempts,updates=self.updates,rows=indices,loss=float(loss.detach()),byte_CE=float(byte_CE.detach()),
            EOS_CE=float(EOS_CE.detach()),AMP_skip=skip,scale=self.scaler.get_scale(),grad_norm=float(norm) if torch.isfinite(norm) else None)
        self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
        if sum(h['AMP_skip'] for h in self.history[-16:])>=3 or self.skips>=8:raise FloatingPointError('persistent probe AMP stop')
        if not skip and not torch.isfinite(norm):raise FloatingPointError('nonfinite applied readout gradient')
        return record

    def export(self):
        return dict(schema=SCHEMA,plan_sha256=self.plan_sha,parent_reference=self.budget.parent_reference,
            head={k:v.detach().cpu() for k,v in self.head.state_dict().items()},optimizer=self.optimizer.state_dict(),
            scaler=self.scaler.state_dict(),orders=self.orders,attempts=self.attempts,updates=self.updates,
            cursor=self.cursor,skips=self.skips,history=self.history,exposure=self.exposure,cache_refs=self.cache_refs,
            calls=self.budget.calls,gradients={},torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else self.inactive_cuda_rng,
            python_rng=random.getstate(),numpy_rng=numpy_state(),old_encoder_memory_optimizer_updates=0,
            phase=self.phase,extract_cursor=self.extract_cursor,extract_shards=self.extract_shards)

    def restore(self,state,verify_files=True):
        if (state['schema']!=SCHEMA or state['plan_sha256']!=self.plan_sha or state['orders']!=self.orders
            or state['parent_reference']!=self.budget.parent_reference or state['gradients']
            or state['cursor']!=sum(len(row) for row in self.orders[:state['attempts']])
            or [h['rows'] for h in state['history']]!=self.orders[:state['attempts']]):raise ValueError('probe recovery contract')
        if verify_files:
            for name,digest in {**state['cache_refs'],state['parent_reference']['path']:state['parent_reference']['sha256']}.items():
                if support.sha(name)!=digest:raise ValueError('parent/cache reference changed')
        self.head.load_state_dict(state['head'],strict=True);self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        for key in ('attempts','updates','cursor','skips','history','exposure','cache_refs','phase','extract_cursor','extract_shards'):setattr(self,key,copy.deepcopy(state[key]))
        self.budget.calls=copy.deepcopy(state['calls']);self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        torch.set_rng_state(state['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);n=state['numpy_rng']
        np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
