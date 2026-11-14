"""One finite document epoch, frozen old encoder, live ordered-Commit training.

No growing cache: source roles are recomputed per input under frozen weights.
Eight active Adam states continue; consumed rows, RNG, LR and loss are durable.
"""
import math
import torch
from balanced_fuzzy_engine_v56 import ReadoutEngine
from research.ordered_fuzzy_readout_v55 import fuzzy_role_reads
from research.text_diffusion_v1 import pack_prompts

class BroadOrderedEngine(ReadoutEngine):
    def step_document(self,model,row,force_overflow=False):
        self.budget.check()
        if self.attempts>=len(self.orders):raise StopIteration('one finite document epoch ended')
        indices=self.orders[self.attempts]
        for group,base in zip(self.optimizer.param_groups,(1e-4,3e-4)):
            group['lr']=base*(.1+.9*(1+math.cos(math.pi*self.attempts/len(self.orders)))/2)
        self.attempts+=1;self.cursor+=1
        raw=row['prompt'].encode();x=pack_prompts([row['prompt']],len(raw),self.device)
        self.budget.charge('encode')
        with torch.no_grad(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            ep=model.encode_input(x,source_ids=(row['id'],),source_versions=(self.budget.p['source_sha'],),model_epoch=self.budget.p['source_sha'])
            z=ep.commit.reader_tensor().float()
            roles=fuzzy_role_reads(ep.diagnostics['token_memberships'],ep.diagnostics['token_values'],x!=256)[0]
            roles=torch.cat((roles,roles.new_zeros(1,16,256)))
            local=torch.cat((z[0,2:],z.new_zeros(1,512)))
            global_views=z[0,:2].expand(len(local),-1,-1)
            terminal=torch.zeros(len(local),device=self.device,dtype=torch.bool);terminal[-1]=True
        y=torch.full((len(local)*16,),257,device=self.device,dtype=torch.long)
        y[:len(raw)]=torch.tensor(list(raw),device=self.device);y[len(raw)]=256;y=y.reshape(-1,16)
        active=y!=257;EOS=y==256;early=torch.arange(len(local),device=self.device)<2
        self.exposure['selected_patches']+=int(active.any(1).sum());self.exposure['active_symbols']+=len(raw)+1
        self.exposure['target_bytes']+=len(raw);self.exposure['EOS']+=1
        self.optimizer.zero_grad(set_to_none=True);self.head.train();self.budget.charge('readout')
        with torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            logits=self.head(local,global_views,terminal,roles).float()
            ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),y.masked_fill(~active,-100),ignore_index=-100,reduction='none')
            weights=self.loss_weights[y]*torch.where(early[:,None],4.,1.)
            loss=(ce*weights).sum()/weights.sum();uniform=ce.sum()/active.sum()
        if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('document loss stop')
        self.scaler.scale(loss).backward()
        if force_overflow:next(self.head.parameters()).grad.fill_(float('inf'))
        self.scaler.unscale_(self.optimizer)
        if any(v.grad is None for v in self.head.parameters()):raise ValueError('active gradient ownership')
        if any(v.grad is not None for v in model.parameters()):raise ValueError('frozen encoder gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.head.parameters(),1.)
        scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
        skip=self.scaler.get_scale()<scale;self.skips+=int(skip);self.updates+=int(not skip)
        record=dict(attempt=self.attempts,updates=self.updates,rows=indices,source_id=row['id'],bytes=len(raw),
            loss=float(loss.detach()),uniform_CE=float(uniform.detach()),AMP_skip=skip,scale=self.scaler.get_scale(),
            grad_norm=float(norm) if torch.isfinite(norm) else None)
        self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
        if sum(h['AMP_skip'] for h in self.history[-16:])>=3 or self.skips>=8:raise FloatingPointError('persistent AMP stop')
        if not skip and not torch.isfinite(norm):raise FloatingPointError('applied gradient nonfinite')
        return record
