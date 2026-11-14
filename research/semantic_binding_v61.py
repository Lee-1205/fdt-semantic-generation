"""Linear-space coupled entity spans before canonical three-memory binding.

This is extraction, not semantic denoising. Shared entity-interior supervision
is a LOSS only; every possible UTF8-aligned contiguous span remains available.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.semantic_binding_v59 import MemoryTokens, fuzzy_role_bank, best_spans
from research.semantic_binding_v60 import SemanticBinderV60, BoundMeaningState
from research.semantic_binding_v59 import training_loss as old_loss


def joint_intervals(start, end, inside):
    """Exact partition and occupancy of all a<=b spans + NULL in O(length).

    S(a,b)=start[a]+end[b]+sum(inside[a:b+1]). It factorizes into
    A[a]=start[a]-prefix[a-1], B[b]=end[b]+prefix[b]. Summing A<=t<=B
    gives occupancy; no independent-start/end product or gold candidates.
    """
    prefix = inside.float().cumsum(-1)
    before = F.pad(prefix[..., :-1], (1, 0))
    a = start[..., :-1].float() - before
    b = end[..., :-1].float() + prefix
    left = a.logcumsumexp(-1)
    right = b.flip(-1).logcumsumexp(-1).flip(-1)
    null = start[..., -1].float() + end[..., -1].float()
    logz_span = (left + b).logsumexp(-1)
    logz = torch.logaddexp(logz_span, null)
    degree = (left + right - logz[..., None]).exp().clamp(max=1)
    presence = -torch.expm1(null - logz)
    # Existing linear hard-span algorithm now operates on coupled energies.
    adjusted_start = torch.cat((a, start[..., -1:].float()), -1)
    adjusted_end = torch.cat((b, end[..., -1:].float()), -1)
    spans = best_spans(adjusted_start, adjusted_end)
    return spans, degree, presence, logz, adjusted_start, adjusted_end


class SemanticBinderV61(SemanticBinderV60):
    def __init__(self, prior, source_epoch, hidden=128):
        super().__init__(prior, source_epoch, hidden)
        self.sequence = nn.GRU(hidden, hidden//2, batch_first=True, bidirectional=True)
        self.sequence_out = nn.Linear(hidden, hidden)
        self.entity_inside = nn.Linear(hidden, 4)  # shared entity, then three roles
        nn.init.zeros_(self.sequence_out.weight); nn.init.zeros_(self.sequence_out.bias)
        nn.init.zeros_(self.entity_inside.weight); nn.init.zeros_(self.entity_inside.bias)

    def extract_state(self, packet, binding_epoch, disabled_reads=()):
        if type(packet) is not MemoryTokens: raise TypeError('source neural memories required')
        packet.validate(self.source_epoch)
        if not binding_epoch or not set(disabled_reads)<={'persistent','contextual','exact'}: raise ValueError('scope')
        valid=packet.valid; n=valid.shape[1]
        x=self.project(torch.cat((packet.mu,packet.contextual_values,packet.exact_keys),-1))
        pos=torch.arange(n,device=x.device).float(); freq=torch.exp(-math.log(10000.)*torch.arange(x.shape[-1],device=x.device)/x.shape[-1])
        x=(x+.02*torch.sin(pos[:,None]*freq)[None])*valid[...,None]
        for block in self.context: x=block(x,valid)
        # Packed bidirectionality prevents right-padding from changing valid tokens.
        with torch.autocast(x.device.type, enabled=False):
            packed=nn.utils.rnn.pack_padded_sequence(x.float(),valid.sum(-1).cpu(),batch_first=True,enforce_sorted=False)
            recurrent,_=self.sequence(packed)
            recurrent,_=nn.utils.rnn.pad_packed_sequence(recurrent,batch_first=True,total_length=n)
        x=(x+self.sequence_out(recurrent))*valid[...,None]
        attention=torch.einsum('qd,btd->bqt',self.pool_queries,x)/math.sqrt(x.shape[-1])
        attention=attention.masked_fill(~valid[:,None],-1e4).float().softmax(-1)
        pool=attention.to(x.dtype)@x; pooled=pool.flatten(1); null=self.null(pooled).reshape(len(x),2,3)
        contextual=(pool[:,:3]@x.transpose(1,2))/math.sqrt(x.shape[-1])
        s=self.start(x).transpose(1,2)+(128*self.role_start_gain).tanh()[None,:,None]*contextual
        e=self.end(x).transpose(1,2)+(128*self.role_end_gain).tanh()[None,:,None]*contextual
        start=torch.cat((s.masked_fill(~packet.starts[:,None],-1e4),null[:,0,:,None]),-1)
        end=torch.cat((e.masked_fill(~packet.ends[:,None],-1e4),null[:,1,:,None]),-1)
        interior=self.entity_inside(x).float().transpose(1,2)
        unary=2*torch.tanh((interior[:,:1]+interior[:,1:])/2)*valid[:,None]
        spans,span_degree,presence,logz,adjusted_start,adjusted_end=joint_intervals(start,end,unary)
        degree=torch.cat((span_degree*valid[:,None],attention[:,3:4]),1)
        presence=torch.cat((presence,torch.ones(len(x),1,device=x.device)),1)
        values=self.value(x).float(); numerator,denominator,means=fuzzy_role_bank(packet.mu,degree,values,valid)
        mass=degree.sum(-1,keepdim=True).clamp_min(1e-8); qmu=(degree@packet.mu.float())/mass
        rho=qmu/qmu.sum(-1,keepdim=True).clamp_min(1e-8); confidence=qmu.amax(-1,keepdim=True)
        kc=confidence*(rho[...,None,:]@means).squeeze(-2); kp=confidence*(rho@self.persistent_prior)
        q=F.normalize(self.exact_query((degree@x.float())/mass),dim=-1)
        scores=q@F.normalize(packet.exact_keys.float(),dim=-1).transpose(1,2)+degree.clamp_min(1e-8).log()
        scores=scores.masked_fill(~packet.starts[:,None],-1e4)
        ids=scores.argsort(dim=-1,descending=True,stable=True)[...,:min(16,n)]
        null_exact=torch.cat((self.exact_null,self.exact_null.mean()[None]))
        weights=torch.cat((scores.gather(-1,ids),null_exact[None,:,None].expand(len(x),-1,-1)),-1).float().softmax(-1)
        picked=packet.exact_values[:,None].expand(-1,4,-1,-1).gather(2,ids[...,None].expand(-1,-1,-1,256))
        ke=(weights[...,:-1,None]*picked).sum(2)
        if 'persistent' in disabled_reads: kp=torch.zeros_like(kp)
        if 'contextual' in disabled_reads: kc=torch.zeros_like(kc)
        if 'exact' in disabled_reads: ke=torch.zeros_like(ke)
        fused=self.fuse(torch.cat((kc,kp,ke),-1))*presence[...,None]
        state=BoundMeaningState(numerator,denominator,fused,spans,packet.source_revisions,self.source_epoch,binding_epoch)
        raw=dict(start_logits=adjusted_start,end_logits=adjusted_end,joint_logZ=logz,entity_inside_logits=interior,
            valid=valid,role_degree=degree,projected_values=values,read_norms=torch.stack((kc.norm(dim=-1),kp.norm(dim=-1),ke.norm(dim=-1)),-1),
            exact_candidate_ids=ids,exact_weights=weights)
        return state,raw


def training_loss(commit, raw, targets, meaning_ids):
    _,parts=old_loss(commit,raw,targets,meaning_ids)
    span=targets['spans']; n=raw['valid'].shape[-1]
    a=span[...,0].masked_fill(span[...,0]<0,n); b=(span[...,1]-1).masked_fill(span[...,1]<0,n)
    gold_score=raw['start_logits'].gather(-1,a[...,None]).squeeze(-1)+raw['end_logits'].gather(-1,b[...,None]).squeeze(-1)
    joint=(raw['joint_logZ']-gold_score).mean()
    positions=torch.arange(n,device=span.device)[None,None]
    role_target=((positions>=span[...,0,None])&(positions<span[...,1,None])&(span[...,0,None]>=0)).float()
    entity_target=role_target.amax(1,keepdim=True)
    y=torch.cat((entity_target,role_target),1); mask=raw['valid'][:,None].expand_as(y)
    # Positive reweighting is loss-only and uses no gold runtime candidate mask.
    positives=(y*mask).sum((0,2)); negatives=((1-y)*mask).sum((0,2))
    ratio=(negatives/positives.clamp_min(1)).clamp(1,12)[None,:,None]
    interior=F.binary_cross_entropy_with_logits(raw['entity_inside_logits'],y,reduction='none',pos_weight=ratio)
    interior=(interior*mask).sum()/mask.sum()
    parts=dict(parts,pointer=joint,interior=interior)
    loss=joint+parts['relation']+parts['polarity']+2*parts['quantity']+.1*parts['contrast']+.25*interior
    return loss,parts
