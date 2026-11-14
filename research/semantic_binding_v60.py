"""Facts are decoded ONLY after role-conditioned three-memory binding.

The fourth role is learned event/cue attention. Entity span predictions route
the first three roles. Neither pooled token context nor targets enter read_state.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.semantic_binding_v59 import SemanticBinder,MemoryTokens,MeaningCommit,fuzzy_role_bank,best_spans
from research.semantic_binding_v59 import training_loss as old_loss

@dataclass(frozen=True)
class BoundMeaningState:
    role_N:torch.Tensor
    role_D:torch.Tensor
    role_values:torch.Tensor
    predicted_spans:torch.Tensor
    source_revisions:tuple
    source_epoch:str
    binding_epoch:str
    donor_revisions:tuple=()

class SemanticBinderV60(SemanticBinder):
    def __init__(self,prior,source_epoch,hidden=128):
        if hidden!=128:raise ValueError('fixed128 role-memory readout width')
        super().__init__(prior,source_epoch,hidden)
        self.role_start_gain=nn.Parameter(torch.zeros(3));self.role_end_gain=nn.Parameter(torch.zeros(3))

    def extract_state(self,packet,binding_epoch,disabled_reads=()):
        if type(packet) is not MemoryTokens:raise TypeError('source-owned neural memory tokens required')
        packet.validate(self.source_epoch)
        if not binding_epoch or not set(disabled_reads)<={'persistent','contextual','exact'}:raise ValueError('explicit semantic scope')
        valid=packet.valid;n=valid.shape[1]
        x=self.project(torch.cat((packet.mu,packet.contextual_values,packet.exact_keys),-1))
        pos=torch.arange(n,device=x.device).float();freq=torch.exp(-math.log(10000.)*torch.arange(x.shape[-1],device=x.device)/x.shape[-1])
        x=(x+.02*torch.sin(pos[:,None]*freq)[None])*valid[...,None]
        for block in self.context:x=block(x,valid)
        attention=torch.einsum('qd,btd->bqt',self.pool_queries,x)/math.sqrt(x.shape[-1])
        attention=attention.masked_fill(~valid[:,None],-1e4).float().softmax(-1)
        pool=attention.to(x.dtype)@x;pooled=pool.flatten(1);null=self.null(pooled).reshape(len(x),2,3)
        contextual=(pool[:,:3]@x.transpose(1,2))/math.sqrt(x.shape[-1])
        s=self.start(x).transpose(1,2)+(128*self.role_start_gain).tanh()[None,:,None]*contextual
        e=self.end(x).transpose(1,2)+(128*self.role_end_gain).tanh()[None,:,None]*contextual
        start=torch.cat((s.masked_fill(~packet.starts[:,None],-1e4),null[:,0,:,None]),-1)
        end=torch.cat((e.masked_fill(~packet.ends[:,None],-1e4),null[:,1,:,None]),-1)
        ps,pe=start.float().softmax(-1),end.float().softmax(-1)
        span=ps[...,:n].cumsum(-1)*pe[...,:n].flip(-1).cumsum(-1).flip(-1)*valid[:,None]
        degree=torch.cat((span,attention[:,3:4]),1)
        presence=torch.cat(((1-ps[...,-1])*(1-pe[...,-1]),torch.ones(len(x),1,device=x.device)),1)
        values=self.value(x).float();numerator,denominator,means=fuzzy_role_bank(packet.mu,degree,values,valid)
        mass=degree.sum(-1,keepdim=True).clamp_min(1e-8);qmu=(degree@packet.mu.float())/mass
        rho=qmu/qmu.sum(-1,keepdim=True).clamp_min(1e-8);confidence=qmu.amax(-1,keepdim=True)
        kc=confidence*(rho[...,None,:]@means).squeeze(-2);kp=confidence*(rho@self.persistent_prior)
        q=F.normalize(self.exact_query((degree@x.float())/mass),dim=-1)
        scores=q@F.normalize(packet.exact_keys.float(),dim=-1).transpose(1,2)+degree.clamp_min(1e-8).log()
        scores=scores.masked_fill(~packet.starts[:,None],-1e4);ids=scores.argsort(dim=-1,descending=True,stable=True)[...,:min(16,n)]
        # The learned event query uses the mean of the existing NULL parameters;
        # existing three entity NULL parameters and all their weights are retained.
        null_exact=torch.cat((self.exact_null,self.exact_null.mean()[None]))
        weights=torch.cat((scores.gather(-1,ids),null_exact[None,:,None].expand(len(x),-1,-1)),-1).float().softmax(-1)
        picked=packet.exact_values[:,None].expand(-1,4,-1,-1).gather(2,ids[...,None].expand(-1,-1,-1,256))
        ke=(weights[...,:-1,None]*picked).sum(2)
        if 'persistent' in disabled_reads:kp=torch.zeros_like(kp)
        if 'contextual' in disabled_reads:kc=torch.zeros_like(kc)
        if 'exact' in disabled_reads:ke=torch.zeros_like(ke)
        fused=self.fuse(torch.cat((kc,kp,ke),-1))*presence[...,None]
        state=BoundMeaningState(numerator,denominator,fused,best_spans(start,end),packet.source_revisions,self.source_epoch,binding_epoch)
        raw=dict(start_logits=start,end_logits=end,role_degree=degree,projected_values=values,
            read_norms=torch.stack((kc.norm(dim=-1),kp.norm(dim=-1),ke.norm(dim=-1)),-1),exact_candidate_ids=ids,exact_weights=weights)
        return state,raw

    def read_state(self,state,diagnostic=False):
        if type(state) is not BoundMeaningState or state.source_epoch!=self.source_epoch or not state.binding_epoch:
            raise TypeError('factor readout accepts a bound meaning state ONLY')
        if state.role_values.shape[1:]!=(4,128) or state.predicted_spans.shape[1:]!=(3,2):raise ValueError('bound meaning shape')
        if state.donor_revisions and not diagnostic:raise ValueError('interchange states are diagnostic-only')
        # There is intentionally no packet, token pool, H or gold factor here.
        h=self.event(state.role_values.flatten(1));relation=self.relation(h).float();negative=self.negative(h).squeeze(-1).float();quantity=self.quantity(h).float()
        pr=relation.softmax(-1);neg=pr[:,1]*negative.sigmoid()
        polarity=torch.stack((neg,1-neg),-1);q=torch.cat((1-pr[:,2,None],pr[:,2,None]*quantity.softmax(-1)),-1)
        commit=MeaningCommit(state.role_N,state.role_D,state.role_values,state.predicted_spans,pr,polarity,q,
            state.source_revisions,state.source_epoch,state.binding_epoch)
        return commit,dict(relation_logits=relation,negative_logits=negative,quantity_logits=quantity)

    def forward(self,packet,binding_epoch,disabled_reads=()):
        state,raw=self.extract_state(packet,binding_epoch,disabled_reads);commit,factors=self.read_state(state)
        return commit,dict(raw,**factors)

def training_loss(commit,raw,targets,meaning_ids):
    loss,parts=old_loss(commit,raw,targets,meaning_ids)
    return loss+parts['quantity'],parts  # explicit quantity CE weight2
