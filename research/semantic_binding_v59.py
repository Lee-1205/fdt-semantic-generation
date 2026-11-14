"""Learned one-event semantic binding over the retained three memory banks.

The encoder predicts role spans from neural memory features. Source bytes,
gold spans, gold factors and templates are not forward inputs. A new typed
MeaningCommit deliberately cannot masquerade as the old surface CodecCommit.
No semantic denoising and no language decoder are implemented in this stage.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class MemoryTokens:
    mu:torch.Tensor
    contextual_values:torch.Tensor
    exact_keys:torch.Tensor
    exact_values:torch.Tensor
    valid:torch.Tensor
    starts:torch.Tensor
    ends:torch.Tensor
    source_revisions:tuple
    source_epoch:str

    def validate(self,epoch):
        b,n,a=self.mu.shape
        if self.source_epoch!=epoch:raise ValueError('stale source-memory embedding epoch')
        if self.contextual_values.shape!=(b,n,256) or self.exact_values.shape!=(b,n,256) or self.exact_keys.shape!=(b,n,128) or a!=64:
            raise ValueError('canonical source memory dimensions')
        for mask in (self.valid,self.starts,self.ends):
            if mask.shape!=(b,n) or mask.dtype!=torch.bool:raise ValueError('input-only source boundary masks')
        if not self.valid.any(1).all() or ((~self.valid).long().cumsum(1).bool()&self.valid).any():raise ValueError('nonempty contiguous source')
        if (self.starts&~self.valid).any() or (self.ends&~self.valid).any() or len(self.source_revisions)!=b:raise ValueError('source binding')
        if not torch.isfinite(self.mu).all() or ((self.mu<0)|(self.mu>1)).any():raise ValueError('fuzzy degree bounds')
        for values in (self.contextual_values,self.exact_keys,self.exact_values):
            if not torch.isfinite(values).all():raise ValueError('nonfinite source memory')

@dataclass(frozen=True)
class MeaningCommit:
    role_N:torch.Tensor
    role_D:torch.Tensor
    role_values:torch.Tensor
    predicted_spans:torch.Tensor
    relation_probabilities:torch.Tensor
    polarity_probabilities:torch.Tensor
    quantity_probabilities:torch.Tensor
    source_revisions:tuple
    source_epoch:str
    binding_epoch:str
    phase:str='language_meaning_extraction_only'

    def semantic_vector(self):
        roles=F.normalize(self.role_values.float(),dim=-1,eps=1e-8).flatten(1)
        return torch.cat((roles,self.relation_probabilities*2,self.polarity_probabilities*2,self.quantity_probabilities*2),-1).clone()

    def predicted_references(self):
        # References only. No source bytes, issued CopyPermit, or automatic copy.
        spans=self.predicted_spans.detach().cpu().tolist()
        return tuple(tuple(None if a<0 else (*revision,a,b) for a,b in row) for revision,row in zip(self.source_revisions,spans))

def fuzzy_role_bank(mu,role_degree,values,valid):
    with torch.autocast(mu.device.type,enabled=False):
        weights=mu.double()[:,None]*role_degree.double()[...,None]*valid[:,None,:,None]
        numerator=torch.einsum('brta,btv->brav',weights,values.double())
        denominator=weights.sum(2)[...,None]
        means=torch.where(denominator>1e-8,numerator/denominator.clamp_min(1e-8),torch.zeros_like(numerator)).float()
    return numerator,denominator,means

def best_spans(start,end):
    # Linear in input length. UTF8 start/end masks were applied before this.
    values,indices=start[...,:-1].cummax(-1)
    scores=values+end[...,:-1];stop=scores.argmax(-1)
    begin=indices.gather(-1,stop[...,None]).squeeze(-1)
    span_score=scores.gather(-1,stop[...,None]).squeeze(-1)
    absent=start[...,-1]+end[...,-1]>=span_score
    return torch.stack((begin.masked_fill(absent,-1),(stop+1).masked_fill(absent,-1)),-1)

class LocalContext(nn.Module):
    def __init__(self,d):
        super().__init__();self.norm=nn.LayerNorm(d);self.conv=nn.Conv1d(d,d,7,padding=3,groups=d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.SiLU(),nn.Linear(2*d,d))
    def forward(self,x,valid):
        y=self.norm(x)*valid[...,None];x=(x+self.conv(y.transpose(1,2)).transpose(1,2))*valid[...,None]
        return (x+self.ff(self.norm(x)))*valid[...,None]

class SemanticBinder(nn.Module):
    def __init__(self,prior,source_epoch,hidden=128):
        super().__init__()
        if prior.shape!=(64,256) or not source_epoch:raise ValueError('frozen original K_P prior required')
        self.register_buffer('persistent_prior',prior.detach().float().clone());self.source_epoch=source_epoch
        self.project=nn.Linear(448,hidden);self.context=nn.ModuleList([LocalContext(hidden),LocalContext(hidden)])
        self.start=nn.Linear(hidden,3);self.end=nn.Linear(hidden,3)
        self.pool_queries=nn.Parameter(torch.randn(4,hidden)/math.sqrt(hidden))
        self.null=nn.Linear(4*hidden,6)
        self.value=nn.Linear(hidden,256);self.exact_query=nn.Linear(hidden,128,bias=False)
        self.exact_null=nn.Parameter(torch.zeros(3));self.fuse=nn.Linear(768,128)
        self.event=nn.Sequential(nn.Linear(4*hidden,hidden),nn.SiLU())
        self.relation=nn.Linear(hidden,3);self.negative=nn.Linear(hidden,1);self.quantity=nn.Linear(hidden,24)

    def forward(self,packet,binding_epoch,disabled_reads=()):
        if type(packet) is not MemoryTokens:raise TypeError('semantic encoder accepts source-owned neural memories only')
        packet.validate(self.source_epoch)
        if not binding_epoch or not set(disabled_reads)<={'persistent','contextual','exact'}:raise ValueError('explicit binding epoch/read controls')
        valid=packet.valid;n=valid.shape[1]
        x=self.project(torch.cat((packet.mu,packet.contextual_values,packet.exact_keys),-1))
        # Relative structural positions have no learned length table/cap.
        position=torch.arange(n,device=x.device).float()
        frequency=torch.exp(-math.log(10000.)*torch.arange(x.shape[-1],device=x.device)/x.shape[-1])
        x=(x+.02*torch.sin(position[:,None]*frequency)[None])*valid[...,None]
        for block in self.context:x=block(x,valid)
        attention=torch.einsum('qd,btd->bqt',self.pool_queries,x)/math.sqrt(x.shape[-1])
        pool=attention.masked_fill(~valid[:,None],-1e4).softmax(-1)@x
        pooled=pool.flatten(1);null=self.null(pooled).reshape(len(x),2,3)
        start=torch.cat((self.start(x).transpose(1,2).masked_fill(~packet.starts[:,None],-1e4),null[:,0,:,None]),-1)
        end=torch.cat((self.end(x).transpose(1,2).masked_fill(~packet.ends[:,None],-1e4),null[:,1,:,None]),-1)
        ps,pe=start.float().softmax(-1),end.float().softmax(-1)
        degree=ps[...,:n].cumsum(-1)*pe[...,:n].flip(-1).cumsum(-1).flip(-1)*valid[:,None]
        presence=(1-ps[...,-1])*(1-pe[...,-1])
        values=self.value(x).float();numerator,denominator,means=fuzzy_role_bank(packet.mu,degree,values,valid)
        mass=degree.sum(-1,keepdim=True).clamp_min(1e-8)
        query_mu=(degree@packet.mu.float())/mass;rho=query_mu/query_mu.sum(-1,keepdim=True).clamp_min(1e-8)
        confidence=query_mu.amax(-1,keepdim=True)
        kc=confidence*(rho[...,None,:]@means).squeeze(-2)
        kp=confidence*(rho@self.persistent_prior)
        role_context=(degree@x.float())/mass
        q=F.normalize(self.exact_query(role_context),dim=-1)
        keys=F.normalize(packet.exact_keys.float(),dim=-1)
        scores=q@keys.transpose(1,2)+degree.clamp_min(1e-8).log()
        scores=scores.masked_fill(~packet.starts[:,None],-1e4)
        ids=scores.argsort(dim=-1,descending=True,stable=True)[...,:min(16,n)]
        selected=scores.gather(-1,ids)
        ew=torch.cat((selected,self.exact_null[None,:,None].expand(len(x),-1,-1)),-1).float().softmax(-1)
        gathered=packet.exact_values[:,None].expand(-1,3,-1,-1).gather(2,ids[...,None].expand(-1,-1,-1,256))
        ke=(ew[...,:-1,None]*gathered).sum(2)
        if 'persistent' in disabled_reads:kp=torch.zeros_like(kp)
        if 'contextual' in disabled_reads:kc=torch.zeros_like(kc)
        if 'exact' in disabled_reads:ke=torch.zeros_like(ke)
        fused=self.fuse(torch.cat((kc,kp,ke),-1))*presence[...,None]
        h=self.event(pooled);relation=self.relation(h).float();neg=self.negative(h).squeeze(-1).float();qty=self.quantity(h).float()
        pr=relation.softmax(-1);negative=pr[:,1]*neg.sigmoid()
        polarity=torch.stack((negative,1-negative),-1)
        quantities=torch.cat(((1-pr[:,2,None]),pr[:,2,None]*qty.softmax(-1)),-1)
        commit=MeaningCommit(numerator,denominator,fused,best_spans(start,end),pr,polarity,quantities,
            packet.source_revisions,self.source_epoch,binding_epoch)
        return commit,dict(start_logits=start,end_logits=end,relation_logits=relation,negative_logits=neg,quantity_logits=qty,
            role_degree=degree,projected_values=values,read_norms=torch.stack((kc.norm(dim=-1),kp.norm(dim=-1),ke.norm(dim=-1)),-1),
            exact_candidate_ids=ids,exact_weights=ew)

def training_loss(commit,raw,targets,meaning_ids):
    # Targets are explicit arguments to LOSS only, never SemanticBinder.forward.
    spans=targets['spans'];n=raw['start_logits'].shape[-1]-1
    start=spans[...,0].masked_fill(spans[...,0]<0,n);end=(spans[...,1]-1).masked_fill(spans[...,1]<0,n)
    pointer=(F.cross_entropy(raw['start_logits'].flatten(0,1),start.flatten())+F.cross_entropy(raw['end_logits'].flatten(0,1),end.flatten()))/2
    relation=F.cross_entropy(raw['relation_logits'],targets['relation'])
    opens=targets['relation']==1;contains=targets['relation']==2
    polarity=F.binary_cross_entropy_with_logits(raw['negative_logits'][opens],1-targets['polarity'][opens].float()) if opens.any() else raw['negative_logits'].sum()*0
    quantity=F.cross_entropy(raw['quantity_logits'][contains],targets['quantity'][contains]-1) if contains.any() else raw['quantity_logits'].sum()*0
    z=F.normalize(commit.semantic_vector(),dim=-1);similarity=z@z.T/.2
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positives=(meaning_ids[:,None]==meaning_ids[None])&~eye
    logp=similarity-similarity.masked_fill(eye,-torch.inf).logsumexp(-1,keepdim=True)
    eligible=positives.any(-1)
    contrast=(-(logp.masked_fill(~positives,0).sum(-1)/positives.sum(-1).clamp_min(1))[eligible].mean()
        if eligible.any() else z.sum()*0)
    loss=pointer+relation+polarity+quantity+.1*contrast
    return loss,dict(pointer=pointer,relation=relation,polarity=polarity,quantity=quantity,contrast=contrast)
