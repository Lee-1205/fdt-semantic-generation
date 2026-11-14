"""CPU research candidate: input meaning -> memory-conditioned VP -> final meaning.

All trainable tensors belong to this NEW thought module. Existing language,
anchors/prior, memory extraction/binding and language readout stay frozen.
The current Exact policy is NULL only; this is not a literal-copy implementation.
No actual learned-model/diffusion trial is launched by this module.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.frozen_language_boundary_contract_v1 import FrozenLanguageBoundary


@dataclass(frozen=True)
class ThoughtConfig:
    dimension: int = 512
    key_dimension: int = 128
    value_dimension: int = 256
    hidden: int = 256
    heads: int = 8
    layers: int = 2
    answer_local_capacity: int = 33  # first bounded SHORT-answer task, not an answer-length oracle
    source_slot_capacity: int = 16386  # input meaning boundary can retain 262144 source bytes
    max_steps: int = 16

    def __post_init__(self):
        if (min(self.dimension,self.key_dimension,self.value_dimension,self.hidden,self.heads,self.layers,self.answer_local_capacity,self.source_slot_capacity,self.max_steps)<1
                or self.hidden%self.heads or self.hidden%2 or self.answer_local_capacity<2):
            raise ValueError('explicit positive latent/memory/capacity contract')


@dataclass(frozen=True)
class InputMeaning:
    source: torch.Tensor
    anchors: torch.Tensor
    prior: torch.Tensor
    contextual_long: torch.Tensor
    contextual_recent: torch.Tensor
    exact_keys: torch.Tensor
    exact_values: torch.Tensor
    exact_valid: torch.Tensor
    model_epoch: str
    memory_revisions: tuple

    @classmethod
    def from_episode(cls,commit,episode):
        memories=episode.memories
        if memories.model_epoch!=commit.model_epoch or memories.persistent.certified_D.count_nonzero():
            raise ValueError('matching frozen epoch; certified writer remains unimplemented')
        if (memories.exact.valid.any() or memories.exact.keys.count_nonzero()
                or memories.exact.values.count_nonzero() or any(memories.exact.spans)):
            raise ValueError('Exact remains dormant; no source copy feature')
        long,recent=memories.contextual.means()
        tensors=(commit.reader_tensor(),memories.persistent.anchors,memories.persistent.prior,
                 long,recent,memories.exact.keys,memories.exact.values,memories.exact.valid)
        detached=tuple(x.detach().clone() for x in tensors)
        return cls(*detached,commit.model_epoch,tuple(commit.memory_revisions))


@dataclass(frozen=True)
class FinalMeaning:
    semantic: torch.Tensor
    model_epoch: str
    memory_revisions: tuple
    denoising_steps: int

    def __post_init__(self):
        if self.semantic.ndim!=3 or not torch.isfinite(self.semantic).all() or self.denoising_steps<1:
            raise ValueError('finite completed answer meaning')
        object.__setattr__(self,'semantic',self.semantic.detach().clone())

    def as_codec_commit(self):
        return DifferentiableCodecCommit.from_semantic(self.semantic,self.memory_revisions,self.model_epoch)


def vp(t):
    if t.ndim!=1 or not torch.isfinite(t).all() or ((t<0)|(t>1)).any():
        raise ValueError('one valid VP time per row')
    angle=t.float()*math.pi/2
    a=torch.where(t==1,torch.zeros_like(t),angle.cos()).reshape(-1,1,1)
    s=torch.where(t==0,torch.zeros_like(t),angle.sin()).reshape(-1,1,1)
    return a,s


def positions(length,width,device):
    pos=torch.arange(length,device=device,dtype=torch.float32)[:,None]
    freq=10000**(-torch.arange(0,width,2,device=device,dtype=torch.float32)/width)
    phase=pos*freq[None]
    return torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)


class MemoryStep(nn.Module):
    def __init__(self,c):
        super().__init__();h=c.hidden
        self.self_norm=nn.LayerNorm(h);self.self_attention=nn.MultiheadAttention(h,c.heads,dropout=0.,batch_first=True)
        self.cross_norm=nn.LayerNorm(h);self.source_attention=nn.MultiheadAttention(h,c.heads,dropout=0.,batch_first=True)
        self.query=nn.Linear(h,c.key_dimension,bias=False)
        self.recent_gate=nn.Linear(h,1)
        self.memory_projection=nn.Linear(3*c.value_dimension,h)
        self.ff_norm=nn.LayerNorm(h);self.ff=nn.Sequential(nn.Linear(h,2*h),nn.SiLU(),nn.Linear(2*h,h))

    def forward(self,h,source,memory):
        n=self.self_norm(h);h=h+self.self_attention(n,n,n,need_weights=False)[0]
        h=h+self.source_attention(self.cross_norm(h),source,source,need_weights=False)[0]
        query=F.normalize(self.query(h).float(),dim=-1)
        anchors=F.normalize(memory.anchors.float(),dim=-1)
        mu=(4.*(query@anchors.T)).sigmoid()
        rho=mu/mu.sum(-1,keepdim=True).clamp_min(1e-8);confidence=mu.amax(-1,keepdim=True)
        kp=confidence*(rho@memory.prior.float())
        long=confidence*torch.einsum('bsa,bav->bsv',rho,memory.contextual_long.float())
        recent=confidence*torch.einsum('bsa,bav->bsv',rho,memory.contextual_recent.float())
        gate=self.recent_gate(h).sigmoid();kc=gate*long+(1-gate)*recent
        # Always explicit NULL in the present policy; no hidden prior-filled
        # Exact value, source bytes, copy span or automatic whole-source recall.
        scores=(query@F.normalize(memory.exact_keys.float(),dim=-1).transpose(-1,-2)).masked_fill(~memory.exact_valid[:,None],-torch.inf)
        null=scores.new_zeros(*scores.shape[:-1],1)
        weights=torch.cat((scores,null),-1).softmax(-1)
        ke=weights[...,:-1]@memory.exact_values.float()
        h=h+self.memory_projection(torch.cat((kp,kc,ke),-1))
        return h+self.ff(self.ff_norm(h))


class MemoryThoughtDiffusion(nn.Module):
    def __init__(self,c=ThoughtConfig()):
        super().__init__();self.c=c
        self.source_projection=nn.Linear(c.dimension,c.hidden)
        self.prior_centre=nn.Linear(c.dimension,c.dimension)
        self.prior_scale=nn.Linear(c.dimension,c.dimension)
        self.state_projection=nn.Linear(c.dimension,c.hidden)
        self.centre_projection=nn.Linear(c.dimension,c.hidden)
        self.time=nn.Sequential(nn.Linear(c.hidden,c.hidden),nn.SiLU(),nn.Linear(c.hidden,c.hidden))
        self.steps=nn.ModuleList(MemoryStep(c) for _ in range(c.layers))
        self.final_norm=nn.LayerNorm(c.hidden);self.velocity=nn.Linear(c.hidden,c.dimension)

    def validate_condition(self,m):
        c=self.c;z=m.source;b,n,d=z.shape if z.ndim==3 else (0,0,0)
        if (b!=1 or not 2<n<=c.source_slot_capacity or d!=c.dimension or m.anchors.ndim!=2
                or m.anchors.shape[1]!=c.key_dimension or m.prior.shape!=(len(m.anchors),c.value_dimension)
                or m.contextual_long.shape!=(b,len(m.anchors),c.value_dimension)
                or m.contextual_recent.shape!=m.contextual_long.shape
                or m.exact_keys.ndim!=3 or m.exact_keys.shape[0]!=b or m.exact_keys.shape[-1]!=c.key_dimension
                or m.exact_values.shape!=(*m.exact_keys.shape[:2],c.value_dimension)
                or m.exact_valid.shape!=m.exact_keys.shape[:2] or m.exact_valid.dtype!=torch.bool):
            raise ValueError('input Commit and three-memory shape contract')
        arrays=(m.source,m.anchors,m.prior,m.contextual_long,m.contextual_recent,m.exact_keys,m.exact_values)
        if (any(x.requires_grad or not torch.isfinite(x).all() or x.device!=z.device for x in arrays)
                or m.exact_valid.device!=z.device or m.exact_valid.any()
                or m.exact_keys.count_nonzero() or m.exact_values.count_nonzero()):
            raise ValueError('detached readonly memories; dormant Exact only')
        if not m.model_epoch or not m.memory_revisions:raise ValueError('input semantic provenance')

    def initial_prior(self,m):
        self.validate_condition(m)
        summary=m.source[:,:2].float().mean(1)
        centre=self.prior_centre(summary)[:,None].expand(-1,self.c.answer_local_capacity+2,-1)
        scale=(.5+self.prior_scale(summary).sigmoid())[:,None]
        return centre,scale

    def initial_state(self,m,noise):
        centre,scale=self.initial_prior(m)
        if noise.shape!=centre.shape or not torch.isfinite(noise).all():raise ValueError('independent noise and fixed output capacity')
        return centre+scale*noise

    def forward(self,z,t,m):
        centre,scale=self.initial_prior(m)
        if z.shape!=centre.shape or t.shape!=(len(z),) or not torch.isfinite(z).all():raise ValueError('fixed-capacity noisy meaning/time shape')
        vp(t)
        state=(z-centre)/scale
        h=self.state_projection(state)+self.centre_projection(centre)
        h=h+positions(z.shape[1],self.c.hidden,z.device)[None]
        # Both sin and cos frequencies vary with t; avoid a constant t embedding.
        f=10000**(-torch.arange(0,self.c.hidden,2,device=z.device).float()/self.c.hidden)
        phase=t.float()[:,None]*f[None]*2*math.pi
        time_code=torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)
        h=h+self.time(time_code)[:,None]
        source=self.source_projection(m.source.float())+positions(m.source.shape[1],self.c.hidden,z.device)[None]
        for block in self.steps:h=block(h,source,m)
        return self.velocity(self.final_norm(h))

    def training_objective(self,m,target,mask,t,noise):
        if not self.training:raise RuntimeError('answer targets/noisy gold are training-only')
        centre,scale=self.initial_prior(m)
        if (target.shape!=centre.shape or mask.shape!=target.shape[:2] or mask.dtype!=torch.bool or not mask.any()
                or noise.shape!=target.shape or not torch.isfinite(noise).all() or not torch.isfinite(target).all()):
            raise ValueError('loss-only fixed-capacity answer meaning/mask')
        # Stop gradient through the corruption/label construction. Only the
        # prediction path learns. Input-conditioned centre/scale remain bounded
        # and learn through predicted-clean consistency; target is frozen E(Y).
        residual=(target.detach()-centre.detach())/scale.detach()
        a,s=vp(t);rt=a*residual+s*noise
        zt=centre.detach()+scale.detach()*rt
        predicted=self(zt,t,m);gold_v=a*noise-s*residual
        predicted_residual=a*((zt-centre)/scale)-s*predicted
        clean=centre+scale*predicted_residual
        mean=lambda values:(values.mean(-1)*mask).sum()/mask.sum()
        loss=mean((predicted-gold_v).square())+.25*mean((clean-target.detach()).square())
        return loss,clean

    @torch.no_grad()
    def sample(self,m,noise,steps=16,charge=lambda kind:None):
        if self.training:raise RuntimeError('inference needs thought.eval; no teacher target allowed')
        if type(steps)is not int or not 1<=steps<=self.c.max_steps:raise ValueError('bounded fixed denoising steps')
        centre,scale=self.initial_prior(m);z=self.initial_state(m,noise)
        for i in range(steps):
            t=z.new_full((len(z),),1-i/steps);next_t=z.new_full((len(z),),1-(i+1)/steps)
            a,s=vp(t);na,ns=vp(next_t);rt=(z-centre)/scale
            charge('denoise');v=self(z,t,m)
            clean_residual=a*rt-s*v;predicted_noise=s*rt+a*v
            z=centre+scale*(na*clean_residual+ns*predicted_noise)
        return FinalMeaning(z,m.model_epoch,m.memory_revisions,steps)


@torch.no_grad()
def target_meaning(answer_commit,terminal,capacity):
    """TRAIN-only E(answer) plus frozen D's learned terminal latent; no raw bytes.

    Capacity is predeclared, never chosen from gold answer length at inference.
    True target extent and the EOS placement affect this loss packet ONLY.
    """
    z=answer_commit.reader_tensor().detach();b,n,d=z.shape
    if capacity<2 or n-2>=capacity or terminal.shape!=(d,):raise ValueError('answer fits fixed capacity incl learned EOS slot')
    target=terminal.detach().to(z).reshape(1,1,d).expand(b,capacity+2,d).clone()
    target[:,:n]=z;mask=torch.zeros(b,capacity+2,dtype=torch.bool,device=z.device);mask[:,:n+1]=True
    return target,mask


def prepare_thought_optimizer(codec,thought,lr=1e-4):
    names=tuple(dict(thought.named_parameters()))
    boundary=FrozenLanguageBoundary(codec,thought,names)
    optimizer=torch.optim.AdamW(thought.parameters(),lr=lr,weight_decay=.01)
    boundary.assert_optimizer(optimizer);boundary.assert_intact()
    return boundary,optimizer


def verbalize_final(codec,meaning,boundary):
    if type(meaning)is not FinalMeaning:raise TypeError('language receives only FINAL meaning, never noisy state/source')
    boundary.assert_intact()
    return codec.verbalize(meaning.as_codec_commit())
