"""FDT variable-order numerical semantics adapter. Not the canonical whole FDT."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Config:
    name: str = 'fdt-variable-order-100m-v0.3'
    lang_dim: int = 768
    lang_heads: int = 12
    lang_ff: int = 2048
    lang_layers: int = 9
    meaning_views: int = 2
    diffusion_role_condition: bool = True
    z_dim: int = 512
    core_heads: int = 8
    core_ff: int = 1536
    init_layers: int = 2
    diffusion_layers: int = 6
    anchors: int = 64
    key_dim: int = 128
    value_dim: int = 256
    top_k: int = 8
    relations: int = 8
    rank: int = 8
    entities: int = 8
    facts: int = 12
    slots: int = 16
    phrase_bytes: int = 48
    window: int = 48
    gradient_checkpointing: bool = False

    @classmethod
    def smoke(cls):
        return cls(name='fdt100-smoke',lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=2,
                   z_dim=32,core_heads=4,core_ff=64,init_layers=1,diffusion_layers=2,
                   anchors=16,key_dim=16,value_dim=16,top_k=4,rank=4)

    def __post_init__(self):
        if self.lang_dim % self.lang_heads or self.z_dim % self.core_heads:
            raise ValueError('head dimension mismatch')
        if self.lang_dim//self.lang_heads % 2 or self.z_dim < 2*self.entities+5:
            raise ValueError('RoPE/semantic basis dimension mismatch')
        if not 0 < self.top_k <= self.anchors or self.entities != 8:
            raise ValueError('unsupported anchor/entity schema')


def linear(a,b):
    return nn.Linear(a,b,bias=False)


class RMS(nn.Module):
    def __init__(self,d):
        super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):
        dtype=x.dtype
        return (x.float()*(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt()).to(dtype)*self.weight


class FF(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.up=linear(d,h);self.gate=linear(d,h);self.down=linear(h,d)
    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))


class Attn(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.q=linear(d,d);self.k=linear(d,d);self.v=linear(d,d);self.o=linear(d,d)
        self.h=h;self.w=d//h
    def forward(self,x,c,mask=None,rotary=False):
        b,n,d=x.shape;m=c.shape[1]
        q=self.q(x).view(b,n,self.h,self.w).transpose(1,2)
        k=self.k(c).view(b,m,self.h,self.w).transpose(1,2)
        v=self.v(c).view(b,m,self.h,self.w).transpose(1,2)
        if rotary:
            pos=torch.arange(n,device=x.device,dtype=torch.float32)
            freq=10000**(-torch.arange(0,self.w,2,device=x.device).float()/self.w)
            phase=pos[:,None]*freq
            def rotate(t):
                a,b=t.float()[...,0::2],t.float()[...,1::2]
                return torch.stack((a*phase.cos()-b*phase.sin(),a*phase.sin()+b*phase.cos()),-1).flatten(-2).to(t.dtype)
            q,k=rotate(q),rotate(k)
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        return self.o(out.transpose(1,2).reshape(b,n,d))


class LanguageBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.n1=RMS(c.lang_dim);self.n2=RMS(c.lang_dim)
        self.attn=Attn(c.lang_dim,c.lang_heads);self.ff=FF(c.lang_dim,c.lang_ff)
    def forward(self,x,mask,valid):
        x=x+self.attn(self.n1(x),self.n1(x),mask,True)
        return (x+self.ff(self.n2(x)))*valid[...,None]


class Language(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.embedding=nn.Embedding(257,c.lang_dim,padding_idx=256)
        self.blocks=nn.ModuleList([LanguageBlock(c) for _ in range(c.lang_layers)])
        self.pool=nn.Parameter(torch.randn(c.meaning_views,c.lang_dim)*.02)
    def forward(self,tokens):
        b,f,n=tokens.shape;tokens=tokens.reshape(b*f,n);valid=tokens!=256
        x=self.embedding(tokens);pos=torch.arange(n,device=tokens.device)
        band=(pos[:,None]>=pos[None]) & (pos[:,None]-pos[None]<self.c.window)
        mask=band[None,None] & (valid[:,None,None,:] | torch.eye(n,device=tokens.device,dtype=torch.bool)[None,None])
        for block in self.blocks:
            if self.c.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x=checkpoint(block,x,mask,valid,use_reentrant=False)
            else:x=block(x,mask,valid)
        weights=(x.float()@self.pool.float().T).masked_fill(~valid[...,None],-1e4).softmax(1)
        return torch.einsum('blv,bld->bvd',weights,x.float()).reshape(b,f,self.c.meaning_views,-1)


class Core(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.z_dim;self.norms=nn.ModuleList([RMS(d) for _ in range(3)])
        self.self_attn=Attn(d,c.core_heads);self.cross=Attn(d,c.core_heads);self.ff=FF(d,c.core_ff)
    def forward(self,x,c):
        y=self.norms[0](x);x=x+self.self_attn(y,y)
        x=x+self.cross(self.norms[1](x),c)
        return x+self.ff(self.norms[2](x))


class Router(nn.Module):
    def __init__(self,d,c):
        super().__init__();self.query=linear(d,c.key_dim);self.view=linear(c.key_dim,c.key_dim)
        self.sigma=nn.Parameter(torch.zeros(c.anchors));self.beta=nn.Parameter(torch.zeros(c.anchors));self.k=c.top_k
    def forward(self,x,anchors):
        # Routing probabilities and N/D intentionally stay FP32 under AMP.
        with torch.autocast(device_type=x.device.type,enabled=False):
            q=self.query(x.float());a=self.view(anchors.float())
            valid=(q.norm(dim=-1)>1e-8)[...,None] & (a.norm(dim=-1)>1e-8)
            q=F.normalize(q,dim=-1);a=F.normalize(a,dim=-1)
            distance=(2-2*q@a.T).clamp(0,4)
            logits=self.beta-distance/(2*(F.softplus(self.sigma)+1e-3).square())
            logits=logits.masked_fill(~valid,-torch.inf)
            ids=logits.argsort(dim=-1,descending=True,stable=True)[...,:self.k]
            chosen=torch.zeros_like(logits,dtype=torch.bool).scatter(-1,ids,True)
            mu=torch.where(chosen&valid,logits.sigmoid(),torch.zeros_like(logits))
            mass=mu.sum(-1,keepdim=True)
            rho=torch.where(mass>1e-8,mu/mass.clamp_min(1e-8),torch.zeros_like(mu))
            return mu,rho,mu.amax(-1)


def fuzzy_closure(grades,roles,mask,n=8):
    """Learned phrase grades -> exact entity-role max-product circuit; no gold closure."""
    s,o=roles[...,:n],roles[...,n:]
    positive=grades[...,:2]*mask[...,None]
    edges=(torch.einsum('bfi,bfj,bf->bfij',s,o,positive[...,0])+
           torch.einsum('bfi,bfj,bf->bfij',o,s,positive[...,1])).amax(1)
    reach=edges
    # n-1 hops suffice for bounded simple paths. Repeated edge extension, not squaring.
    for _ in range(n-2):reach=torch.maximum(reach,(reach[:,:,:,None]*edges[:,None,:,:]).amax(-2))
    not_g=torch.einsum('bfi,bfj,bf->bfij',s,o,grades[...,2]*mask).amax(1)
    not_l=torch.einsum('bfi,bfj,bf->bfij',s,o,grades[...,3]*mask).amax(1)
    reverse=reach.transpose(-1,-2)
    contradiction=torch.maximum(reach*reverse,torch.maximum(reach*not_g,reverse*not_l))
    # A cycle is a global inconsistency in the supplied strict-order premise set.
    cycle=reach.diagonal(dim1=-2,dim2=-1).amax(-1)[:,None,None]
    conflict=torch.maximum(contradiction.amax((-2,-1))[:,None,None],cycle).expand_as(reach)
    unknown=(1-reach)*(1-reverse)*(1-conflict)
    scores=torch.stack((unknown,reach*(1-conflict),reverse*(1-conflict),conflict),-1)
    return scores/scores.sum(-1,keepdim=True).clamp_min(1e-8)


class FDT100(nn.Module):
    def __init__(self,c=Config()):
        super().__init__();self.c=c;d=c.z_dim
        self.language=Language(c)
        self.anchors=nn.Parameter(torch.randn(c.anchors,c.key_dim)*.02)
        self.prior=nn.Parameter(torch.randn(c.anchors,c.value_dim)*.1)
        self.phrase_router=Router(c.lang_dim,c);self.slot_router=Router(d,c)
        self.ground_head=nn.Linear(c.meaning_views*c.value_dim,4)
        self.value=linear(c.lang_dim+16,c.value_dim)
        self.condition=nn.Sequential(nn.Linear(c.value_dim+16,d),nn.SiLU(),linear(d,d))
        self.memory_project=linear(c.value_dim,d)
        self.role_project=linear(16,d)
        self.initializer=nn.ModuleList([Core(c) for _ in range(c.init_layers)])
        self.initial_out=linear(d,d)
        self.rel_left=nn.Parameter(torch.randn(c.relations,c.anchors,c.rank)*.02)
        self.rel_right=nn.Parameter(torch.randn(c.relations,c.anchors,c.rank)*.02)
        self.rel_score=nn.Sequential(nn.Linear(2*d,d),nn.SiLU(),nn.Linear(d,c.relations+1))
        self.rel_message=nn.Sequential(nn.Linear(2*c.anchors+16,d),nn.SiLU(),linear(d,d))
        self.time=nn.Sequential(nn.Linear(64,d),nn.SiLU(),linear(d,d))
        self.refiner=nn.ModuleList([Core(c) for _ in range(c.diffusion_layers)])
        self.refine_norm=RMS(d);self.velocity=linear(d,d)
        # Fixed orthonormal target basis: no trainable teacher, no labels at runtime.
        g=torch.Generator().manual_seed(3901)
        basis=torch.linalg.qr(torch.randn(d,21,generator=g),mode='reduced').Q.T
        self.register_buffer('basis',basis)
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),32)))

    def semantic(self,query_roles,predicate):
        fields=torch.cat((query_roles.float(),predicate.float(),torch.ones_like(predicate[...,:1])), -1)
        return fields@self.basis.float()*math.sqrt(self.c.z_dim/4)

    def target(self,features,labels):
        return self.semantic(features['queries'],F.one_hot(labels,4))*features['query_mask'][...,None]

    def readout(self,z):
        with torch.autocast(device_type=z.device.type,enabled=False):
            return 4*(z.float()@self.basis[16:20].float().T)/math.sqrt(self.c.z_dim/4)

    def encode(self,features,ablate=False):
        h=self.language(features['tokens']);roles=features['roles'];mask=features['fact_mask']
        mu,rho,gamma=self.phrase_router(h,self.anchors)
        mu=mu*mask[...,None,None];rho=rho*mask[...,None,None];gamma=gamma*mask[...,None]
        if ablate:mu,rho,gamma=mu*0,rho*0,gamma*0
        with torch.autocast(device_type=h.device.type,enabled=False):
            prior=gamma[...,None]*(rho@self.prior.float())
            ground=self.ground_head(prior.flatten(-2))
            # Independent views preserve operator qualifications; their nonexclusive
            # memberships combine by a declared probabilistic-sum t-conorm.
            mu=1-(1-mu).prod(-2)
            rho=mu/mu.sum(-1,keepdim=True).clamp_min(1e-8);gamma=mu.amax(-1)
            prior=prior.mean(-2)
        grade=ground.float().softmax(-1)
        possibility=fuzzy_closure(grade,roles.float(),mask.float())
        qroles=features['queries'];s=qroles[...,:8];o=qroles[...,8:]
        pred=torch.einsum('bqi,bqj,bijc->bqc',s,o,possibility)
        with torch.autocast(device_type=h.device.type,enabled=False):
            v=self.value(torch.cat((h.float().mean(-2),roles.float()),-1))
            num=torch.einsum('bfk,bfv->bkv',mu,v);den=mu.sum(1)[...,None]
            memory=torch.where(den>1e-8,num/den.clamp_min(1e-8),torch.zeros_like(num))
            contextual=gamma[...,None]*(rho@memory)
        facts=self.condition(torch.cat((contextual+prior,roles),-1))*mask[...,None]
        null=facts.new_zeros(len(h),1,self.c.z_dim)
        context=torch.cat((facts,null),1)
        z=self.role_project(qroles)
        for block in self.initializer:z=block(z,context)
        soft=self.semantic(qroles,pred)
        z_init=(soft+.2*torch.tanh(self.initial_out(z)))*features['query_mask'][...,None]
        return {'init':z_init,'facts':facts,'memory':memory,'ground':ground,'grades':grade,
                'query_roles':qroles,'query_mask':features['query_mask']}

    def condition_at(self,z,enc,ablate=False):
        mu,rho,gamma=self.slot_router(z,self.anchors)
        if ablate:mu,rho,gamma=mu*0,rho*0,gamma*0
        with torch.autocast(device_type=z.device.type,enabled=False):
            mem=gamma[...,None]*(rho@enc['memory'].float())
            left,right=F.softplus(self.rel_left)+1e-6,F.softplus(self.rel_right)+1e-6
            g=left@right.transpose(-1,-2)/math.sqrt(self.c.rank)
            transport=g/g.sum(-1,keepdim=True);fuzzy=g/(1+g)
        b,q,d=z.shape
        pair=torch.cat((z[:,:,None].expand(b,q,q,d),z[:,None].expand(b,q,q,d)),-1)
        pi=self.rel_score(pair).float().softmax(-1)
        with torch.autocast(device_type=z.device.type,enabled=False):
            # Fuzzy max-product and row-normalized transport keep distinct meanings.
            mixture_t=torch.einsum('bqpr,rij->bqpij',pi[...,1:],transport)
            mixture_f=torch.einsum('bqpr,rij->bqpij',pi[...,1:],fuzzy)
            p=torch.einsum('bqi,bqpij->bqpj',rho,mixture_t)
            nu=(mu[:,:,None,:,None]*mixture_f).amax(-2)
        bind=enc['query_roles'][:,None].expand(b,q,q,16)
        message=self.rel_message(torch.cat((p,nu,bind),-1))
        kappa=gamma[:,:,None]*pi[...,1:].amax(-1)*enc['query_mask'][:,:,None]
        w=kappa/kappa.sum(1,keepdim=True).clamp_min(1e-8)
        rel=kappa.amax(1)[...,None]*(w[...,None]*message).sum(1)
        recalled=(self.memory_project(mem)+rel)*enc['query_mask'][...,None]
        return torch.cat((recalled,enc['init'],enc['facts'],z.new_zeros(b,1,d)),1)

    def denoise(self,z,t,enc,ablate=False):
        phase=t.float()[:,None]*self.frequencies[None]
        time=self.time(torch.cat((phase.sin(),phase.cos()),-1))[:,None]
        context=self.condition_at(z,enc,ablate)
        x=z+time
        # Exact query roles identify each slot even at the pure-noise endpoint.
        if self.c.diffusion_role_condition:x=x+self.role_project(enc['query_roles'])
        for block in self.refiner:
            if self.c.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x=checkpoint(block,x,context,use_reentrant=False)
            else:x=block(x,context)
        return self.velocity(self.refine_norm(x))*enc['query_mask'][...,None]

    @torch.no_grad()
    def rollout(self,enc,noise,schedule=(1.,.75,.5,.25,0.),ablate=False):
        z=noise.float()*enc['query_mask'][...,None]
        for t,t_next in zip(schedule,schedule[1:]):
            times=z.new_full((len(z),),t);v=self.denoise(z,times,enc,ablate).float()
            a,s=math.cos(math.pi*t/2),math.sin(math.pi*t/2)
            if t==1:a,s=0.,1.
            clean=a*z-s*v;eps=s*z+a*v
            z=math.cos(math.pi*t_next/2)*clean+math.sin(math.pi*t_next/2)*eps
        return z*enc['query_mask'][...,None]

    def counts(self):
        total=sum(p.numel() for p in self.parameters())
        parts={name:sum(p.numel() for p in m.parameters()) for name,m in self.named_children()}
        return {'parameters':total,'modules':parts,'config':asdict(self.c)}
