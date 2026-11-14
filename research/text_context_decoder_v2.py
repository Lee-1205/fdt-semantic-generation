"""Learned causal byte discretizer over frozen VP proposals and raw prompt context.

Joint byte modeling addresses observed invalid UTF8/marginal mixtures. This is a
hybrid autoregressive realizer, not proof that diffusion alone generated text.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from fdt_local.model_masked import Core,RMS
from research.text_diffusion_v1 import EOS,PAD,VOCAB
BOS=258

class ContextDecoder(nn.Module):
    def __init__(self,config,length=40,layers=2):
        super().__init__();d=config.z_dim;self.length=length
        self.embedding=nn.Embedding(259,d);self.positions=nn.Embedding(length,d)
        self.blocks=nn.ModuleList([Core(config) for _ in range(layers)])
        self.norm=RMS(d);self.head=nn.Linear(d,VOCAB)
        self.context_types=nn.Parameter(torch.randn(2,d)*.02)

    def context(self,prompt,mask,latent,zero_latent=False):
        z=torch.zeros_like(latent) if zero_latent else latent
        values=torch.cat((prompt+self.context_types[0],z+self.context_types[1]),1)
        valid=torch.cat((mask,torch.ones(z.shape[:2],device=z.device,dtype=torch.bool)),1)
        return values,valid

    def forward(self,prefix,context):
        values,mask=context; n=prefix.shape[1]
        x=self.embedding(prefix)+self.positions(torch.arange(n,device=prefix.device))[None]
        causal=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()[None,None]
        for block in self.blocks:
            y=block.norms[0](x);x=x+block.self_attn(y,y,causal)
            x=x+block.cross(block.norms[1](x),values,mask[:,None,None,:])
            x=x+block.ff(block.norms[2](x))
        return self.head(self.norm(x))

    def loss(self,ids,context):
        prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1)
        logits=self.forward(prefix,context).float()
        weights=torch.where(ids==PAD,.1,1.)
        return (F.cross_entropy(logits.transpose(1,2),ids,reduction='none')*weights).sum()/weights.sum()

    @torch.no_grad()
    def generate(self,context,charge=lambda kind:None):
        batch=len(context[0]);device=context[0].device
        prefix=torch.full((batch,1),BOS,device=device,dtype=torch.long)
        done=torch.zeros(batch,device=device,dtype=torch.bool);outputs=[]
        for _ in range(self.length):
            charge('decoder'); ids=self.forward(prefix,context)[:,-1].argmax(-1)
            ids=torch.where(done,PAD,ids);outputs.append(ids);done=done|(ids==EOS)
            prefix=torch.cat((prefix,ids[:,None]),1)
            if done.all():break
        while len(outputs)<self.length: outputs.append(torch.full((batch,),PAD,device=device,dtype=torch.long))
        return torch.stack(outputs,1)
