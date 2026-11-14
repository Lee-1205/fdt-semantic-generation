"""Learn query roles/semantics from prompt context, never supply gold at inference."""
import torch
from torch import nn
from torch.nn import functional as F
from fdt_local.model_masked import Core
from research.text_context_decoder_v2 import ContextDecoder

class SemanticReader(nn.Module):
    def __init__(self,config):
        super().__init__();d=config.z_dim
        self.queries=nn.Parameter(torch.randn(4,d)*.02)
        self.blocks=nn.ModuleList([Core(config) for _ in range(2)])
        self.heads=nn.ModuleList([nn.Linear(d,n) for n in (8,8,3,2)])
        self.values=nn.ModuleList([nn.Embedding(n,d) for n in (8,8,3,2)])
        self.types=nn.Parameter(torch.randn(4,d)*.02)

    def forward(self,prompt,mask):
        x=self.queries[None].expand(len(prompt),-1,-1)
        valid=torch.ones(x.shape[:2],device=x.device,dtype=torch.bool)
        for block in self.blocks:x=block(x,prompt,valid,mask)
        logits=[head(x[:,i]) for i,head in enumerate(self.heads)]
        factors=torch.stack([scores.float().softmax(-1)@emb.weight.float()+self.types[i]
                             for i,(scores,emb) in enumerate(zip(logits,self.values))],1)
        return factors,logits

class SemanticGenerator(nn.Module):
    def __init__(self,config):
        super().__init__();self.reader=SemanticReader(config);self.decoder=ContextDecoder(config)

    def context(self,prompt,mask,latent,zero_latent=False):
        factors,logits=self.reader(prompt,mask)
        # Decoder sees predicted soft factors; gold auxiliary labels never enter.
        values,valid=self.decoder.context(factors,torch.ones(factors.shape[:2],device=factors.device,dtype=torch.bool),latent,zero_latent)
        return (values,valid),logits

    def loss(self,ids,context,logits,gold):
        ar=self.decoder.loss(ids,context)
        losses=[F.cross_entropy(scores.float(),gold[:,i]) for i,scores in enumerate(logits)]
        auxiliary=losses[0]+losses[1]+losses[2]+.25*losses[3]
        return ar+.5*auxiliary,ar,auxiliary
