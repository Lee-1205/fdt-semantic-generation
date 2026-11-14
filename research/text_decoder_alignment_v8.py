"""Prospective interface repair: ordered latent memory and span repetition loss.

CPU candidate only. No runtime gold semantics, forced EOS, or output templates.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.text_context_decoder_v2 import ContextDecoder,BOS
from research.text_diffusion_v1 import PAD


class AlignedContextDecoder(ContextDecoder):
    def __init__(self,config,length=129,layers=2):
        super().__init__(config,length,layers)
        self.memory_positions=nn.Embedding(length,config.z_dim)
        with torch.no_grad():self.memory_positions.weight.copy_(self.positions.weight)

    def import_parent(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.missing_keys!=['memory_positions.weight'] or result.unexpected_keys:raise ValueError('exact decoder ancestry required')
        with torch.no_grad():self.memory_positions.weight.copy_(self.positions.weight)
        return {'preserved_decoder_keys':len(state),'new_parameters':self.memory_positions.weight.numel(),
            'memory_position_initialization':'copy parent query position table; separate subsequent learning'}

    def context(self,prompt,mask,latent,zero_latent=False):
        z=torch.zeros_like(latent) if zero_latent else latent
        pos=self.memory_positions(torch.arange(z.shape[1],device=z.device)).to(z.dtype)
        return super().context(prompt,mask,z+pos[None])

    def training_reconstruction(self,prompt,mask,ids,codec,noise,sigma=.1):
        if not self.training:raise RuntimeError('clean-codec conditioning is training-only')
        if not 0<=sigma<=.5 or noise.shape!=(*ids.shape,codec.shape[1]):raise ValueError('explicit bounded corruption')
        clean=F.embedding(ids,codec)*math.sqrt(codec.shape[1])
        context=self.context(prompt,mask,clean+sigma*noise)
        prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1)
        logits=self(prefix,context).float();valid=ids!=PAD
        loss=F.cross_entropy(logits.transpose(1,2),ids.masked_fill(~valid,-100),ignore_index=-100,reduction='none')
        return loss.sum()/valid.sum()


def span_unlikelihood(logits,ids,negative_spans,beta=10.):
    """Mean -log(1-P(byte span | own/pseudo prefix)); no positive gold answers.

    Caller supplies TRAIN-ONLY (row,start,exclusive_end) word/sentence negatives.
    No rule parser or penalty is installed into runtime generation by this helper.
    """
    if beta<0:raise ValueError('nonnegative coefficient')
    logp=logits.float().log_softmax(-1);terms=[]
    for row,start,end in negative_spans:
        if not 0<=row<len(ids) or not 0<=start<end<=ids.shape[1]:raise ValueError('span range')
        chosen=ids[row,start:end]
        if (chosen>=256).any() or (chosen<0).any():raise ValueError('only actual UTF8 byte spans')
        q=logp[row,start:end].gather(-1,chosen[:,None]).sum().exp().clamp(max=1-1e-6)
        terms.append(-torch.log1p(-q))
    return beta*torch.stack(terms).mean() if terms else logits.sum()*0
