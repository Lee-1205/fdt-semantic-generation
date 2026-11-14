"""V49 length path with separate new position parameters for honest Adam ownership."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.long_memory_codec_v49 import LongMemoryCodec, LongMemoryDecoder, attention_memory, attention_query

class SplitPositions(nn.Embedding):
    def __init__(self, old, length):
        super().__init__(len(old),old.shape[1],_weight=old.detach().clone())
        index=torch.arange(len(old),length,device=old.device)
        frequency=torch.exp(-math.log(10000.)*torch.arange(old.shape[1],device=old.device)/old.shape[1])
        self.extension=nn.Parameter(old[index.remainder(len(old)-1)].clone()+.02*torch.sin(index[:,None]*frequency))

    def forward(self,index):
        original=super().forward(index.clamp_max(self.num_embeddings-1))
        extra=F.embedding((index-self.num_embeddings).clamp_min(0),self.extension)
        return torch.where((index<self.num_embeddings)[...,None],original,extra)

class SplitPositionDecoder(LongMemoryDecoder):
    def extend_positions(self,length):
        if length<=len(self.positions.weight):raise ValueError('long capacity must exceed129')
        self.positions=SplitPositions(self.positions.weight,length)
        self.length=length

    def cached_step(self,token,cache):
        position=cache['position'];context=cache['context']
        index=torch.tensor([position],device=token.device)
        x=self.embedding(token)+self.positions(index)[None]
        if len(context)==4:mask,has=self.local_mask(index,context[3])
        for i,(block,item) in enumerate(zip(self.blocks,cache['layers'])):
            normalized=block.norms[0](x)
            k,v=attention_memory(block.self_attn,normalized)
            item['self_k']=k if item['self_k'] is None else torch.cat((item['self_k'],k),2)
            item['self_v']=v if item['self_v'] is None else torch.cat((item['self_v'],v),2)
            x=x+attention_query(block.self_attn,normalized,item['self_k'],item['self_v'])
            q=block.norms[1](x)
            x=x+attention_query(block.cross,q,*item['global'],context[1][:,None,None,:])
            if len(context)==4:
                extra=attention_query(block.cross,q,*item['local'],mask[:,None])
                x=x+(128*self.local_read_gain[i]).tanh().to(x.dtype)*extra*has[:,None,None]
            x=x+block.ff(block.norms[2](x))
        cache['position']+=1
        return self.head(self.norm(x))

class TrainableLongCodec(LongMemoryCodec):
    def __init__(self,config,input_capacity=16384,chunk_bytes=16):
        super().__init__(config,input_capacity,chunk_bytes)
        old=self.decoder
        self.decoder=SplitPositionDecoder(config,length=old.length,layers=len(old.blocks),chunk_bytes=chunk_bytes)
        self.decoder.load_state_dict(old.state_dict(),strict=True)
