"""Inference-only execution of the SAME finite-window Transformer and banks.

Carries per-layer rotated K/V across compute chunks. Full input H is then used
by the original whole-document fuzzy N/D and Commit construction, never reset
per chunk. No weight/table resize, source-text retrieval or Exact content.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64

class StreamingFuzzyCodecV69(FuzzyCodecV64):
    def __init__(self,*args,stream_capacity=131072,compute_chunk=2048,**kwargs):
        super().__init__(*args,**kwargs)
        self.stream_capacity=stream_capacity;self.compute_chunk=compute_chunk
        self._dense_reference=False;self.stream_guard=None;self.stream_stats={}

    @contextmanager
    def dense_reference(self):
        if self._dense_reference:raise ValueError('nested dense reference')
        self._dense_reference=True
        try:yield
        finally:self._dense_reference=False

    def token_features(self,tokens):
        if self._dense_reference:return super().token_features(tokens)
        if self.training or torch.is_grad_enabled():raise ValueError('streaming is inference only; no detached training shortcut')
        if tokens.ndim!=2 or tokens.dtype!=torch.long or not 1<=tokens.shape[1]<=self.stream_capacity:raise ValueError('new bounded stream input')
        if ((tokens<0)|(tokens>256)).any():raise ValueError('input byte range')
        valid=tokens!=256
        if not valid.any(1).all() or ((~valid).long().cumsum(1).bool()&valid).any():raise ValueError('contiguous valid input')
        language=self.core.language;window=self.c.window;cache=[None]*len(language.blocks);parts=[];chunks=0
        for start in range(0,tokens.shape[1],self.compute_chunk):
            end=min(tokens.shape[1],start+self.compute_chunk);inside=valid[:,start:end];x=language.embedding(tokens[:,start:end])
            qpos=torch.arange(start,end,device=tokens.device)
            for i,block in enumerate(language.blocks):
                a=block.attn;n1=block.n1(x);batch,count,dimension=x.shape
                q=a.q(n1).view(batch,count,a.h,a.w).transpose(1,2)
                k=a.k(n1).view(batch,count,a.h,a.w).transpose(1,2)
                v=a.v(n1).view(batch,count,a.h,a.w).transpose(1,2)
                freq=10000**(-torch.arange(0,a.w,2,device=x.device).float()/a.w)
                phase=qpos.float()[:,None]*freq
                def rotate(t):
                    left,right=t.float()[...,0::2],t.float()[...,1::2]
                    return torch.stack((left*phase.cos()-right*phase.sin(),left*phase.sin()+right*phase.cos()),-1).flatten(-2).to(t.dtype)
                q,k=rotate(q),rotate(k)
                previous=cache[i]
                if previous is None:keys,values,kpos,kvalid=k,v,qpos,inside
                else:
                    pk,pv,pp,pm=previous;keys=torch.cat((pk,k),2);values=torch.cat((pv,v),2);kpos=torch.cat((pp,qpos));kvalid=torch.cat((pm,inside),1)
                band=(qpos[:,None]>=kpos[None,:])&((qpos[:,None]-kpos[None,:])<window)
                eye=qpos[:,None]==kpos[None,:]
                mask=band[None,None]&(kvalid[:,None,None,:]|eye[None,None])
                output=F.scaled_dot_product_attention(q,keys,values,attn_mask=mask,dropout_p=0.)
                x=x+a.o(output.transpose(1,2).reshape(batch,count,dimension))
                x=(x+block.ff(block.n2(x)))*inside[...,None]
                keep=min(max(0,window-1),keys.shape[2])
                cache[i]=(keys[:,:,-keep:].contiguous(),values[:,:,-keep:].contiguous(),kpos[-keep:].clone(),kvalid[:,-keep:].clone()) if keep else None
                if self.stream_guard:self.stream_guard('layer',start,end,i)
            parts.append(x.float());chunks+=1
            if self.stream_guard:self.stream_guard('chunk',start,end,len(language.blocks))
        h=torch.cat(parts,1)
        weights=(h@language.pool.float().T).masked_fill(~valid[...,None],-1e4).softmax(1)
        pooled=torch.einsum('blv,bld->bvd',weights,h)
        self.stream_stats=dict(chunks=chunks,compute_chunk=self.compute_chunk,language_window=window,KV_carry_per_layer=max(0,window-1),
            actual_input_bytes=int(valid.sum()),whole_document_pool=True,whole_document_fuzzy_banks=True,per_chunk_bank_reset=False,
            trained_capacity=self.input_capacity,test_capacity=self.stream_capacity,weight_or_position_table_resize=False)
        return h,pooled,valid
