"""Sentence -> original fuzzy-anchor extraction -> neural sentence reconstruction.

No reasoning rollout or output-meaning diffusion in this first cycle. Fuzzy mu
is nonexclusive; rho is its normalized retrieval weight, not the fuzzy membership.
"""
import torch
from torch import nn
from torch.nn import functional as F
from fdt_local.model_masked import FDT100
from research.text_context_decoder_v2 import ContextDecoder,BOS
from research.text_diffusion_v1 import PAD

class FuzzySentenceAE(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.core=FDT100(c);self.decoder=ContextDecoder(c,length=129)
        self.requires_grad_(False)
        for b in self.core.language.blocks[-2:]:b.requires_grad_(True)
        self.core.language.pool.requires_grad_(True)
        for name in ('phrase_router','value','condition'):getattr(self.core,name).requires_grad_(True)
        self.core.anchors.requires_grad_(True);self.core.prior.requires_grad_(True);self.decoder.requires_grad_(True)

    def extract(self,tokens):
        if tokens.ndim!=2 or tokens.shape[1]>128:raise ValueError('sentence capacity')
        h=self.core.language(tokens[:,None])[:,0]
        mu,rho,gamma=self.core.phrase_router(h,self.core.anchors)
        # Existing fuzzy routing and anchor prior/value/condition modules reused.
        # No external entity roles are provided for these unannotated sentences.
        with torch.autocast(tokens.device.type,enabled=False):
            union=1-(1-mu).prod(1);roles=h.new_zeros(len(tokens),16,dtype=torch.float32)
            value=self.core.value(torch.cat((h.float().mean(1),roles),-1))
            memory=torch.where(union[...,None]>1e-8,value[:,None].expand(-1,self.c.anchors,-1),torch.zeros_like(value[:,None]))
            prior=gamma[...,None]*(rho@self.core.prior.float())
            contextual=gamma[...,None]*(rho@memory.float())
            views=self.core.condition(torch.cat((contextual+prior,roles[:,None].expand(-1,self.c.meaning_views,-1)), -1))
        return {'views':views,'memberships':mu,'union_memberships':union,'retrieval_weights':rho,'confidence':gamma}

    def context(self,meaning,mode='normal'):
        views=meaning['views']
        if mode=='swap':views=views.roll(max(1,len(views)//2),0)
        elif mode=='zero':views=torch.zeros_like(views)
        elif mode!='normal':raise ValueError('explicit control')
        mask=torch.ones(views.shape[:2],device=views.device,dtype=torch.bool)
        return self.decoder.context(views,mask,views.new_empty(len(views),0,views.shape[-1]))

    def teacher_logits(self,meaning,target_ids,mode='normal'):
        prefix=torch.cat((torch.full_like(target_ids[:,:1],BOS),target_ids[:,:-1]),1)
        return self.decoder(prefix,self.context(meaning,mode)).float()

    def reconstruction_loss(self,tokens,target_ids,charge=lambda kind:None):
        if not self.training:raise RuntimeError('reconstruction targets training-only')
        charge('encode');meaning=self.extract(tokens);charge('decoder');logits=self.teacher_logits(meaning,target_ids)
        active=target_ids!=PAD
        loss=F.cross_entropy(logits.transpose(1,2),target_ids.masked_fill(~active,-100),ignore_index=-100,reduction='none')
        return loss.sum()/active.sum(),meaning

    @torch.no_grad()
    def reconstruct(self,meaning,charge=lambda kind:None,mode='normal'):
        return self.decoder.generate(self.context(meaning,mode),charge)

    def warmstart(self,typed,language):
        self.core.load_state_dict(typed,strict=True)
        self.core.language.load_state_dict({k[len('vp.language.'):]:v for k,v in language.items() if k.startswith('vp.language.')},strict=True)
        # V9 byte-index addressing and its extra position table are not used for
        # an unordered fuzzy semantic memory. All source files remain preserved.
        self.decoder.load_state_dict({k[len('decoder.'):]:v for k,v in language.items() if k.startswith('decoder.') and k!='decoder.memory_positions.weight'},strict=True)
        return {'typed_core':'strict original full model','language':'V9 language weights',
            'decoder':'V9 34 original tensors; no byte-index prior on semantic views',
            'fuzzy_router':'original Router/mu/rho/topk/anchors/prior/value/condition',
            'reasoning_or_diffusion_trained':False,'new_parameters':0,'optimizer':'fresh composed AE AdamW, source AdamRNG preserved'}
