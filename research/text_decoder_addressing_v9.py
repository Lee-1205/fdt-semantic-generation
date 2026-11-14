"""Explicit query-index to latent-index attention prior, not a gold length rule."""
import torch
from research.text_decoder_alignment_training_v8 import AlignedLanguage
from research.text_decoder_alignment_v8 import AlignedContextDecoder

ADDRESS_POLICY={'kind':'same-index soft cross-attention prior','boost':'log(valid_memory_count-1)',
    'uniform_score_own_slot_prior':0.5,'hard_mask_other_memory':False,
    'uses_gold_length_or_positions':False,'new_learnable_parameters':0}


class AddressedContextDecoder(AlignedContextDecoder):
    def address_bias(self,valid,queries):
        batch,memory=valid.shape;offset=memory-self.length
        if offset<1 or queries>self.length:raise ValueError('fixed latent capacity/context shape')
        bias=torch.zeros(batch,1,queries,memory,device=valid.device,dtype=torch.float32)
        bias.masked_fill_(~valid[:,None,None,:],-float('inf'))
        address=(offset+torch.arange(queries,device=valid.device))[None,None,:,None].expand(batch,1,queries,1)
        boost=(valid.sum(-1).float()-1).clamp_min(1).log()[:,None,None,None].expand(batch,1,queries,1)
        return bias.scatter_add(-1,address,boost)

    def forward(self,prefix,context):
        values,valid=context;n=prefix.shape[1]
        x=self.embedding(prefix)+self.positions(torch.arange(n,device=prefix.device))[None]
        causal=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()[None,None]
        bias=self.address_bias(valid,n)
        for block in self.blocks:
            y=block.norms[0](x);x=x+block.self_attn(y,y,causal)
            x=x+block.cross(block.norms[1](x),values,bias)
            x=x+block.ff(block.norms[2](x))
        return self.head(self.norm(x))


class AddressedLanguage(AlignedLanguage):
    def __init__(self,core,policy):
        super().__init__(core,policy);self.decoder=AddressedContextDecoder(core,policy.target_bytes_with_eos)
        self.original_trainable={n for n,p in self.named_parameters() if p.requires_grad}

    def import_weights(self,state):
        self.load_state_dict(state,strict=True)  # retain learned V8 positions; no reinitialization
