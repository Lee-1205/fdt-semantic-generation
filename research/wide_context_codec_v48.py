"""Additional ordered fuzzy contextual summaries with an initially inert decoder read."""
from dataclasses import replace

import torch
from torch import nn

from research.early_byte_codec_v47 import EarlyByteCodec
from research.text_context_decoder_v2 import ContextDecoder
from research.three_memory_codec_v33 import DifferentiableCodecCommit, SealedCodecCommit

CHUNK_BYTES = 16
CHUNKS = 8
LOCAL_GAIN = 'decoder.local_read_gain'
LOCAL_TAG = 'decoder.local_slot_tag'
GAIN_SCALE = 128.0


class LocalMemoryDecoder(ContextDecoder):
    def __init__(self, config, *, length, layers):
        super().__init__(config, length=length, layers=layers)
        self.register_parameter('local_read_gain', None)
        self.register_parameter('local_slot_tag', None)

    def enable_local_read(self):
        if self.local_read_gain is not None:
            raise RuntimeError('local memory already enabled')
        weight = self.embedding.weight
        self.local_read_gain = nn.Parameter(weight.new_zeros(len(self.blocks)))
        self.local_slot_tag = nn.Parameter(weight.new_zeros(CHUNKS, weight.shape[-1]))
        return self.local_read_gain, self.local_slot_tag

    def forward(self, prefix, context):
        if len(context) == 2:
            return super().forward(prefix, context)
        values, valid, local, local_valid = context
        n = prefix.shape[1]
        index = torch.arange(n, device=prefix.device)
        x = self.embedding(prefix)+self.positions(index)[None]
        causal = torch.ones(n,n,device=x.device,dtype=torch.bool).tril()[None,None]
        slots = torch.arange(CHUNKS, device=x.device)
        nearby = (index[:,None]//CHUNK_BYTES-slots[None]).abs() <= 1
        near_valid = nearby[None] & local_valid[:,None]
        nearby_or_all = torch.where(near_valid.any(-1,keepdim=True), near_valid, local_valid[:,None])
        has_local = local_valid.any(-1)
        safe_mask = nearby_or_all | ((~has_local)[:,None,None] & (slots==0)[None,None])
        tagged = local + (GAIN_SCALE*self.local_slot_tag).tanh().to(local.dtype)[None]
        for i,block in enumerate(self.blocks):
            y = block.norms[0](x)
            x = x+block.self_attn(y,y,causal)
            query = block.norms[1](x)
            old_read = block.cross(query,values,valid[:,None,None,:])
            local_read = block.cross(query,tagged,safe_mask[:,None])
            gain = (GAIN_SCALE*self.local_read_gain[i]).tanh().to(x.dtype)
            x = x+old_read
            x = x+gain*local_read*has_local[:,None,None]
            x = x+block.ff(block.norms[2](x))
        return self.head(self.norm(x))


class WideContextCodec(EarlyByteCodec):
    def __init__(self, config):
        super().__init__(config)
        old = self.decoder
        decoder = LocalMemoryDecoder(config, length=old.length, layers=len(old.blocks))
        decoder.load_state_dict(old.state_dict(), strict=True)
        self.decoder = decoder

    def encode_input(self, tokens, *, source_ids, source_versions, model_epoch, disabled_memories=()):
        episode = super().encode_input(
            tokens, source_ids=source_ids, source_versions=source_versions,
            model_epoch=model_epoch, disabled_memories=disabled_memories)
        if self.decoder.local_read_gain is None:
            return episode
        z = episode.commit.reader_tensor()
        local = z.new_zeros(len(z),CHUNKS,z.shape[-1])
        if 'contextual' not in disabled_memories:
            # Each new slot holds a fuzzy summary over up to 16 input bytes.
            # Its Commit carries neither raw bytes, token H, nor a target length.
            with torch.autocast(tokens.device.type, enabled=False):
                valid = tokens != 256
                mu = episode.diagnostics['token_memberships'].float()
                values = episode.diagnostics['token_values'].float()
                remaining = CHUNK_BYTES*CHUNKS-tokens.shape[1]
                weights = torch.nn.functional.pad(mu*valid[...,None], (0,0,0,remaining))
                chunk_values = torch.nn.functional.pad(values, (0,0,0,remaining))
                weights = weights.reshape(len(z),CHUNKS,CHUNK_BYTES,-1)
                chunk_values = chunk_values.reshape(len(z),CHUNKS,CHUNK_BYTES,-1)
                mass = weights.sum(2)
                means = (weights.transpose(-1,-2)@chunk_values) / mass[...,None].clamp_min(1e-8)
                membership = 1-(1-weights).prod(2)
                rho = membership / membership.sum(-1,keepdim=True).clamp_min(1e-8)
                summary = (rho[...,None,:]@means).squeeze(-2)*membership.amax(-1,keepdim=True)
                roles = summary.new_zeros(*summary.shape[:2],16)
                local = self.core.condition(torch.cat((summary,roles),-1))
                local = local*(mass.sum(-1)>0)[...,None]
        commit = DifferentiableCodecCommit.from_semantic(torch.cat((z,local),1),
            episode.commit.memory_revisions, episode.commit.model_epoch)
        return replace(episode, commit=commit)

    def context(self, commit, mode='normal'):
        if type(commit) not in (DifferentiableCodecCommit, SealedCodecCommit):
            raise TypeError('decoder requires a codec Commit')
        z = commit.reader_tensor()
        if z.shape[1] == 2:
            return super().context(commit,mode)
        if z.shape[1] != 2+CHUNKS or self.decoder.local_read_gain is None:
            raise ValueError('ordered local Commit does not match decoder')
        base = DifferentiableCodecCommit.from_semantic(
            z[:,:2],commit.memory_revisions,commit.model_epoch)
        values,valid = super().context(base,mode)
        local = z[:,2:].to(values.device)
        if mode == 'swap':
            local = local.roll(max(1,len(local)//2),0)
        elif mode == 'zero':
            local = torch.zeros_like(local)
        return values,valid,local,(local!=0).any(-1)
