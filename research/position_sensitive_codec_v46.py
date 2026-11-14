"""Keep the three memories and make the two Commit slots position sensitive."""
from dataclasses import replace

import torch
from torch import nn

from research.three_memory_codec_v33 import DifferentiableCodecCommit, ThreeMemoryCodec

ADAPTER_NAME = 'memory_codec.position_adapter'
ADAPTER_SCALE = 128.0


class PositionSensitiveCodec(ThreeMemoryCodec):
    def __init__(self, config):
        super().__init__(config)
        if config.meaning_views != 2:
            raise ValueError('two ordered Commit slots')
        self.memory_codec.register_parameter('position_adapter', None)

    def enable_position_adapter(self):
        if self.memory_codec.position_adapter is not None:
            raise RuntimeError('position adapter already installed')
        weight = self.decoder.embedding.weight
        adapter = nn.Parameter(weight.new_zeros(2, weight.shape[-1]+1))
        self.memory_codec.position_adapter = adapter
        return adapter

    def encode_input(self, tokens, *, source_ids, source_versions, model_epoch, disabled_memories=()):
        episode = super().encode_input(
            tokens, source_ids=source_ids, source_versions=source_versions,
            model_epoch=model_epoch, disabled_memories=disabled_memories)
        adapter = self.memory_codec.position_adapter
        if adapter is None or 'contextual' in disabled_memories:
            return episode

        # Summaries of fuzzy values, formed before Commit. The decoder receives
        # neither source bytes nor the per-token states used to form them.
        with torch.autocast(tokens.device.type, enabled=False):
            diagnostics = episode.diagnostics
            valid = tokens != 256
            cut = (valid.sum(1, keepdim=True)+1)//2
            first = torch.arange(tokens.shape[1], device=tokens.device)[None] < cut
            mu = diagnostics['token_memberships'].float()
            values = diagnostics['token_values'].float()
            rho = diagnostics['retrieval_weights'].float()
            confidence = diagnostics['confidence'].float()[...,None]
            bank = episode.memories.contextual
            long_values, _ = bank.means()
            global_read = confidence * (rho @ long_values)
            local_reads = []
            for view, inside in enumerate((first, ~first)):
                mask = valid & inside
                weights = mu * mask[...,None]
                mass = weights.sum(1)[...,None]
                means = (weights.transpose(1,2) @ values) / mass.clamp_min(1e-8)
                summary = confidence[:,view:view+1] * (rho[:,view:view+1] @ means)
                local_reads.append(torch.where(mask.any(1)[:,None,None], summary, global_read[:,view:view+1]))
            local_read = torch.cat(local_reads, 1)
            roles = local_read.new_zeros(*local_read.shape[:2], 16)
            local_z = self.core.condition(torch.cat((local_read, roles), -1))
            global_z = self.core.condition(torch.cat((global_read, roles), -1))
            gain = (ADAPTER_SCALE * adapter[:,0]).tanh()[None,:,None]
            z = episode.commit.reader_tensor() + gain*(local_z-global_z)
        return replace(episode, commit=DifferentiableCodecCommit.from_semantic(
            z, episode.commit.memory_revisions, episode.commit.model_epoch))

    def context(self, commit, mode='normal'):
        values, valid = super().context(commit, mode)
        adapter = self.memory_codec.position_adapter
        if adapter is not None and mode != 'zero':
            values = values + (ADAPTER_SCALE * adapter[:,1:]).tanh().to(values.dtype)[None]
        return values, valid
