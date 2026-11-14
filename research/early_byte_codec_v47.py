"""Guarded reconstruction with additional supervision on the first 16 bytes."""
import torch
from torch.nn import functional as F
from research.text_diffusion_v1 import PAD
from research.utf8_codec_decode import Utf8PositionCodec

EARLY_BYTES = 16
EARLY_WEIGHT = 4.0


def early_byte_average(per_byte, active):
    positions = torch.arange(per_byte.shape[-1], device=per_byte.device)
    weight = torch.where(positions < EARLY_BYTES, EARLY_WEIGHT, 1.0) * active
    return (per_byte * weight).sum() / weight.sum()


class EarlyByteCodec(Utf8PositionCodec):
    def reconstruction_loss(self, tokens, target_ids, *, source_ids, source_versions, model_epoch, charge=lambda kind: None):
        if not self.training: raise RuntimeError('targets are training-only')
        if tokens.ndim != 2 or target_ids.shape != (len(tokens), 129) or target_ids.dtype != torch.long:
            raise ValueError('same-input reconstruction target shape')
        expected = torch.full_like(target_ids, PAD)
        expected[:, :tokens.shape[1]] = tokens.masked_fill(tokens == 256, PAD)
        expected.scatter_(1, (tokens != 256).sum(1, keepdim=True), 256)
        if not torch.equal(expected, target_ids):
            raise ValueError('codec target must be the SAME input bytes plus EOS')
        charge('encode')
        episode = self.encode_input(tokens, source_ids=source_ids, source_versions=source_versions, model_epoch=model_epoch)
        charge('decoder'); clean_logits = self.teacher_logits(episode.commit, target_ids)
        prefix, selected, eligible = self.corrupt_teacher_prefix(target_ids)
        charge('decoder'); corrupt_logits = self.decoder(prefix, self.context(episode.commit)).float()
        active = target_ids != PAD; labels = target_ids.masked_fill(~active, -100); count = active.sum()
        clean = F.cross_entropy(clean_logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
        corrupt = F.cross_entropy(corrupt_logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
        clean = early_byte_average(clean, active)
        corrupt = early_byte_average(corrupt, active)
        self.last_objective_stats = {
            'clean_CE': float(clean.detach()), 'corrupted_prefix_CE': float(corrupt.detach()),
            'masked_prefix_bytes': int(selected.sum()), 'eligible_prefix_bytes': int(eligible.sum()),
            'active_target_symbols_per_pass': int(count), 'decoder_supervised_symbol_occurrences': 2*int(count),
            'denoise_calls': 0, 'copy_bytes': 0,
        }
        return .5*clean + .5*corrupt, episode

