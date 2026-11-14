"""Greedy byte decoding with UTF-8 legality enforced independently of the input."""
import torch

from research.position_sensitive_codec_v46 import PositionSensitiveCodec
from research.text_context_decoder_v2 import BOS
from research.text_diffusion_v1 import EOS, PAD


@torch.no_grad()
def generate_utf8(decoder, context, charge=lambda kind: None):
    batch = len(context[0])
    device = context[0].device
    prefix = torch.full((batch,1), BOS, device=device, dtype=torch.long)
    remaining = torch.zeros(batch, device=device, dtype=torch.long)
    low = torch.full_like(remaining, 0x80)
    high = torch.full_like(remaining, 0xBF)
    done = torch.zeros(batch, device=device, dtype=torch.bool)
    alphabet = torch.arange(PAD+1, device=device)
    two = (alphabet >= 0xC2) & (alphabet <= 0xDF)
    three = (alphabet >= 0xE0) & (alphabet <= 0xEF)
    four = (alphabet >= 0xF0) & (alphabet <= 0xF4)
    outputs = []
    for step in range(decoder.length):
        charge('decoder')
        logits = decoder(prefix, context)[:,-1]
        slots = decoder.length-step
        starter = ((alphabet < 0x80) | (alphabet == EOS)
                   | (two & (slots >= 2)) | (three & (slots >= 3)) | (four & (slots >= 4)))
        continuation = (alphabet >= low[:,None]) & (alphabet <= high[:,None])
        valid = torch.where(remaining[:,None] > 0, continuation, starter)
        valid = torch.where(done[:,None], alphabet == PAD, valid)
        masked = logits.masked_fill(~valid, -torch.inf)
        byte = masked.argmax(-1)
        if not torch.isfinite(masked.gather(-1, byte[:,None])).all():
            raise FloatingPointError('no finite legal UTF-8 decoding choice')
        outputs.append(byte)

        # Restrict the first continuation to exclude overlong encodings,
        # UTF-16 surrogates, and values above U+10FFFF.
        low = torch.where(byte == 0xE0, 0xA0, torch.where(byte == 0xF0, 0x90, 0x80))
        high = torch.where(byte == 0xED, 0x9F, torch.where(byte == 0xF4, 0x8F, 0xBF))
        lead_continuations = two[byte].long() + 2*three[byte].long() + 3*four[byte].long()
        remaining = torch.where(remaining > 0, remaining-1, lead_continuations)
        done |= byte == EOS
        prefix = torch.cat((prefix, byte[:,None]), 1)
        if done.all():
            break
    if (remaining != 0).any():
        raise RuntimeError('UTF-8 decoder reached cap within a codepoint')
    result = torch.stack(outputs, 1)
    if result.shape[1] < decoder.length:
        fill = torch.full((batch, decoder.length-result.shape[1]), PAD, device=device, dtype=torch.long)
        result = torch.cat((result, fill), 1)
    return result


class Utf8PositionCodec(PositionSensitiveCodec):
    """Position-sensitive codec with a legal byte generator at its normal API."""

    def reconstruct(self, commit, charge=lambda kind: None, mode='normal'):
        return generate_utf8(self.decoder, self.context(commit, mode), charge)
