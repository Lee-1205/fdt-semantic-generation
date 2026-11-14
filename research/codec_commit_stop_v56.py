"""Codec-only stopping from Commit topology, never a gold output byte count.

This phase reconstructs the entire input. An EOS before the final local Commit
slot cannot represent that contract. The final local slot AND structural tail
remain neural choices. No forced EOS or literal source is used.
"""
import torch
from research.commit_patch_readout_v54 import decode_fixed_logits

def guard_codec_stop(logits):
    if logits.ndim!=3 or logits.shape[1:]!=(16,258) or len(logits)<2:
        raise ValueError('one local Commit and a structural tail required')
    scores=logits.clone()
    scores[:-2,:,256]=-torch.inf
    return scores

def reconstruct_commit_logits(logits,max_symbols=16385):
    return decode_fixed_logits(guard_codec_stop(logits),max_symbols)
