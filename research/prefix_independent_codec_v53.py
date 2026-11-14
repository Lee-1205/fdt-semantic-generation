"""Same three-memory codec; half the reconstruction loss has no answer prefix."""
import torch
from research.local_three_bank_codec_v52 import LocalThreeBankCodec
from research.text_context_decoder_v2 import BOS
from research.text_diffusion_v1 import PAD

class PrefixIndependentCodec(LocalThreeBankCodec):
    def corrupt_teacher_prefix(self,target_ids):
        if not self.training or target_ids.shape[1]>self.input_capacity+1:
            raise ValueError('training-only prefix-free reconstruction')
        prefix=torch.cat((torch.full_like(target_ids[:,:1],BOS),target_ids[:,:-1]),1)
        selected=(prefix>=0)&(prefix<256)
        return prefix.masked_fill(selected,PAD),selected,selected
