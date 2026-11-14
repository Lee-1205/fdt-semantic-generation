"""Same fuzzy AE and inference; finite training-only p50 hold, without restarting the previous ramp."""
import copy
import torch
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE
from research.owt_prefix_curriculum_v27 import OBJECTIVE as SOURCE_OBJECTIVE
from research.text_context_decoder_v2 import BOS
from research.text_diffusion_v1 import PAD

OBJECTIVE={'clean_CE_weight':.5,'corrupted_prefix_CE_weight':.5,
           'past_byte_mask_probability':.50,'mask_id':PAD,'prefix_length':129,
           'inference_changed':False,'reasoning_diffusion_training':False}

def mask_probability(attempt):
    if type(attempt) is not int or not 1<=attempt<=8192:raise ValueError('finite hold attempt')
    return .50

class PrefixHoldFuzzyAE(PrefixRobustFuzzyAE):
    def __init__(self,c):
        super().__init__(c);self.hold_attempt=1

    def set_training_attempt(self,attempt):
        mask_probability(attempt);self.hold_attempt=attempt

    def corrupt_teacher_prefix(self,target_ids):
        if not self.training:raise RuntimeError('prefix corruption is training-only')
        if target_ids.ndim!=2 or target_ids.shape[1]!=129:raise ValueError('fixed full prefix shape')
        prefix=torch.cat((torch.full_like(target_ids[:,:1],BOS),target_ids[:,:-1]),1)
        uniforms=torch.rand(prefix.shape,device=prefix.device)
        eligible=(prefix>=0)&(prefix<256)
        selected=(uniforms<mask_probability(self.hold_attempt))&eligible
        return prefix.masked_fill(selected,PAD),selected,eligible

    def reconstruction_loss(self,tokens,target_ids,charge=lambda kind:None):
        loss,meaning=super().reconstruction_loss(tokens,target_ids,charge)
        self.last_objective_stats.update(mask_probability=mask_probability(self.hold_attempt),
                                         hold_attempt=self.hold_attempt)
        return loss,meaning
