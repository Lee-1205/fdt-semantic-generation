"""Same fuzzy AE/runtime, train-only clean + corrupted past-prefix reconstruction.

No new parameters or target features at inference. Full fixed-shape draws keep
future target values/length from altering past corruption randomness.
"""
import torch
from torch.nn import functional as F
from research.owt_fuzzy_reconstruction_v12 import FuzzySentenceAE
from research.text_context_decoder_v2 import BOS
from research.text_diffusion_v1 import PAD

OBJECTIVE={'clean_CE_weight':.5,'corrupted_prefix_CE_weight':.5,'past_byte_mask_probability':.15,
           'mask_id':PAD,'prefix_length':129,'inference_changed':False,'reasoning_diffusion_training':False}

class PrefixRobustFuzzyAE(FuzzySentenceAE):
    def __init__(self,c):
        super().__init__(c);self.last_objective_stats=None

    def corrupt_teacher_prefix(self,target_ids):
        if not self.training:raise RuntimeError('prefix corruption is training-only')
        if target_ids.ndim!=2 or target_ids.shape[1]!=129:raise ValueError('fixed full prefix shape')
        prefix=torch.cat((torch.full_like(target_ids[:,:1],BOS),target_ids[:,:-1]),1)
        # Draw before eligibility, with full fixed shape, not target-dependent lengths.
        uniforms=torch.rand(prefix.shape,device=prefix.device)
        eligible=(prefix>=0)&(prefix<256)
        selected=(uniforms<OBJECTIVE['past_byte_mask_probability'])&eligible
        return prefix.masked_fill(selected,PAD),selected,eligible

    def reconstruction_loss(self,tokens,target_ids,charge=lambda kind:None):
        self.last_objective_stats=None
        if not self.training:raise RuntimeError('reconstruction targets training-only')
        charge('encode');meaning=self.extract(tokens)
        charge('decoder');clean_logits=self.teacher_logits(meaning,target_ids)
        prefix,selected,eligible=self.corrupt_teacher_prefix(target_ids)
        charge('decoder');corrupt_logits=self.decoder(prefix,self.context(meaning)).float()
        active=target_ids!=PAD;labels=target_ids.masked_fill(~active,-100);count=active.sum()
        clean=F.cross_entropy(clean_logits.transpose(1,2),labels,ignore_index=-100,reduction='sum')/count
        corrupt=F.cross_entropy(corrupt_logits.transpose(1,2),labels,ignore_index=-100,reduction='sum')/count
        loss=.5*clean+.5*corrupt
        self.last_objective_stats={'clean_CE':float(clean.detach()),'corrupted_prefix_CE':float(corrupt.detach()),
            'masked_prefix_bytes':int(selected.sum()),'eligible_prefix_bytes':int(eligible.sum()),
            'active_target_symbols_per_pass':int(count),'decoder_supervised_symbol_occurrences':2*int(count)}
        return loss,meaning
