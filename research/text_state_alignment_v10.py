"""CPU candidate: train VP/language tail through the learned, frozen V9 interface.

The target remains a byte-code sequence, not an independently established semantic
or conscious state. No new execution allocation or optimizer driver is defined here.
"""
import math
import torch
from torch.nn import functional as F
from research.text_decoder_addressing_v9 import AddressedLanguage
from research.text_diffusion_v1 import PAD
from research.text_context_decoder_v2 import BOS

STATE_ALIGNMENT_POLICY={
    'phase':'state_alignment','decoder_frozen':True,'decoder_CE_weight':0.25,
    'direct_prompt_to_decoder_during_training':False,
    'decoder_gradient_to_predicted_clean':True,
    'new_parameters':0,'runtime_gold_features':False,
    'target_representation':'existing fixed orthogonal UTF8-byte/EOS codec',
    'independent_semantic_state_achieved':False,
    'training_calls_per_microbatch':{'encode':1,'denoise':1,'decoder':1},
}

class StateAlignedLanguage(AddressedLanguage):
    def __init__(self,core,policy):
        super().__init__(core,policy)
        self.set_phase('state_alignment')

    def set_phase(self,phase):
        if phase!='state_alignment':raise ValueError('single explicit state-alignment phase')
        self.phase=phase
        for name,param in self.named_parameters():
            param.requires_grad_(name in self.original_trainable and name.startswith('vp.'))

    def training_loss(self,tokens,ids,t,noise,unused_proposal=None,charge=lambda kind:None):
        if not self.training:raise RuntimeError('gold is training-only')
        if ids.shape!=(len(tokens),self.policy.target_bytes_with_eos):raise ValueError('fixed target capacity')
        if noise.shape!=(*ids.shape,self.core_config.z_dim) or t.shape!=(len(ids),):raise ValueError('objective noise/time shape')
        if not torch.isfinite(t).all() or (t<0).any() or (t>1).any():raise ValueError('VP time in[0,1]')
        active=ids!=PAD
        if not active.any(-1).all():raise ValueError('empty target')
        charge('encode');encoded=self.vp.encode(tokens)
        clean=self.vp.target(ids);angle=t[:,None,None]*math.pi/2;a,s=angle.cos(),angle.sin()
        z=torch.where(active[...,None],a*clean+s*noise,noise)
        charge('denoise');velocity=self.vp.denoise(z,t,encoded).float()
        predicted=a*z-s*velocity
        norm=active.sum();target_velocity=a*noise-s*clean
        vmse=((velocity-target_velocity).square().mean(-1)*active).sum()/norm
        cmse=((predicted-clean).square().mean(-1)*active).sum()/norm
        weights=active*torch.where(ids==256,self.policy.training_eos_weight,1.)
        vp_ce=(F.cross_entropy(self.vp.readout(predicted).transpose(1,2),ids,reduction='none')*weights).sum()/norm
        # Parameters are frozen; autograd through its input remains enabled.
        # Raw context is zero, so the loss cannot train a direct prompt bypass.
        context=self.decoder.context(encoded[0]*0,encoded[1],predicted)
        prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1)
        charge('decoder');logits=self.decoder(prefix,context).float()
        ce=(F.cross_entropy(logits.transpose(1,2),ids.masked_fill(~active,-100),ignore_index=-100,reduction='none')*weights).sum()/norm
        loss=vmse+.25*cmse+.5*vp_ce+STATE_ALIGNMENT_POLICY['decoder_CE_weight']*ce
        return loss,{'velocity_mse':vmse,'clean_mse':cmse,'VP_byte_ce':vp_ce,
            'AR_byte_ce':ce,'span_UL':ce*0,'loss':loss,'AR_logits':logits,'predicted_clean':predicted}
