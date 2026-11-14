"""Paired codec realization + ordered memory + strong train-only span negatives."""
import math
import torch
from torch.nn import functional as F
from research.text_language_continuation_v7 import LanguageContinuation,ContinuationConfig
from research.text_decoder_alignment_v8 import AlignedContextDecoder
from research.text_context_decoder_v2 import BOS
from research.text_diffusion_v1 import PAD


def vector_span_loss(logits,ids,spans,beta=10.):
    if not spans:return logits.sum()*0
    rows=torch.tensor([s[0] for s in spans],device=logits.device)
    positions=torch.arange(ids.shape[1],device=logits.device)[None]
    starts=torch.tensor([s[1] for s in spans],device=logits.device)[:,None]
    ends=torch.tensor([s[2] for s in spans],device=logits.device)[:,None]
    selected=logits.float().log_softmax(-1).gather(-1,ids[...,None]).squeeze(-1)
    logq=(selected[rows]*((positions>=starts)&(positions<ends))).sum(-1)
    return -beta*torch.log1p(-logq.exp().clamp(max=1-1e-6)).mean()


class AlignedLanguage(LanguageContinuation):
    def __init__(self,core,policy=ContinuationConfig()):
        super().__init__(core,policy);self.decoder=AlignedContextDecoder(core,policy.target_bytes_with_eos)
        self.original_trainable={n for n,p in self.named_parameters() if p.requires_grad}
        self.phase='joint';self.training_negative_batch=None;self.span_beta=0.0

    def import_weights(self,state):
        missing=self.load_state_dict(state,strict=False)
        if missing.missing_keys!=['decoder.memory_positions.weight'] or missing.unexpected_keys:raise ValueError('exact parent model keys')
        with torch.no_grad():self.decoder.memory_positions.weight.zero_()

    def set_phase(self,phase):
        if phase not in ('reconstruction','joint'):raise ValueError('phase')
        self.phase=phase
        for name,p in self.named_parameters():p.requires_grad_(name in self.original_trainable and (phase=='joint' or name.startswith('decoder.')))

    def training_loss(self,tokens,ids,t,noise,unused_proposal,charge=lambda kind:None):
        if not self.training or self.training_negative_batch is None:raise RuntimeError('explicit training targets/negatives only')
        charge('encode');encoded=self.vp.encode(tokens);active=ids!=PAD;norm=active.sum()
        clean=self.vp.target(ids);vmse=cmse=byte_ce=clean.sum()*0
        if self.phase=='joint':
            angle=t[:,None,None]*math.pi/2;a,s=angle.cos(),angle.sin();z=a*clean+s*noise
            charge('denoise');velocity=self.vp.denoise(z,t,encoded).float();predicted=a*z-s*velocity
            vmse=((velocity-(a*noise-s*clean)).square().mean(-1)*active).sum()/norm
            cmse=((predicted-clean).square().mean(-1)*active).sum()/norm
            weights=active*torch.where(ids==256,self.policy.training_eos_weight,1.)
            byte_ce=(F.cross_entropy(self.vp.readout(predicted).transpose(1,2),ids,reduction='none')*weights).sum()/norm
        # Decoder targets and conditioning describe the SAME training text.
        # No unrelated free sample is forced to realize an arbitrary gold suffix.
        # Inactive memory is random, not a PAD-codec length oracle.
        augmented=torch.where(active[...,None],clean+(.15*t[:,None,None])*noise,noise)
        keep=(torch.rand(len(ids),device=ids.device)>=.5)[:,None,None]
        prompt=encoded[0]*keep;context=self.decoder.context(prompt,encoded[1],augmented.detach())
        prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1)
        charge('decoder');logits=self.decoder(prefix,context).float()
        weights=active*torch.where(ids==256,self.policy.training_eos_weight,1.)
        ar_ce=(F.cross_entropy(logits.transpose(1,2),ids.masked_fill(~active,-100),ignore_index=-100,reduction='none')*weights).sum()/norm
        ul=ar_ce*0
        if self.span_beta>0:
            negative_ids,spans,micro_normalization=self.training_negative_batch
            prefix=torch.cat((torch.full_like(negative_ids[:,:1],BOS),negative_ids[:,:-1]),1)
            charge('decoder');negative_logits=self.decoder(prefix,context).float()
            ul=vector_span_loss(negative_logits,negative_ids,spans,beta=self.span_beta)*micro_normalization
        total=vmse+.25*cmse+.5*byte_ce+ar_ce+ul
        return total,{'velocity_mse':vmse,'clean_mse':cmse,'VP_byte_ce':byte_ce,'AR_byte_ce':ar_ce,
            'span_UL':ul,'loss':total,'AR_logits':logits}
