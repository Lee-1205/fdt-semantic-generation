"""TRAIN-only alignment of denoised meaning with the frozen language endpoint.

V95 shows E(answer) preserves all31 canonical answer prefixes but does NOT
specify a correct within-patch neural EOS. Pure latent MSE inherits that error.
Supervise bytes and EOS separately through frozen D, training ONLY thought.
No new output weights, inferred-length crop, answer template or inference API.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from research.three_memory_codec_v33 import DifferentiableCodecCommit


@dataclass(frozen=True)
class ThoughtLanguageLossConfig:
    byte_weight:float=.5
    EOS_weight:float=.5

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in (self.byte_weight,self.EOS_weight)):
            raise ValueError('positive fixed training-only language weights')


class ThoughtLanguageObjective:
    def __init__(self,thought,codec,update,codec_SHA,config=ThoughtLanguageLossConfig()):
        if update.boundary.reasoner is not thought or update.boundary.codec is not codec:
            raise ValueError('actual thought-only optimizer boundary required')
        self.thought,self.codec,self.update,self.codec_SHA,self.config=thought,codec,update,codec_SHA,config
        update.assert_fast()

    def __call__(self,source,packet,t,noise,charge=lambda kind:None):
        if not self.thought.training:raise RuntimeError('target/EOS supervision TRAIN-only; eval uses target-free sample')
        self.update.assert_fast()
        required={'schema','answer_SHA','source_codec_SHA','semantic_target','valid_extent_mask','labels','model_epoch','target_condition_features_never_used'}
        if (set(packet)!=required or packet['schema']!='LOSS_ONLY_answer_meaning_v95' or packet['source_codec_SHA']!=self.codec_SHA
                or packet['model_epoch']!=source.model_epoch or not packet['target_condition_features_never_used']):raise ValueError('separate frozen answer loss packet')
        if any(packet[k].requires_grad for k in ('semantic_target','valid_extent_mask','labels')):raise ValueError('frozen target tensors')
        device=source.source.device
        target=packet['semantic_target'].to(device);mask=packet['valid_extent_mask'].to(device);labels=packet['labels'].to(device)
        if labels.dtype!=torch.long or labels.ndim!=2 or len(target)!=1 or labels.shape!=(target.shape[1]-1,16):raise ValueError('one genuine ordered byte readout loss geometry')
        y=labels.flatten();eos=torch.where(y==256)[0]
        if (len(eos)!=1 or int(eos[0])<1 or not ((y[:eos[0]]>=0)&(y[:eos[0]]<256)).all() or not (y[eos[0]+1:]==257).all()):raise ValueError('exactly one loss-side EOS; only later PAD ignored')
        charge('denoise');latent_loss,clean=self.thought.training_objective(source,target,mask,t,noise)
        # Frozen parameters; retain d(language loss)/d(clean meaning). Never
        # put this call under no_grad or detach the predicted clean state.
        charge('readout');logits=self.codec.verbalize(DifferentiableCodecCommit.from_semantic(clean,source.memory_revisions,source.model_epoch))
        if logits.shape!=(*labels.shape,258) or not torch.isfinite(logits).all():raise FloatingPointError('finite genuine258-class frozenD output')
        z=logits.float().reshape(-1,258);end=int(eos[0])
        byte_loss=F.cross_entropy(z[:end],y[:end]);EOS_loss=F.cross_entropy(z[end:end+1],y[end:end+1])
        loss=latent_loss+self.config.byte_weight*byte_loss+self.config.EOS_weight*EOS_loss
        if not torch.isfinite(loss):raise FloatingPointError('finite thought/language alignment objective')
        self.update.assert_fast()
        return loss,dict(latent=float(latent_loss.detach()),byte_CE=float(byte_loss.detach()),EOS_CE=float(EOS_loss.detach()),
            active_bytes=end,EOS_count=1,PAD_ignored=int((y==257).sum()),no_source_encoder_call=True,codec_weights_trainable=False)
