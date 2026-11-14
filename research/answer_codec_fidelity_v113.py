"""Fixed realizer content-byte feedback to theta; E/D weights never train.

The source-only trajectory is still completed before the fixed D is called.
True bytes are loss labels only; no target prefix/length enters generation.
No EOS auxiliary is added; this isolates content fidelity from termination.
"""
import torch
from torch.nn import functional as F
from research.answer_pair_contrast_v112 import PairObjective,PairContrastEngine
from research.three_memory_codec_v33 import DifferentiableCodecCommit


class CodecContentObjective(PairObjective):
    def __call__(self,m,packet,t,noise,charge):
        latent,metrics,z=super().__call__(m,packet,t,noise,charge)
        e=self.engine;e.update.assert_fast();charge('readout')
        logits=e.codec.verbalize(DifferentiableCodecCommit.from_semantic(z,m.memory_revisions,m.model_epoch)).float()
        labels=packet['labels'].to(z.device);flat=labels.flatten();ends=torch.where(flat==256)[0]
        if (logits.shape!=(*labels.shape,258) or len(ends)!=1 or not torch.isfinite(logits).all()):raise ValueError('fixed D loss-only label geometry')
        n=int(ends[0])
        if n<1 or not ((flat[:n]>=0)&(flat[:n]<256)).all() or not (flat[n+1:]==257).all():raise ValueError('bytes/EOS/PAD loss contract')
        ce=F.cross_entropy(logits.reshape(-1,258)[:n],flat[:n])
        if not ce.requires_grad or not torch.isfinite(ce):raise RuntimeError('frozenD must remain differentiable to theta')
        self.last_byte_CE=ce  # ephemeral diagnostic; never serialized or inference state
        metrics.update(byte_CE=float(ce.detach()),language_losses_NOT_used=False,language_forward_calls0=False,
            frozen_D_weights=True,EOS_auxiliary_used=False)
        e.update.assert_fast()
        return latent+e.byte_weight*ce,metrics,z


class PairCodecEngine(PairContrastEngine):
    def __init__(self,*args,byte_weight=.5,**kwargs):
        if byte_weight!=.5:raise ValueError('fixed prospective content weight .5')
        self.byte_weight=byte_weight;super().__init__(*args,**kwargs)
        self.metadata['D_during_training']=True
        self.metadata['frozen_D_content_feedback_v113']=dict(byte_weight=byte_weight,EOS_auxiliary=False,D_weights_frozen=True)
        self.objective=CodecContentObjective(self)
    def finish(self,**kwargs):
        h=super().finish(**kwargs);h['D_during_training']=True
        h['combined_latent_plus_byte_CE']=h['loss']['latent']+self.byte_weight*h['loss']['byte_CE']
        return h
    def payload(self,*,audit=True):
        p=super().payload(audit=audit)
        p.update(schema='answer_codec_fidelity_fullstate_v113',language_losses_used=True,frozen_D_byte_weight=self.byte_weight)
        return p
    def restore(self,p,*,audit=True):
        if p['schema']!='answer_codec_fidelity_fullstate_v113' or not p['language_losses_used'] or p['frozen_D_byte_weight']!=self.byte_weight:
            raise ValueError('exact content feedback objective restore')
        super().restore(dict(p,schema='answer_pair_contrast_fullstate_v112',language_losses_used=False),audit=audit)
