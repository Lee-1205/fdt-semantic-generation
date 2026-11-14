"""TRAIN-only gradient through the actual target-free16step meaning trajectory.

Targets enter ONLY endpoint losses and a small VP auxiliary branch. The main
trajectory consumes only input meaning/memories and independent noise. Frozen
language output parameters remain untouched; FinalMeaning detachment remains
mandatory for inference but would incorrectly sever this training gradient.
"""
from dataclasses import asdict,dataclass
import math
import torch
from torch.nn import functional as F
from research.memory_thought_diffusion_v94 import vp
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.thought_grid_tail_v104 import GridThoughtTrainEngine


def training_rollout(thought,source,noise,steps,charge=lambda kind:None):
    if not thought.training:raise RuntimeError('differentiable trajectory TRAIN-only')
    if type(steps)is not int or not 1<=steps<=thought.c.max_steps:raise ValueError('fixed bounded steps')
    centre,scale=thought.initial_prior(source);z=thought.initial_state(source,noise)
    for i in range(steps):
        t=z.new_full((len(z),),1-i/steps);nt=z.new_full((len(z),),1-(i+1)/steps)
        a,s=vp(t);na,ns=vp(nt);rt=(z-centre)/scale
        charge('denoise');v=thought(z,t,source)
        clean=a*rt-s*v;epsilon=s*rt+a*v
        z=centre+scale*(na*clean+ns*epsilon)
    return z


@dataclass(frozen=True)
class RuntimeLossConfig:
    steps:int=16
    body_weight:float=.25
    tail_weight:float=.05
    byte_weight:float=.5
    EOS_weight:float=.5
    VP_aux_weight:float=.1
    def __post_init__(self):
        if type(self.steps)is not int or not 1<=self.steps<=16 or any(not math.isfinite(x) or x<=0 for x in (self.body_weight,self.tail_weight,self.byte_weight,self.EOS_weight,self.VP_aux_weight)):
            raise ValueError('fixed positive endpoint/VP objective and steps')


class RuntimeAlignedObjective:
    def __init__(self,thought,codec,update,codec_SHA,config=RuntimeLossConfig()):
        if update.boundary.reasoner is not thought or update.boundary.codec is not codec:raise ValueError('exact thought-only optimizer ownership')
        self.thought,self.codec,self.update,self.codec_SHA,self.config=thought,codec,update,codec_SHA,config
        if config.steps>thought.c.max_steps:raise ValueError('trajectory capacity')

    def __call__(self,m,packet,t,noise,charge=lambda kind:None):
        if not self.thought.training:raise RuntimeError('no target packet at inference')
        self.update.assert_fast();required={'schema','answer_SHA','source_codec_SHA','semantic_target','valid_extent_mask','labels','model_epoch','target_condition_features_never_used'}
        if (set(packet)!=required or packet['schema']!='LOSS_ONLY_answer_meaning_v95' or packet['source_codec_SHA']!=self.codec_SHA
                or packet['model_epoch']!=m.model_epoch or not packet['target_condition_features_never_used']):raise ValueError('strict loss-only target packet')
        if any(packet[k].requires_grad for k in ('semantic_target','valid_extent_mask','labels')):raise ValueError('frozen target')
        # The FULL main trajectory runs before looking at target values. Its
        # function signature has no target, extent, role or answer-length input.
        final=training_rollout(self.thought,m,noise,self.config.steps,charge)
        target=packet['semantic_target'].to(final.device);mask=packet['valid_extent_mask'].to(final.device);labels=packet['labels'].to(final.device)
        if final.shape!=target.shape or mask.shape!=final.shape[:2] or mask.dtype!=torch.bool or not mask.any() or labels.dtype!=torch.long or labels.shape!=(final.shape[1]-1,16):raise ValueError('fixed endpoint geometry')
        flat=labels.flatten();ends=torch.where(flat==256)[0]
        if len(ends)!=1:raise ValueError('one loss-side EOS')
        n=int(ends[0])
        if n<1 or not ((flat[:n]>=0)&(flat[:n]<256)).all() or not (flat[n+1:]==257).all():raise ValueError('bytes then EOS then ignored PAD')
        charge('readout');logits=self.codec.verbalize(DifferentiableCodecCommit.from_semantic(final,m.memory_revisions,m.model_epoch)).float()
        if logits.shape!=(*labels.shape,258) or not torch.isfinite(logits).all():raise FloatingPointError('finite frozenD')
        z=logits.reshape(-1,258);byte=F.cross_entropy(z[:n],flat[:n]);eos=F.cross_entropy(z[n:n+1],flat[n:n+1])
        error=(final-target.detach()).square().mean(-1);body=(error*mask).sum()/mask.sum();tail=(error*(~mask)).sum()/(~mask).sum().clamp_min(1)
        # Small standard noisy-target VP loss retained, separately labelled.
        # It is NOT the condition of the main runtime trajectory above.
        charge('denoise');aux,_=self.thought.training_objective(m,target,mask,t,noise)
        c=self.config;latent=c.body_weight*body+c.tail_weight*tail+c.VP_aux_weight*aux
        loss=latent+c.byte_weight*byte+c.EOS_weight*eos
        if not torch.isfinite(loss):raise FloatingPointError('finite actual-trajectory loss')
        self.update.assert_fast()
        return loss,dict(latent=float(latent.detach()),byte_CE=float(byte.detach()),EOS_CE=float(eos.detach()),
            runtime_body_MSE=float(body.detach()),runtime_tail_MSE=float(tail.detach()),aux_VP=float(aux.detach()),
            active_bytes=n,EOS_count=1,PAD_ignored=int((flat==257).sum()),no_source_encoder_call=True,codec_weights_trainable=False)


class RuntimeAlignedEngine(GridThoughtTrainEngine):
    def __init__(self,*args,config=RuntimeLossConfig(),**kwargs):
        self.runtime_loss_config=config
        metadata=dict(kwargs.get('source_metadata') or {});metadata['runtime_objective_v107']=asdict(config);kwargs['source_metadata']=metadata
        super().__init__(*args,**kwargs)
        self.objective=RuntimeAlignedObjective(self.thought,self.codec,self.update,self.codec_SHA,config)

    def finish(self,**kwargs):
        values=list(self.runtime['pending_losses']);h=super().finish(**kwargs)
        h['runtime_loss']={k:sum(x[k] for x in values)/len(values) for k in ('runtime_body_MSE','runtime_tail_MSE','aux_VP')}
        return h

    def payload(self,*,audit=True):
        p=super().payload(audit=audit);p.update(schema='runtime_aligned_fullstate_v107',runtime_loss_config=asdict(self.runtime_loss_config));return p

    def restore(self,p,*,audit=True):
        if p['schema']!='runtime_aligned_fullstate_v107' or p['runtime_loss_config']!=asdict(self.runtime_loss_config):raise ValueError('exact runtime-objective state')
        super().restore(dict(p,schema='grid_tail_thought_fullstate_v104'),audit=audit)
