"""Meaning-only identity VP training; language is completely outside train graph.
Use hidden520 for latent512: even after LayerNorm centering the output map can
span512 directions. This removes the proven256->512 affine velocity restriction.
"""
from dataclasses import dataclass,asdict
import torch
from research.runtime_aligned_thought_v107 import training_rollout
from research.thought_grid_tail_v104 import GridThoughtTrainEngine

@dataclass(frozen=True)
class LatentPolicy:
    steps:int=16
    body_weight:float=1.
    tail_weight:float=.05
    VP_aux_weight:float=.1

def effective_velocity_rank(thought):
    w=thought.velocity.weight.detach().double();gamma=thought.final_norm.weight.detach().double()
    wg=w*gamma[None];effective=wg-wg.sum(1,keepdim=True)/wg.shape[1]
    return int(torch.linalg.matrix_rank(effective))

class LatentIdentityObjective:
    def __init__(self,thought,update,codec_SHA,policy):
        self.thought,self.update,self.codec_SHA,self.policy=thought,update,codec_SHA,policy
    def __call__(self,m,packet,t,noise,charge=lambda kind:None):
        if not self.thought.training:raise RuntimeError('TRAIN-only clean meaning')
        self.update.assert_fast()
        required={'schema','answer_SHA','source_codec_SHA','semantic_target','valid_extent_mask','labels','model_epoch','target_condition_features_never_used'}
        if set(packet)!=required or packet['schema']!='LOSS_ONLY_answer_meaning_v95' or packet['source_codec_SHA']!=self.codec_SHA or packet['model_epoch']!=m.model_epoch or not packet['target_condition_features_never_used']:raise ValueError('separate frozen meaning loss packet')
        if any(packet[k].requires_grad for k in ('semantic_target','valid_extent_mask','labels')):raise ValueError('frozen loss-only meaning')
        final=training_rollout(self.thought,m,noise,self.policy.steps,charge)
        target=packet['semantic_target'].to(final.device);mask=packet['valid_extent_mask'].to(final.device)
        if final.shape!=target.shape or mask.shape!=final.shape[:2] or mask.dtype!=torch.bool or not mask.any():raise ValueError('fixed-capacity meaning loss')
        error=(final-target).square().mean(-1);body=(error*mask).sum()/mask.sum();tail=(error*(~mask)).sum()/(~mask).sum().clamp_min(1)
        charge('denoise');aux,_=self.thought.training_objective(m,target,mask,t,noise)
        c=self.policy;loss=c.body_weight*body+c.tail_weight*tail+c.VP_aux_weight*aux
        if not torch.isfinite(loss):raise FloatingPointError('finite latent-only loss')
        self.update.assert_fast()
        return loss,dict(latent=float(loss.detach()),byte_CE=0.,EOS_CE=0.,language_losses_NOT_used=True,
            runtime_body_MSE=float(body.detach()),runtime_tail_MSE=float(tail.detach()),aux_VP=float(aux.detach()),
            active_bytes=int((packet['labels']<256).sum()),EOS_count=int((packet['labels']==256).sum()),
            PAD_ignored=int((packet['labels']==257).sum()),source_encoder_calls0=True,language_forward_calls0=True)

class LatentIdentityEngine(GridThoughtTrainEngine):
    def __init__(self,*args,policy=LatentPolicy(),**kwargs):
        if type(policy.steps)is not int or policy.steps<1 or min(policy.body_weight,policy.tail_weight,policy.VP_aux_weight)<=0:raise ValueError('fixed latent policy')
        self.latent_policy=policy
        metadata=dict(kwargs.get('source_metadata') or {});metadata['LATENT_ONLY_v110']=asdict(policy);metadata['D_during_training']=False;kwargs['source_metadata']=metadata
        super().__init__(*args,**kwargs)
        if policy.steps>self.thought.c.max_steps:raise ValueError('bounded meaningful trajectory')
        self.objective=LatentIdentityObjective(self.thought,self.update,self.codec_SHA,policy)
    def finish(self,**kwargs):
        values=list(self.runtime['pending_losses']);h=super().finish(**kwargs)
        h['runtime_loss']={k:sum(x[k] for x in values)/len(values) for k in ('runtime_body_MSE','runtime_tail_MSE','aux_VP')};h['D_during_training']=False
        return h
    def payload(self,*,audit=True):
        p=super().payload(audit=audit);p.update(schema='latent_identity_fullstate_v110',latent_policy=asdict(self.latent_policy),language_losses_used=False);return p
    def restore(self,p,*,audit=True):
        if p['schema']!='latent_identity_fullstate_v110' or p['latent_policy']!=asdict(self.latent_policy) or p['language_losses_used']:raise ValueError('exact meaning-only objective restore')
        super().restore(dict(p,schema='grid_tail_thought_fullstate_v104'),audit=audit)
