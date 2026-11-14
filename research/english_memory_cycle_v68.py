"""Training-only meaning alignment INSIDE the live fuzzy language cycle.

No new inference path or parameters. Contrast reads actual global Commit values,
not factor logits or raw H. Random local-slot masking makes a second language
loss use the retained neural global memory; no gold role/span selects the mask.
"""
import torch
from torch.nn import functional as F
from research.three_memory_codec_v33 import DifferentiableCodecCommit

OBJECTIVE='live_reconstruction+.1_random_local_mask_reconstruction+.05_actual_global_Commit_SupCon'

def global_memory_contrast(commits,meaning_ids,temperature=.2):
    if len(commits)!=len(meaning_ids) or len(commits)<4:raise ValueError('grouped actual memory states')
    vectors=[]
    for commit in commits:
        if type(commit) is not DifferentiableCodecCommit:raise TypeError('differentiable Commit values required')
        z=commit.reader_tensor()
        if len(z)!=1 or z.shape[1]<=2:raise ValueError('one encoded sentence per Commit')
        vectors.append(z[:,:2].float().flatten(1))
    z=F.normalize(torch.cat(vectors),dim=-1,eps=1e-8)
    labels=torch.as_tensor(meaning_ids,device=z.device)
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive=(labels[:,None]==labels[None,:])&~eye
    if not positive.any(-1).all():raise ValueError('each row needs an equivalent paraphrase')
    scores=z@z.T/temperature
    logp=scores-scores.masked_fill(eye,-torch.inf).logsumexp(-1,keepdim=True)
    return -(logp.masked_fill(~positive,0).sum(-1)/positive.sum(-1)).mean()

def training_masked_local_logits(model,commit,drop_probability=.25):
    if not model.training or not torch.is_grad_enabled():raise ValueError('local-slot augmentation is training only')
    z=commit.reader_tensor();slots=z.shape[1]-2
    if len(z)!=1 or slots<1 or not 0<drop_probability<1:raise ValueError('input Commit topology')
    dropped=torch.rand(slots,device=z.device)<drop_probability
    if not dropped.any():dropped[torch.randint(slots,(1,),device=z.device)]=True
    local=z[:,2:]*(~dropped)[None,:,None]
    masked=DifferentiableCodecCommit.from_semantic(torch.cat((z[:,:2],local),1),commit.memory_revisions,commit.model_epoch)
    logits=model.verbalize(masked)
    positions=torch.cat((dropped,torch.zeros(1,device=z.device,dtype=torch.bool)))[:,None].expand(-1,model.readout.chunk)
    return logits,positions,dropped

def masked_reconstruction_loss(logits,targets,positions):
    if positions.dtype!=torch.bool or positions.shape!=targets.shape:raise ValueError('random mask layout')
    selected=positions&(targets!=257)
    if not selected.any():raise ValueError('at least one active masked position')
    ce=F.cross_entropy(logits.float().transpose(1,2),targets,ignore_index=257,reduction='none')
    return ce[selected].mean()
