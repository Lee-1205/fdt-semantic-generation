"""CPU candidate: learned question pointers over raw A-H byte candidates.

Finite alphabet filtering is explicit; uppercase words provide distractors.
It does not parse query roles or accept gold spans at inference. Not broad NER.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from fdt_local.model_masked import Core
from research.text_context_decoder_v2 import ContextDecoder

class SpanReader(nn.Module):
 def __init__(self,config,max_prompt_bytes=160):
  super().__init__();d=config.z_dim;self.max_prompt_bytes=max_prompt_bytes
  self.queries=nn.Parameter(torch.randn(4,d)*.02)
  self.blocks=nn.ModuleList([Core(config) for _ in range(2)])
  self.pointer_query=nn.Linear(d,d,bias=False);self.pointer_key=nn.Linear(d,d,bias=False)
  self.byte_positions=nn.Embedding(max_prompt_bytes,d)
  self.semantic_heads=nn.ModuleList([nn.Linear(d,3),nn.Linear(d,2)])
  self.values=nn.ModuleList([nn.Embedding(n,d) for n in (8,8,3,2)])
  self.types=nn.Parameter(torch.randn(4,d)*.02)

 def factor_values(self,probabilities):
  return torch.stack([p.float()@emb.weight.float()+self.types[i]
   for i,(p,emb) in enumerate(zip(probabilities,self.values))],1)

 def forward(self,prompt,mask,tokens):
  n=tokens.shape[1]
  if n>self.max_prompt_bytes or prompt.shape[:2]!=(len(tokens),n+1) or mask.shape!=prompt.shape[:2]:raise ValueError('byte/context/null alignment')
  if tokens.dtype!=torch.long or ((tokens<0)|(tokens>256)).any():raise ValueError('raw byte vocabulary')
  candidates=(tokens>=65)&(tokens<=72)&mask[:,:n].bool()
  if not candidates.any(-1).all():raise ValueError('finite A-H candidate scope')
  x=self.queries[None].expand(len(prompt),-1,-1)
  for block in self.blocks:x=block(x,prompt,torch.ones(x.shape[:2],device=x.device,dtype=torch.bool),mask)
  keys=self.pointer_key(prompt[:,:n]+self.byte_positions(torch.arange(n,device=x.device))[None])
  queries=self.pointer_query(x[:,:2])
  scores=torch.einsum('brd,bnd->brn',queries.float(),keys.float())/math.sqrt(x.shape[-1])
  scores=scores.masked_fill(~candidates[:,None],-torch.inf)
  pointer_prob=scores.softmax(-1)
  ids=(tokens-65).clamp(0,7)
  entity_prob=pointer_prob.new_zeros(len(tokens),2,8).scatter_add(-1,ids[:,None].expand(-1,2,-1),pointer_prob)
  relation=self.semantic_heads[0](x[:,2]).float();language=self.semantic_heads[1](x[:,3]).float()
  probabilities=[entity_prob[:,0],entity_prob[:,1],relation.softmax(-1),language.softmax(-1)]
  factors=self.factor_values(probabilities)
  return {'factors':factors,'pointer_logits':scores,'entity_probabilities':entity_prob,
          'semantic_logits':[relation,language],'candidate_mask':candidates}

 def warmstart_v3(self,state):
  own=self.state_dict();mapped={}
  for name,value in own.items():
   previous='reader.'+name
   if name.startswith('semantic_heads.'):
    parts=name.split('.');previous='reader.heads.'+str(int(parts[1])+2)+'.'+'.'.join(parts[2:])
   if previous in state:
    if state[previous].shape!=value.shape:raise ValueError('mapped parameter shape '+name)
    mapped[name]=state[previous]
  missing,unexpected=self.load_state_dict(mapped,strict=False)
  allowed=('pointer_query.','pointer_key.','byte_positions.')
  if unexpected or any(not k.startswith(allowed) for k in missing):raise ValueError('unexpected warmstart difference')
  return {'imported_keys':len(mapped),'new_keys':missing}

class SpanSemanticGenerator(nn.Module):
 def __init__(self,config,max_prompt_bytes=160):
  super().__init__();self.reader=SpanReader(config,max_prompt_bytes);self.decoder=ContextDecoder(config)

 def context(self,prompt,mask,tokens,latent,zero_latent=False):
  read=self.reader(prompt,mask,tokens)
  context=self.decoder.context(read['factors'],torch.ones(read['factors'].shape[:2],device=prompt.device,dtype=torch.bool),latent,zero_latent)
  return context,read

 def teacher_context_for_training(self,gold,latent):
  if not self.training:raise RuntimeError('teacher factors forbidden in evaluation/inference')
  if gold.ndim!=2 or gold.shape[1]!=4:raise ValueError('teacher schema')
  probabilities=[F.one_hot(gold[:,i],n).float() for i,n in enumerate((8,8,3,2))]
  factors=self.reader.factor_values(probabilities)
  return self.decoder.context(factors,torch.ones(factors.shape[:2],device=factors.device,dtype=torch.bool),latent)

 def reader_loss(self,read,span_starts,head_labels):
  span=F.cross_entropy(read['pointer_logits'].reshape(-1,read['pointer_logits'].shape[-1]),span_starts.reshape(-1))
  relation=F.cross_entropy(read['semantic_logits'][0],head_labels[:,2])
  language=F.cross_entropy(read['semantic_logits'][1],head_labels[:,3])
  return span+relation+.25*language

 def pair_consistency_loss(self,read,pairs):
  if not self.training:raise RuntimeError('paired training loss forbidden in inference')
  if not pairs:return read['entity_probabilities'].sum()*0
  first=torch.tensor([a for a,b in pairs],device=read['factors'].device)
  second=torch.tensor([b for a,b in pairs],device=read['factors'].device)
  entities=read['entity_probabilities']
  e=(entities[first]-entities[second].flip(1)).square().mean()
  relation=read['semantic_logits'][0].float().softmax(-1)
  r=(relation[first]-relation[second][:,[1,0,2]]).square().mean()
  return e+r

 def consistent_context_for_training(self,read,gold,latent):
  if not self.training:raise RuntimeError('teacher/predicted transition is training-only')
  matched=self.matched_prediction_mask(read,gold)
  teacher=self.reader.factor_values([F.one_hot(gold[:,i],n).float() for i,n in enumerate((8,8,3,2))])
  factors=torch.where(matched[:,None,None],read['factors'],teacher)
  context=self.decoder.context(factors,torch.ones(factors.shape[:2],device=factors.device,dtype=torch.bool),latent)
  return context,matched

 def matched_prediction_mask(self,read,head_labels):
  # Training selector only. Does not change inference or silently discard scores.
  if not self.training:raise RuntimeError('matched-condition selector is training-only')
  entity=read['entity_probabilities'].argmax(-1)
  relation=read['semantic_logits'][0].argmax(-1)
  language=read['semantic_logits'][1].argmax(-1)
  predicted=torch.cat((entity,relation[:,None],language[:,None]),-1)
  return (predicted==head_labels).all(-1).detach()
