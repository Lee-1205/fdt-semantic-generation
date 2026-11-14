"""CPU-testable two-phase optimization/state contract. No process launching."""
from __future__ import annotations
import copy,math,random
import numpy as np
import torch
from contextlib import nullcontext
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts,pack_answers
from research.text_span_data_v4 import local_pairs
import text_generation_support_v1 as support

PHASE_LIMIT=512

class PhaseEngine:
 def __init__(self,model,frozen,features,targets,pairs,cache,orders,budget,plan_sha,device='cuda'):
  self.model=model.to(device);self.frozen=frozen;self.features=features;self.targets=targets
  self.pairs=pairs;self.cache=cache;self.orders=orders;self.budget=budget;self.plan_sha=plan_sha;self.device=device
  self.phase='reader';self.attempts_by_phase={'reader':0,'realizer':0};self.updates_by_phase={'reader':0,'realizer':0}
  self.cursors={'reader':0,'realizer':0};self.history=[];self.evaluations={};self.skips=0
  self.exposure={'reader_rows':0,'realizer_rows':0,'transition_reader_rows':0,
   'teacher_rows':0,'matched_prediction_rows':0,'fallback_teacher_rows':0,'realizer_target_utf8_bytes':0}
  self._configure_phase();self.optimizers={'reader':self._make_optimizer('reader')}
  self.scaler=torch.amp.GradScaler('cuda' if str(device).startswith('cuda') else 'cpu',init_scale=1024.)
  self.frozen_sha=support.state_sha(frozen)
  self.factor_sha=support.state_sha({k:v for k,v in model.reader.state_dict().items() if k=='types' or k.startswith('values.')})
  self.reader_final_sha=None

 @property
 def attempts(self):return sum(self.attempts_by_phase.values())
 @property
 def updates(self):return sum(self.updates_by_phase.values())
 @property
 def cursor(self):return sum(self.cursors.values())

 def reader_names(self):
  return [name for name,_ in self.model.reader.named_parameters() if name!='types' and not name.startswith('values.')]

 def _make_optimizer(self,phase):
  if phase=='reader':
   params=dict(self.model.reader.named_parameters());selected=[params[n] for n in self.reader_names()]
  else:selected=list(self.model.decoder.parameters())
  return torch.optim.AdamW(selected,lr=3e-4,weight_decay=.01)

 def _configure_phase(self):
  self.model.zero_grad(set_to_none=True);self.model.requires_grad_(False);self.model.train()
  if self.phase=='reader':
   for name,param in self.model.reader.named_parameters():param.requires_grad_(name in set(self.reader_names()))
   self.model.decoder.eval()
  else:self.model.decoder.requires_grad_(True);self.model.reader.eval()

 def next_indices(self):
  attempt=self.attempts_by_phase[self.phase];cursor=self.cursors[self.phase]
  if attempt>=PHASE_LIMIT:raise StopIteration('phase attempt cap reached')
  indices=self.orders[self.phase][cursor:cursor+8]
  if len(indices)!=8:raise StopIteration('finite phase order exhausted')
  return indices

 def finish_reader(self):
  if self.phase!='reader' or self.attempts_by_phase['reader']!=PHASE_LIMIT:raise ValueError('phase boundary before512')
  self.reader_final_sha=support.state_sha(self.model.reader.state_dict())
  self.phase='realizer';self._configure_phase();self.optimizers['realizer']=self._make_optimizer('realizer')
  # The SAME enabled scaler and all RNG states survive this boundary.

 def batch(self,indices,need_features=True):
  rows=[self.features['train'][i] for i in indices];labels=[self.targets['train'][i] for i in indices]
  gold=torch.tensor([r['head_labels'] for r in labels],device=self.device)
  spans=torch.tensor([r['span_starts'] for r in labels],device=self.device)
  latent=self.cache['train']['latent'][indices].to(self.device)
  if not str(self.device).startswith('cuda'):latent=latent.float()
  if not need_features:return None,None,None,latent,gold,spans,labels
  prompt=self.cache['train']['prompt'][indices].to(self.device)
  if not str(self.device).startswith('cuda'):prompt=prompt.float()
  mask=self.cache['train']['mask'][indices].to(self.device)
  tokens=pack_prompts([r['prompt'] for r in rows],160,self.device)
  return prompt,mask,tokens,latent,gold,spans,labels

 def step(self):
  try:return self._step()
  except BaseException as exc:
   if len(self.history)<self.attempts:
    cursor=self.cursors[self.phase]
    self.history.append({'global_attempt':self.attempts,'phase':self.phase,
     'phase_attempt':self.attempts_by_phase[self.phase],'updates':self.updates,
     'rows':self.orders[self.phase][cursor-8:cursor],'outcome':'FAILED_ATTEMPT',
     'error':repr(exc),'actual_calls':dict(self.budget.calls)})
   raise

 def _step(self):
  self.budget.check();indices=self.next_indices();phase=self.phase
  self.attempts_by_phase[phase]+=1;self.cursors[phase]+=8
  attempt=self.attempts_by_phase[phase];optimizer=self.optimizers[phase]
  lr=3e-4*(.1+.9*(1+math.cos(math.pi*(attempt-1)/PHASE_LIMIT))/2)
  for group in optimizer.param_groups:group['lr']=lr
  optimizer.zero_grad(set_to_none=True)
  mixed=phase=='realizer' and attempt>256
  prompt,mask,tokens,latent,gold,spans,labels=self.batch(indices,phase=='reader' or mixed)
  amp=torch.autocast('cuda',dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
  matched=0;aux=paired=0.
  with amp:
   if phase=='reader':
    self.budget.charge('reader');self.exposure['reader_rows']+=8
    read=self.model.reader(prompt,mask,tokens)
    basic=self.model.reader_loss(read,spans,gold)
    consistency=self.model.pair_consistency_loss(read,local_pairs(indices,self.pairs['train']))
    loss=basic+.1*consistency;aux=float(basic.detach());paired=float(consistency.detach())
   else:
    if mixed:
     self.budget.charge('reader')
     with torch.no_grad():read=self.model.reader(prompt,mask,tokens)
     context,match=self.model.consistent_context_for_training(read,gold,latent)
     matched=int(match.sum());self.exposure['transition_reader_rows']+=8
     self.exposure['matched_prediction_rows']+=matched;self.exposure['fallback_teacher_rows']+=8-matched
    else:context=self.model.teacher_context_for_training(gold,latent)
    ids=pack_answers([r['answer'] for r in labels],40,self.device)
    self.budget.charge('decoder')
    self.exposure['teacher_rows']+=8-matched
    self.exposure['realizer_rows']+=8
    self.exposure['realizer_target_utf8_bytes']+=sum(len(r['answer'].encode()) for r in labels)
    loss=self.model.decoder.loss(ids,context)
  if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('V4 loss nonfinite/explosion')
  self.scaler.scale(loss).backward();self.scaler.unscale_(optimizer)
  for name,param in self.model.named_parameters():
   if not param.requires_grad and param.grad is not None:raise RuntimeError('frozen gradient ownership '+name)
  norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
  previous=self.scaler.get_scale();self.scaler.step(optimizer);self.scaler.update()
  skipped=self.scaler.get_scale()<previous;self.skips+=int(skipped);self.updates_by_phase[phase]+=int(not skipped)
  record={'global_attempt':self.attempts,'phase':phase,'phase_attempt':attempt,'updates':self.updates,
   'loss':float(loss.detach()),'reader_aux':aux,'pair_mse':paired,'grad_norm':float(norm),'AMP_skip':skipped,
   'lr':lr,'rows':indices,'teacher_condition_rows':8-matched if phase=='realizer' else 0,'predicted_condition_rows':matched,
   'outcome':'AMP_SKIP' if skipped else 'UPDATED'}
  self.history.append(record)
  if self.skips>=3 or (not skipped and not math.isfinite(float(norm))):raise FloatingPointError('V4 AMP/gradient stop')
  return record

 def verify_frozen(self):
  if support.state_sha(self.frozen)!=self.frozen_sha:raise RuntimeError('VP source mutated')
  factor=support.state_sha({k:v for k,v in self.model.reader.state_dict().items() if k=='types' or k.startswith('values.')})
  if factor!=self.factor_sha:raise RuntimeError('semantic value/type embeddings changed')
  if self.phase=='realizer' and support.state_sha(self.model.reader.state_dict())!=self.reader_final_sha:raise RuntimeError('reader changed during realizer phase')

 def export(self):
  return {'schema':'fdt_text_span_alignment_v4_fullstate','plan_sha256':self.plan_sha,
   'model':{'generator':self.frozen,'span_generator':{k:v.detach().cpu() for k,v in self.model.state_dict().items()}},
   'optimizers':{k:v.state_dict() for k,v in self.optimizers.items()},'optimizer_parameter_names':{
    'reader':self.reader_names(),'realizer':[n for n,_ in self.model.decoder.named_parameters()]},
   'scaler':self.scaler.state_dict(),'phase':self.phase,'attempts':self.attempts,'updates':self.updates,'cursor':self.cursor,
   'attempts_by_phase':dict(self.attempts_by_phase),'updates_by_phase':dict(self.updates_by_phase),'cursors':dict(self.cursors),
   'orders':self.orders,'history':self.history,'exposure':dict(self.exposure),'actual_calls':dict(self.budget.calls),
   'evaluations':self.evaluations,'skips':self.skips,'frozen_sha':self.frozen_sha,'factor_sha':self.factor_sha,
   'reader_final_sha':self.reader_final_sha,'scheduler':{'horizon_each_phase':PHASE_LIMIT,'next_phase_attempt':self.attempts_by_phase[self.phase]},
   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else [],
   'python_rng':random.getstate(),'numpy_rng':numpy_state()}

 def restore(self,payload):
  if payload['schema']!='fdt_text_span_alignment_v4_fullstate' or payload['plan_sha256']!=self.plan_sha:raise ValueError('exact new contract required')
  if payload['phase'] not in ('reader','realizer') or set(payload['orders'])!={'reader','realizer'}:raise ValueError('phase schema')
  if payload['orders']!=self.orders:raise ValueError('pinned finite orders changed')
  if len(payload['history'])!=payload['attempts'] or [r['global_attempt'] for r in payload['history']]!=list(range(1,payload['attempts']+1)):raise ValueError('complete attempted-update history required')
  if not payload['scaler'] or payload['scaler'].get('scale',0)<=0:raise ValueError('enabled scaler state required')
  if payload['phase']=='realizer' and (payload['attempts_by_phase']['reader']!=PHASE_LIMIT or 'realizer' not in payload['optimizers']):raise ValueError('phase2 missing preserved boundary state')
  if payload['optimizer_parameter_names']['reader']!=self.reader_names():raise ValueError('reader optimizer ownership changed')
  if payload['optimizer_parameter_names']['realizer']!=[n for n,_ in self.model.decoder.named_parameters()]:raise ValueError('decoder optimizer ownership changed')
  self.model.load_state_dict(payload['model']['span_generator'],strict=True);self.frozen=payload['model']['generator']
  self.phase=payload['phase'];self._configure_phase()
  self.optimizers={k:self._make_optimizer(k) for k in payload['optimizers']}
  for key,opt in self.optimizers.items():opt.load_state_dict(payload['optimizers'][key])
  self.scaler.load_state_dict(payload['scaler'])
  for field in ('attempts_by_phase','updates_by_phase','cursors','orders','history','exposure','evaluations','skips','frozen_sha','factor_sha','reader_final_sha'):
   setattr(self,field,copy.deepcopy(payload[field]))
  self.budget.calls=dict(payload['actual_calls'])
  if (self.attempts,self.updates,self.cursor)!=(payload['attempts'],payload['updates'],payload['cursor']):raise ValueError('counter sums')
  for phase in ('reader','realizer'):
   if not 0<=self.attempts_by_phase[phase]<=PHASE_LIMIT or self.cursors[phase]!=8*self.attempts_by_phase[phase]:raise ValueError('phase cursor/exposure contract')
  torch.set_rng_state(payload['torch_rng'])
  if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(payload['cuda_rng'])
  random.setstate(payload['python_rng']);n=payload['numpy_rng']
  np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
  self.verify_frozen()
