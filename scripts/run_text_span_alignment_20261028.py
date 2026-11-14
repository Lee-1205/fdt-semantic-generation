"""One NEW bounded learned-pointer / faithful realizer study, root only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,random,sys,time,traceback
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.text_diffusion_v1 import pack_prompts,pack_answers,decode_bytes,PAD
from research.text_span_reader_v4 import SpanSemanticGenerator
from research.text_span_data_v4 import sha,finite_pair_orders
from text_span_engine_v4 import PhaseEngine,PHASE_LIMIT
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_text_span_alignment_v4'
LIMITS={'attempts':1024,'batch':8,'wall':900,'reserve':90,'max_calls':2800,
        'max_output_bytes':8*1024**3,'start_free_bytes':36*1024**3,'min_gpu_free_mib':12288}
PARENT_SHA='a5e1c0e774c89a22dde794d2cffd138926ae961fd8cde592bf466d25d63a71b3'
CACHE_SHA='c0fe127015ddd17319e46d352be526264ffec0693990679585916051bca3fce9'
MANIFEST_SHA='dff9d3f73a6d8039d3b6bdfeca67e29afff7c9155a735d4534ceed6b9e379cce'

def validate(plan_path,output,worker=None):
 p=io.load_json(plan_path)
 if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p['limits']!=LIMITS:raise ValueError('V4 schema/caps')
 if Path(p['output']).resolve()!=Path(output).resolve() or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('output/core')
 if p['parent_sha']!=PARENT_SHA or p['cache_sha']!=CACHE_SHA or p['manifest_sha']!=MANIFEST_SHA:raise ValueError('wrong V4 source')
 if p.get('phase_limits')!={'reader':512,'realizer':512} or p.get('transition_boundary')!=256:raise ValueError('fixed phase allocation')
 required={str(Path(p['parent']).resolve()):PARENT_SHA,str(Path(p['cache']).resolve()):CACHE_SHA,
  str((Path(p['artifact'])/'manifest.json').resolve()):MANIFEST_SHA}
 if any(p['files'].get(k)!=v for k,v in required.items()):raise ValueError('required parent/cache/manifest binding missing')
 for file,h in p['files'].items():
  if sha(file)!=h:raise ValueError('V4 pinned file changed '+file)
 if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('worker unpinned')
 if p.get('E2_final_accessed') is not False or p.get('evaluation_teacher_allowed') is not False:raise ValueError('evaluation/holdout contract')
 if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=900 or p['cutoff']!=p['deadline']-90):raise ValueError('fixed allocation')
 a=Path(p['artifact']);features=io.load_json(a/'features.json');targets=io.load_json(a/'targets.json');pairs=io.load_json(a/'pairs.json')
 manifest=io.load_json(a/'manifest.json')
 for filename,value in manifest['files'].items():
  path=(a/filename).resolve()
  if p['files'].get(str(path))!=value or sha(path)!=value:raise ValueError('artifact file identity '+filename)
 orders={phase:io.load_json(a/(phase+'_order.json')) for phase in ('reader','realizer')}
 if set(features)!=set(targets) or {k:len(v) for k,v in features.items()}!={'train':1024,'train_probe':16,'dev_pairs':32,'dev_wording':32}:raise ValueError('fixed panels/pool')
 for panel,rows in features.items():
  if len(targets[panel])!=len(rows):raise ValueError('target length mismatch')
  tokens=pack_prompts([r['prompt'] for r in rows])
  if any(set(r)!={'id','prompt'} for r in rows):raise ValueError('gold in feature file')
  for i,(r,y) in enumerate(zip(rows,targets[panel])):
   if support.hashlib.sha256(r['prompt'].encode()).hexdigest()!=r['id']:raise ValueError('feature row identity')
   if len(y['head_labels'])!=4 or len(y['span_starts'])!=2:raise ValueError('target schema')
   for j,pos in enumerate(y['span_starts']):
    if int(tokens[i,pos])!=65+y['head_labels'][j]:raise ValueError('byte span label')
  pack_answers([y['answer'] for y in targets[panel]])
 for phase,seed in (('reader',31),('realizer',32)):
  if orders[phase]!=finite_pair_orders(pairs['train'],1024,4,seed):raise ValueError('finite pair order changed')
 return p,features,targets,pairs,orders

class Budget(support.Budget):
 def __init__(self,p):super().__init__(p);self.calls={'reader':0,'decoder':0,'encode':0,'denoise':0}

def batch_features(features,cache,indices,device):
 prompt=cache['prompt'][indices].to(device);latent=cache['latent'][indices].to(device)
 if not str(device).startswith('cuda'):prompt=prompt.float();latent=latent.float()
 tokens=pack_prompts([features[i]['prompt'] for i in indices],160,device)
 return prompt,cache['mask'][indices].to(device),tokens,latent

def evaluate(engine,features,targets,cache,budget,zero=False,heads_only=False):
 model=engine.model;model.eval();result={}
 amp=torch.autocast('cuda',dtype=torch.float16) if str(engine.device).startswith('cuda') else nullcontext()
 with torch.no_grad(),amp:
  for name in ('train_probe','dev_pairs','dev_wording'):
   rows=features[name];ys=targets[name];raw=[];exact=good=bytecorrect=bytetotal=0;heads=[0]*4;spans=[0]*2;positive=positive_exact=0
   for start in range(0,len(rows),8):
    indices=list(range(start,min(start+8,len(rows))))
    prompt,mask,tokens,latent=batch_features(rows,cache[name],indices,engine.device)
    budget.charge('reader');context,read=model.context(prompt,mask,tokens,latent,zero)
    generated=None if heads_only else model.decoder.generate(context,budget.charge)
    pred=torch.cat((read['entity_probabilities'].argmax(-1),torch.stack([v.argmax(-1) for v in read['semantic_logits']],-1)),-1)
    pointer=read['pointer_logits'].argmax(-1)
    # Gold is first used AFTER the model's prediction/generation above.
    gold=torch.tensor([ys[i]['head_labels'] for i in indices],device=engine.device)
    goldspans=torch.tensor([ys[i]['span_starts'] for i in indices],device=engine.device)
    for j in range(4):heads[j]+=int((pred[:,j]==gold[:,j]).sum())
    for j in range(2):spans[j]+=int((pointer[:,j]==goldspans[:,j]).sum())
    if not heads_only:
     answer=pack_answers([ys[i]['answer'] for i in indices],40,engine.device);m=answer!=PAD
     bytecorrect+=int(((generated==answer)&m).sum());bytetotal+=int(m.sum())
    for j,index in enumerate(indices):
     y=ys[index];record={'id':rows[index]['id'],'prompt':rows[index]['prompt'],'answer':y['answer'],
      'language':y['language'],'class':y['class'],'reader_pred':pred[j].cpu().tolist(),'reader_gold':gold[j].cpu().tolist(),
      'span_pred':pointer[j].cpu().tolist(),'span_gold':goldspans[j].cpu().tolist(),
      'entity_probabilities':read['entity_probabilities'][j].float().cpu().tolist(),
      'semantic_logits':[v[j].float().cpu().tolist() for v in read['semantic_logits']]}
     if not heads_only:
      decoded=decode_bytes(generated[j].cpu().tolist());hit=decoded['well_formed'] and decoded['text']==y['answer']
      exact+=int(hit);good+=int(decoded['well_formed']);positive+=int(y['class']!='unknown');positive_exact+=int(hit and y['class']!='unknown')
      record.update(generated=decoded,exact_match=hit)
     raw.append(record)
   result[name]={'rows':len(rows),'head_accuracy':[n/len(rows) for n in heads],'span_accuracy':[n/len(rows) for n in spans],
    'zero_VP_latent':zero,'heads_only':heads_only,'raw':raw}
   if not heads_only:result[name].update(exact=exact,exact_rate=exact/len(rows),well_formed=good)
   if not heads_only:
    result[name]['byte_accuracy_including_eos']=bytecorrect/bytetotal
    result[name]['positive_rows']=positive;result[name]['positive_exact']=positive_exact
 return result

def load_source(plan,features):
 parent=torch.load(plan['parent'],map_location='cpu',weights_only=True)
 if parent['selected_train_indices']!=io.load_json(Path(plan['artifact'])/'selected_train_indices.json'):raise ValueError('inherited pool ordering')
 model=SpanSemanticGenerator(Config(**parent['core']));state=parent['model']['semantic_generator']
 mapping=model.reader.warmstart_v3(state)
 model.decoder.load_state_dict({k[len('decoder.'):]:v for k,v in state.items() if k.startswith('decoder.')},strict=True)
 cache=torch.load(plan['cache'],map_location='cpu',weights_only=True)
 for panel,rows in features.items():
  tokens=pack_prompts([r['prompt'] for r in rows]);expected=torch.cat((tokens!=256,torch.ones(len(rows),1,dtype=torch.bool)),1)
  if set(cache[panel])!={'prompt','mask','latent'} or cache[panel]['prompt'].shape!=(len(rows),161,512) or cache[panel]['latent'].shape!=(len(rows),40,512):raise ValueError('cache schema/capacity')
  if not torch.equal(cache[panel]['mask'],expected) or not torch.isfinite(cache[panel]['prompt']).all() or not torch.isfinite(cache[panel]['latent']).all():raise ValueError('cache/raw input contract')
 return model,parent['model']['generator'],cache,parent['core'],mapping

def run(plan_path,output):
 plan,features,targets,pairs,orders=validate(plan_path,output,__file__)
 if not plan['authorized'] or plan['deadline'] is None:raise PermissionError('unsealed V4')
 if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
 configure_determinism();torch.set_num_threads(4);torch.manual_seed(31);torch.cuda.manual_seed_all(31);random.seed(31);np.random.seed(31)
 budget=Budget(plan);budget.check();engine=None;latest=None;phase='initializing';started=time.time();guard=False
 def progress():
  io.atomic_json(output/'progress.json',{'phase':phase,'training_phase':engine.phase if engine else None,
   'attempts':engine.attempts if engine else 0,'updates':engine.updates if engine else 0,
   'attempts_by_phase':engine.attempts_by_phase if engine else None,'actual_calls':budget.calls,
   'native_pid':os.getpid(),'parent_pid':os.getppid(),'time_unix':time.time(),'elapsed':time.time()-started,'latest':latest})
 def commit(emergency=False):
  nonlocal latest
  frozen_error=None
  if not emergency:budget.check();engine.verify_frozen()
  else:
   if time.time()>=plan['deadline']:raise TimeoutError('owner end reached; emergency save unconfirmed')
   try:engine.verify_frozen()
   except BaseException as exc:frozen_error=repr(exc)
  relative=engine.phase+('_emergency' if emergency else '')
  directory=output/relative;directory.mkdir(exist_ok=True)
  payload=engine.export();payload.update(core=core,parent_sha256=plan['parent_sha'],cache_sha256=plan['cache_sha'],artifact_manifest_sha256=plan['manifest_sha'])
  payload.update(emergency_snapshot=emergency,frozen_verification_error=frozen_error)
  latest={**support.save_full(directory,payload),'phase':engine.phase,'relative_directory':relative}
  io.atomic_json(output/'latest.json',latest)
 progress()
 try:
  if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
  model,frozen,cache,core,mapping=load_source(plan,features)
  engine=PhaseEngine(model,frozen,features,targets,pairs,cache,orders,budget,sha(plan_path),'cuda')
  io.atomic_json(output/'specification.json',{'frozen_VP_parameters':91633920,'reader_decoder_parameters':sum(p.numel() for p in model.parameters()),
   'warmstart_mapping':mapping,'phase_limits':{'reader':512,'realizer':512},'raw_features':'byte IDs + fixed prompt context/mask/VP latent',
   'scaler':'ONE enabled scale1024 state kept across phases','weight_and_global_rng_seed':31,
   'order_seeds':{'reader':31,'realizer':32},'phase_boundary_rng_reset':False,
   'evaluation_teacher_allowed':False,'VP_calls':0})
  commit();phase='baseline';progress();baseline=evaluate(engine,features,targets,cache,budget)
  io.atomic_json(output/'baseline_generation.json',baseline);engine.evaluations['baseline']={'path':'baseline_generation.json','sha256':sha(output/'baseline_generation.json')}
  phase='reader_training';engine._configure_phase();progress()
  while engine.attempts_by_phase['reader']<512:
   if time.time()+90>=plan['cutoff']:raise TimeoutError('V4 final reserve')
   engine.step()
   if engine.attempts_by_phase['reader']%256==0:commit()
   progress()
  phase='reader_headcheck';progress();headcheck=evaluate(engine,features,targets,cache,budget,heads_only=True)
  io.atomic_json(output/'reader_phase_headcheck.json',headcheck);engine.evaluations['reader_headcheck']={'path':'reader_phase_headcheck.json','sha256':sha(output/'reader_phase_headcheck.json')}
  engine.finish_reader();commit();phase='realizer_training';progress()
  while engine.attempts_by_phase['realizer']<512:
   if time.time()+90>=plan['cutoff']:raise TimeoutError('V4 final reserve')
   engine.step()
   if engine.attempts_by_phase['realizer']%256==0:commit()
   progress()
  phase='final_generation';progress();final=evaluate(engine,features,targets,cache,budget)
  io.atomic_json(output/'final_generation.json',final)
  zero=evaluate(engine,features,targets,cache,budget,zero=True);io.atomic_json(output/'zero_latent_generation.json',zero)
  engine.verify_frozen()
  if sha(plan['parent'])!=plan['parent_sha'] or sha(plan['cache'])!=plan['cache_sha'] or support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('frozen source/core/cache changed')
  summary=lambda d:{k:{a:b for a,b in v.items() if a!='raw'} for k,v in d.items()}
  result={'status':'TEXT_SPAN_ALIGNMENT_V4_COMPLETE','attempts':engine.attempts,'updates':engine.updates,
   'attempts_by_phase':engine.attempts_by_phase,'updates_by_phase':engine.updates_by_phase,'AMP_skips':engine.skips,
   'actual_calls':budget.calls,'exposure':engine.exposure,'baseline':summary(baseline),'reader_headcheck':summary(headcheck),
   'final':summary(final),'zero_latent':summary(zero),'frozen_sources_unchanged':True,'factor_embeddings_unchanged':True,
   'reader_frozen_during_realization':True,'E2_final_accessed':False,'all_evaluation_rows_predicted_only':True,
   'latest':latest,'seconds':time.time()-started,'interpretation':'Observed finite A-H synthetic span reading and learned realizer; not broad NER/generalLM or diffusion-only causal proof.'}
  io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
 except BaseException as exc:
  phase='stopped_or_error'
  if engine is not None:
   try:commit(emergency=True)
   except BaseException as save_exc:io.atomic_json(output/'save_error.json',{'error':repr(save_exc)})
  io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'calls':budget.calls});progress();return 2
 finally:
  if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 if a.preflight:
  torch.set_num_threads(2);plan,features,_,_,_=validate(a.plan,a.output,__file__);load_source(plan,features)
  print('CPU_PREFLIGHT_PASS');return 0
 return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
