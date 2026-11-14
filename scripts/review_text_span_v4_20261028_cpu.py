"""Read-only root audit of completed V4 evidence. No neural forward/CUDA use."""
import json,re,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import text_generation_support_v1 as support
from research.text_diffusion_v1 import decode_bytes,pack_answers,PAD

def semantics(text):
 if text in ('It is unknown.','알 수 없습니다.'):return ('unknown',)
 m=re.fullmatch(r'([A-H]) is (larger|smaller) than ([A-H])\.',text)
 if m:
  a,k,b=m.groups();return ('positive',a,b) if k=='larger' else ('positive',b,a)
 m=re.fullmatch(r'([A-H])는 ([A-H])보다 (큽니다|작습니다)\.',text)
 if m:
  a,b,k=m.groups();return ('positive',a,b) if k=='큽니다' else ('positive',b,a)
 return None

def head_semantics(pred):
 a,b=chr(65+pred[0]),chr(65+pred[1]);r=pred[2]
 return ('unknown',) if r==2 else ('positive',a,b) if r==0 else ('positive',b,a)

def run():
 out=ROOT/'runs/text_span_alignment_20261028_v4';plan=json.loads((out/'plan.json').read_text())
 result=json.loads((out/'result.json').read_text());owner=json.loads((out/'supervisor/final_status.json').read_text())
 assert owner['status']=='COMPLETE' and owner['launches']==1 and owner['exit_code']==0 and owner['exit']['all_owned_processes_gone']
 terminal=json.loads((out/'code_snapshot/root_terminal_native.json').read_text(encoding='utf-8-sig'))
 assert not terminal['native']
 for path,h in plan['files'].items():assert support.sha(path)==h
 review={};generation_calls=0;all_final={};raw_total=0
 for filename in ('baseline_generation','reader_phase_headcheck','final_generation','zero_latent_generation'):
  panels=json.loads((out/(filename+'.json')).read_text(encoding='utf-8'));review[filename]={}
  for panel,v in panels.items():
   exact=good=semantic=positive=positive_exact=joint=joint_bad=faithful=bytecorrect=bytetotal=0
   heads=[0]*4;spans=[0]*2;decoder_calls=0;class_counts={}
   for row in v['raw']:
    ep=torch.tensor(row['entity_probabilities']);sem=[torch.tensor(a) for a in row['semantic_logits']]
    assert torch.isfinite(ep).all() and all(torch.isfinite(a).all() for a in sem)
    pred=ep.argmax(-1).tolist()+[int(a.argmax()) for a in sem];assert pred==row['reader_pred']
    gold=row['reader_gold'];correct_head=pred[:3]==gold[:3]
    for i in range(4):heads[i]+=int(pred[i]==gold[i])
    for i in range(2):spans[i]+=int(row['span_pred'][i]==row['span_gold'][i])
    raw=row['prompt'].encode()
    assert [raw[pos]-65 for pos in row['span_gold']]==gold[:2]
    if v['heads_only']:continue
    decoded=decode_bytes(row['generated']['token_ids']);assert decoded==row['generated']
    hit=decoded['well_formed'] and decoded['text']==row['answer'];sm=decoded['well_formed'] and semantics(decoded['text'])==semantics(row['answer'])
    exact+=int(hit);good+=int(decoded['well_formed']);semantic+=int(sm)
    positive+=int(row['class']!='unknown');positive_exact+=int(hit and row['class']!='unknown')
    joint+=int(correct_head);joint_bad+=int(correct_head and not sm)
    faithful+=int(decoded['well_formed'] and semantics(decoded['text'])==head_semantics(pred))
    target=pack_answers([row['answer']])[0];ids=torch.tensor(decoded['token_ids']);mask=target!=PAD
    bytecorrect+=int(((target==ids)&mask).sum());bytetotal+=int(mask.sum())
    key=row['language']+'|'+row['class'];bucket=class_counts.setdefault(key,{'rows':0,'exact':0,'semantic':0})
    bucket['rows']+=1;bucket['exact']+=int(hit);bucket['semantic']+=int(sm)
    raw_total+=1
   assert [n/v['rows'] for n in heads]==v['head_accuracy'] and [n/v['rows'] for n in spans]==v['span_accuracy']
   score={'rows':v['rows'],'head_accuracy':v['head_accuracy'],'span_accuracy_from_saved_predictions':v['span_accuracy']}
   if not v['heads_only']:
    for start in range(0,len(v['raw']),8):
     lengths=[]
     for r in v['raw'][start:start+8]:
      ids=r['generated']['token_ids'];lengths.append(ids.index(256)+1 if 256 in ids else 40)
     decoder_calls+=max(lengths)
    assert exact==v['exact'] and good==v['well_formed'] and positive_exact==v['positive_exact']
    assert abs(bytecorrect/bytetotal-v['byte_accuracy_including_eos'])<1e-12
    score.update(exact=exact,semantic_equivalent_descriptive=semantic,well_formed=good,
     positive_rows=positive,positive_exact=positive_exact,correct_three_heads=joint,
     generation_semantic_wrong_given_correct_heads=joint_bad,generation_agrees_with_predicted_semantics=faithful,
     decoder_calls_reconstructed=decoder_calls,unknown_constant_baseline=v['rows']-positive,class_language=class_counts)
    generation_calls+=decoder_calls
   review[filename][panel]=score
   if filename=='final_generation':all_final[panel]=v['raw']
 assert raw_total==240
 assert result['actual_calls']=={'reader':512+256+40,'decoder':512+generation_calls,'encode':0,'denoise':0}
 checkpoint_review=[];reference=None;reader_end=None;reader_opt_end=None;boundary=None;final_payload=None
 paths=[out/'reader/checkpoints'/f'state_{n:04d}.json' for n in (0,256,512)]
 paths += [out/'realizer/checkpoints'/f'state_{n:04d}.json' for n in (512,768,1024)]
 for path in paths:
  meta=json.loads(path.read_text());pt=path.with_suffix('.pt');assert support.sha(pt)==meta['sha256'] and pt.stat().st_size==meta['bytes']
  p=torch.load(pt,map_location='cpu',weights_only=True);assert support.state_sha(p)==meta['algorithm_sha256']
  assert p['plan_sha256']==support.sha(out/'plan.json') and p['attempts']==sum(p['attempts_by_phase'].values())
  assert p['cursor']==8*p['attempts'] and len(p['history'])==p['attempts'] and p['updates']==p['attempts']
  assert p['scaler']['scale']==1024 and p['scaler']['_growth_tracker']==p['attempts']
  assert len(p['cuda_rng'])==1 and p['torch_rng'].dtype==torch.uint8 and p['python_rng'] and p['numpy_rng']['keys']
  state=p['model']['span_generator']
  hashes={'VP':support.state_sha(p['model']['generator']),
   'factors':support.state_sha({k:v for k,v in state.items() if k=='reader.types' or k.startswith('reader.values.')}),
   'reader':support.state_sha({k:v for k,v in state.items() if k.startswith('reader.')}),
   'decoder':support.state_sha({k:v for k,v in state.items() if k.startswith('decoder.')})}
  if reference is None:reference=hashes
  assert hashes['VP']==reference['VP'] and hashes['factors']==reference['factors']
  if p['phase']=='reader':assert hashes['decoder']==reference['decoder']
  if p['phase']=='reader' and p['attempts']==512:
   reader_end=hashes['reader'];reader_opt_end=support.state_sha(p['optimizers']['reader'])
   boundary={k:copy for k,copy in p.items() if k in ('model','scaler','torch_rng','cuda_rng','python_rng','numpy_rng','orders','history','cursors','exposure')}
   boundary_digest=support.state_sha(boundary)
  if p['phase']=='realizer':
   assert hashes['reader']==reader_end and support.state_sha(p['optimizers']['reader'])==reader_opt_end
   if p['attempts']==512:
    assert not p['optimizers']['realizer']['state']
    assert support.state_sha({k:v for k,v in p.items() if k in boundary})==boundary_digest
   else:assert p['optimizers']['realizer']['state']
  if p['attempts']>0:assert p['optimizers']['reader']['state']
  checkpoint_review.append({'phase':p['phase'],'attempts':p['attempts'],'updates':p['updates'],'cursor':p['cursor'],
   'sha256':meta['sha256'],'algorithm_sha256':meta['algorithm_sha256'],'scaler_tracker':p['scaler']['_growth_tracker']})
  if p['attempts']==1024:final_payload=p
  else:del p
 p=final_payload
 assert p['exposure']==result['exposure']
 hist=p['history'];r=[h for h in hist if h['phase']=='reader'];d=[h for h in hist if h['phase']=='realizer']
 assert len(r)==len(d)==512 and all(h['outcome']=='UPDATED' for h in hist)
 assert sum(h['teacher_condition_rows'] for h in d)==2486 and sum(h['predicted_condition_rows'] for h in d)==1610
 assert all(h['teacher_condition_rows']==8 and h['predicted_condition_rows']==0 for h in d[:256])
 assert 2048-1610==result['exposure']['fallback_teacher_rows']
 for phase in ('reader','realizer'):
  assert [i for h in hist if h['phase']==phase for i in h['rows']]==p['orders'][phase]
  for start in range(0,4096,1024):assert sorted(p['orders'][phase][start:start+1024])==list(range(1024))
 targets=json.loads((Path(plan['artifact'])/'targets.json').read_text(encoding='utf-8'))
 assert sum(len(targets['train'][i]['answer'].encode()) for h in d for i in h['rows'])==80632
 assert support.core_sha(ROOT)==support.CORE_SHA
 review.update(status='ROOT_V4_RAW_CALLS_PHASE_FULLSTATE_VERIFIED',additional_model_calls=0,
  raw_generation_rows=240,headcheck_rows=80,actual_calls=result['actual_calls'],
  actual_call_total=sum(result['actual_calls'].values()),checkpoint_review=checkpoint_review,
  phase_boundary_model_readerAdam_scaler_ALLRNG_orders_history_cursors_exposure_identical=True,
  reader_frozen_during_realization=True,decoder_frozen_during_reading=True,VP_and_factor_embeddings_fixed=True,
  exposure_verified=result['exposure'],result_sha256=support.sha(out/'result.json'),audit_script_sha256=support.sha(__file__),
  production_CUDA_restore_executed=False,CPU_phase_restore_tests=11,E2_final_accessed=False,
  span_logit_limitation='Raw pointer logits were not persisted. Span accuracy is independently scored from saved predicted indices and checked gold byte offsets; a pointer argmax cannot be independently rederived.',
  development_targets_met=False,interpretation='Known-wording pair score improved, held wording regressed; learned positional binding not broad language generalization.',verified_at=time.time())
 (out/'root_result.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in review.items() if k not in ('baseline_generation','reader_phase_headcheck','final_generation','zero_latent_generation','checkpoint_review')},ensure_ascii=True))
 for name,score in review['final_generation'].items():print(name,json.dumps(score,ensure_ascii=True))

if __name__=='__main__':run()
