"""Independent V77 repeated-head fullstate/raw proof, CPU no extra NN."""
import collections,gc,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import critical_mask
from research.competing_margin_objective_v78 import OBJECTIVE,MARGIN_COEFFICIENTS
from research.critical_retention_objective_v75 import COEFFICIENTS
from review_critical_retention_v75_saved_cpu import raw_panel

def main():
 torch.set_num_threads(4);run=ROOT/'runs/competing_margin_20261113_v78';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json')
 terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
 assert terminal['all4owned_and_proxy_gone'] and not terminal['PythonProcesses'] and terminal['before_result_analysis']
 assert owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
 for f,s in p['files'].items():assert support.sha(f)==s,f
 storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True);oldenv=torch.load(p['cycle_source'],map_location='cpu',weights_only=True)
 warm=dict(storage['model']);warm.update(oldenv['active_weights']);parent=dict(oldenv['runtime'],model=warm)
 assert support.state_sha(parent)==p['cycle_algorithm_sha'] and len(parent['optimizer']['state'])==7 and parent['updates']==640
 d=Path(p['data']);rows=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');orders=io.load_json(Path(p['balance_data'])/'orders.json');states=[];first=final=None
 for folder,attempt,cursor in [('initial',0,0),('teacher_complete',0,1024),('model',384,1024)]:
  f=run/folder/'runtime.pt';meta=io.load_json(run/folder/'runtime.json');assert support.sha(f)==meta['sha256']
  e=torch.load(f,map_location='cpu',weights_only=True);ref=e['parent_reference']
  assert ref['path']==p['storage_parent'] and ref['SHA']==p['storage_parent_sha'] and ref['model_key']=='model' and not e['standalone']
  assert ref['model_SHA']==support.state_sha(storage['model'])
  model=dict(storage['model']);model.update(e['active_weights']);s=dict(e['runtime'],model=model)
  assert support.state_sha(s)==meta['algorithm_sha256']==e['reconstructed_fullstate_SHA']
  names=s['optimizer_names'][0];assert names==e['active_names'] and set(names)==set(e['active_weights']) and len(names)==7 and all(k.startswith('readout.readout.') for k in names)
  assert torch.equal(s['symbol_weights'],parent['symbol_weights'])
  assert s['schema']=='fdt_grouped_competing_margin_v78_fullstate' and s['objective']==OBJECTIVE and s['objective_report']==p['objective_report'] and s['plan_sha256']==support.sha(run/'plan.json')
  assert s['cache_cursor']==len(s['cache_refs'])==cursor and s['attempts']==s['updates']==s['cursor']==attempt and s['skips']==0 and s['inflight'] is None and s['new_dev_refs']==[] and s['gradients']=={}
  assert s['orders']==orders and [h['rows'] for h in s['history']]==orders[:attempt]
  assert support.state_sha({k:v for k,v in model.items() if k not in names})==s['embedding_sha']==parent['embedding_sha']==result['frozen_upstream_SHA']
  assert len(s['optimizer']['state'])==(7 if attempt else 0) and all(int(v['step'])==attempt for v in s['optimizer']['state'].values())
  assert s['optimizer']['param_groups'][0]['lr']==1e-5 and s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==attempt
  assert s['torch_rng'].numel() and len(s['cuda_rng'])==1 and s['python_rng'] and s['numpy_rng']['keys']
  sourcebytes=sum(len(rows[i]['prompt'].encode()) for o in orders[:attempt] for i in o)
  expectedcritical=sum(sum(v['end']-v['start'] for v in labels[i]['critical_spans']) for o in orders[:attempt] for i in o)
  assert s['exposure']==dict(encoder_rows=cursor,teacher_rows=cursor,encoder_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows[:cursor]),
    head_rows=5*attempt,head_UTF8_bytes=sourcebytes,head_symbols=sourcebytes+5*attempt,head_critical_positions=expectedcritical,head_encoder_rows=5*attempt,head_encoder_UTF8_bytes=sourcebytes)
  for group in s['history']:
   assert len(group['micros'])==5 and group['clip_and_Adam_calls']==1 and group['rows']==[m['row'] for m in group['micros']]
   assert not group['AMP_skip'] and abs(group['loss']-sum(.2*m['loss'] for m in group['micros']))<1e-6
   for h in group['micros']:
    value=h['weighted_byte_CE']+.1*h['EOS_CE']+.1*h['hard_body_CE']+.1*h['tail_false_EOS_UL']+sum(c*h[k] for k,c in COEFFICIENTS.items())+sum(c*h[k] for k,c in MARGIN_COEFFICIENTS.items())
    assert math.isclose(h['loss'],value,rel_tol=1e-6,abs_tol=1e-6) and h['weight']==.2
    c,w=h['critical_positions'],h['wrong_critical_positions'];den=max(w,c/8)
    assert h['wrong_critical_denominator']==den and h['wrong_critical_position_amplification']==(c/den if w else 0.) and h['wrong_critical_position_amplification']<=8.
  if folder=='initial':
   assert support.state_sha(model)==support.state_sha(warm);first=s
  elif folder=='teacher_complete':
   for k in ('model','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):assert support.state_sha(s[k])==support.state_sha(first[k])
   assert s['calls']==dict(encode=1088,readout=1088,stream_chunk=0,denoise=0)
  else:final=s
  states.append(dict(directory=folder,SHA=meta['sha256'],algorithm_SHA=meta['algorithm_sha256'],bytes=f.stat().st_size,attempt=attempt,teacher_cursor=cursor,Adam_states=len(s['optimizer']['state']),scaler=s['scaler'],standalone=False))
 assert all(not torch.equal(final['model'][k],warm[k]) for k in final['optimizer_names'][0])
 assert final['calls']==result['calls']==dict(encode=3076,readout=3418,stream_chunk=258,denoise=0) and final['long_block_work']==2322 and final['exposure']==result['exposure']
 counts=collections.Counter();per_source={};inverse={m['row']:m for h in final['history'] for m in h['micros']}
 for i,(row,label,ref) in enumerate(zip(rows,labels,final['cache_refs'])):
  assert set(ref)=={'targets','targets_SHA','cache_key','source_id'} and support.sha(ref['targets'])==ref['targets_SHA']
  t=torch.load(ref['targets'],map_location='cpu',weights_only=True);b=row['prompt'].encode();y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256
  assert torch.equal(y.flatten(),expected) and t['source_SHA']==support.hashlib.sha256(b).hexdigest() and t['source_id']==row['id'] and t['embedding_sha']==final['embedding_sha'] and t['teacher_sha']==final['teacher_sha']
  key=support.state_sha((final['embedding_sha'],row['id'],support.sha(d/'manifest.json'),support.hashlib.sha256(b).hexdigest()))
  assert t['cache_key']==ref['cache_key']==key and t['schema']=='V78_TRAIN_loss_only_teacher_margin'
  mask=critical_mask(y,label['critical_spans'],len(b));assert torch.equal(mask,t['critical_mask']) and torch.isfinite(t['teacher_gold_probability']).all()
  assert ((t['teacher_gold_probability']>=0)&(t['teacher_gold_probability']<=1)).all()
  cc=int((mask&t['teacher_correct']).sum());cw=int((mask&~t['teacher_correct']).sum());nc=int(((y<256)&~mask&t['teacher_correct']).sum());h=inverse[i]
  assert (h['correct_critical_positions'],h['wrong_critical_positions'],h['noncritical_anchor_positions'])==(cc,cw,nc)
  per_source[i]=(cc,cw,nc)
  margin=t['teacher_margin'];competitor=t['teacher_competitor'];assert margin.shape==y.shape and competitor.shape==y.shape and competitor.dtype==torch.int16 and torch.isfinite(margin).all() and (competitor.long()!=y).all()
  assert not ((margin<0)&t['teacher_correct']).any() and not ((margin>0)&~t['teacher_correct']).any()
  eligible=(y<256)&t['teacher_correct']&(margin<=3.)
  assert h['critical_margin_eligible']==int((eligible&mask).sum()) and h['noncritical_margin_eligible']==int((eligible&~mask).sum())
  counts.update(correct_critical=cc,wrong_critical=cw,noncritical_anchor=nc)
 for group in final['history']:
  for h in group['micros']:assert (h['correct_critical_positions'],h['wrong_critical_positions'],h['noncritical_anchor_positions'])==per_source[h['row']]
 assert not list(run.rglob('*.features.pt'))
 initial_digest=support.state_sha(first['model']);del storage,warm,parent,oldenv,first,final,s,model,e;gc.collect()
 dev=io.load_json(Path(p['eval_data'])/'features.json')
 specs=[('new_baseline',dev),('OWT',io.load_json(p['retention_OWT'])),('retained_English',io.load_json(Path(p['parent_data'])/'retained_English.json')),
  ('old_untrained128',io.load_json(Path(p['old_eval_data'])/'short_features.json')),('old_expression',io.load_json(Path(p['old_expression_data'])/'dev_features.json')),
  ('observed_V74_DEV',io.load_json(Path(p['observed_V74_data'])/'dev_features.json')),('observed_V75_DEV',io.load_json(Path(p['observed_V75_data'])/'dev_features.json')),
  ('new_English',dev),('long',io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2])]
 panels={};n=0
 for label,source in specs:
  records=io.load_json(run/(label+'_reconstructions.json'));score=raw_panel(p,run,label,source,records);panels[label]=score;n+=len(records)
  assert abs(score['NLL']-result['evaluations'][label]['NLL'])<4*torch.finfo(torch.float32).eps and score['byte_EOS_accuracy']==result['evaluations'][label]['byte_EOS_accuracy']
 assert n==470 and len(list(run.glob('raw*.pt')))==471
 samecommit=0
 for i in range(64):
  before=torch.load(run/f'raw_new_baseline_{i:02d}.pt',map_location='cpu',weights_only=True)
  after=torch.load(run/f'raw_new_English_{i:02d}.pt',map_location='cpu',weights_only=True)
  assert torch.equal(before['fuzzy_commit'],after['fuzzy_commit']);samecommit+=1
 parity=io.load_json(run/'execution_parity.json');assert support.sha(run/'raw_execution_parity.pt')==parity['raw_SHA']
 raw=torch.load(run/'raw_execution_parity.pt',map_location='cpu',weights_only=True)
 assert torch.equal(raw['native']['logits'].argmax(-1),raw['stream']['logits'].argmax(-1)) and parity['commit_relative_L2']<=1e-4
 assert all(v['commit_relative_L2']<=1e-4 and v['current_head_all_argmax_identical'] for v in io.load_json(run/'fresh_long_cache_parity.json'))
 owned=sum(f.stat().st_size for f in run.rglob('*') if f.is_file());assert owned<p['max_output_bytes'] and not torch.cuda.is_initialized()
 proof=dict(status='ROOT_V78_SAVED_REVIEW_PASS',all_pins_rehashed=len(p['files']),all3_parentlinked_states_independently_reconstructed=states,
  V75_warm_all_model_exact_initial_digest=initial_digest,mandatory_standalone_V74_storage_parent=True,all_frozen_upstream_identical=True,
  all7_final_readout_tensors_changed=True,fresh7Adam_steps384_enabledscaler_tracker384_ALLRNG_phase_cursor_exposure=True,frozen_V75_symbol_weights_all3states_exact=True,all384groups_1920micro_history_rows_verified=True,
  teacher1024_loss_only_packets_and_capped_loss_normalization_verified=True,TRAIN_anchor_counts=dict(counts),
  teacher_limitation='p(gold)/correct/margin/competitor saved with source checks; full teacher logits not saved, no independent full teacher argmax/margin recomputation',
  all470raw_targets_argmax_NLL_independent_UTF8_FSM_and_outputs_verified=True,same_new64_Commit_bitwise=samecommit,
  native_stream_parity_and_fresh_two262KiB_source_verified=True,owned_output_bytes=owned,panels=panels,extra_model_calls=0,
  CUDA_initialized=False,Exact_entries=0,copy_bytes=0,denoise_calls=0)
 io.atomic_json(run/'root_saved_review.json',proof)
 print(json.dumps(dict(status=proof['status'],states=states,anchors=dict(counts),owned_output_bytes=owned,panels={k:{a:b for a,b in v.items() if a!='details'} for k,v in panels.items()})))
if __name__=='__main__':main()
