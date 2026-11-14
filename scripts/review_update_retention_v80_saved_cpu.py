"""Independent V80 lastLinear-only fullstate/raw proof, CPU no extra NN."""
import collections,gc,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import critical_mask
from research.update_retention_objective_v80 import OBJECTIVE
from research.competing_margin_objective_v78 import MARGIN_COEFFICIENTS
from research.lossless_raw_tensor_v80 import unpack
from research.critical_retention_objective_v75 import COEFFICIENTS
from review_factual_cycle_v70_saved_cpu import verify_output
import hashlib

def raw_panel(p,run,label,rows,records):
    n=correct=NLL=errors=decoded=0;classes=collections.Counter();details=[]
    for i,(row,record) in enumerate(zip(rows,records)):
        file=run/f'raw_{label}_{i:02d}.pt';assert support.sha(file)==record['raw_sha256'];a=unpack(torch.load(file,map_location='cpu',weights_only=True));z,y=a['logits'],a['targets'];b=row['prompt'].encode()
        expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256
        assert torch.equal(y.flatten(),expected) and a['Exact_entries']==0 and a['source_id']==row['id'] and a['source_SHA']==hashlib.sha256(b).hexdigest()
        if 'fuzzy_commit_reference' in a:
            ref=a['fuzzy_commit_reference'];assert ref['key']=='fuzzy_commit' and ref['SHA']==p['files'][ref['path']] and ref['embedding_sha']==a['embedding_sha']
            old=torch.load(ref['path'],map_location='cpu',weights_only=True);assert old['fuzzy_commit'].count_nonzero() and torch.equal(old['targets'],y);del old
        else:assert a['fuzzy_commit'].count_nonzero()
        assert verify_output(z,record['ids'],z.numel()//258 if label=='long' else 16385)==record['output']
        active=y!=257;count=int(active.sum());ok=int(((z.argmax(-1)==y)&active).sum());ce=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'))
        assert ok==record['correct'] and math.isclose(ce,record['NLL_sum'],rel_tol=1e-6,abs_tol=4*torch.finfo(torch.float32).eps*count+1e-5)
        out=record['output'].encode();err=sum(x!=v for x,v in zip(b,out))+abs(len(b)-len(out));changes=[]
        for j,(x,v) in enumerate(zip(b,out)):
            if x!=v:
                classes['digit' if 48<=x<=57 else 'upper' if 65<=x<=90 else 'lower' if 97<=x<=122 else 'other']+=1
                if len(changes)<32:changes.append(dict(byte=j,source=b[max(0,j-28):j+35].decode(errors='replace'),output=out[max(0,j-28):j+35].decode(errors='replace')))
        n+=count;correct+=ok;NLL+=ce;decoded+=err;details.append(dict(row=i,id=row['id'],source_bytes=len(b),output_bytes=len(out),decoded_errors=err,changes=changes))
    return dict(rows=len(rows),NLL=NLL/n,byte_EOS_accuracy=correct/n,raw_errors=n-correct,decoded_errors=decoded,decoded_error_classes=dict(classes),exact_auxiliary=sum(r['prompt']==s['output'] for r,s in zip(rows,records)),details=details)


def main():
 torch.set_num_threads(4);run=ROOT/'runs/update_retention_20261113_v80';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json')
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
  names=s['optimizer_names'][0];stored=s['storage_delta_names'];assert names==['readout.readout.net.2.weight','readout.readout.net.2.bias'] and len(stored)==7 and stored==e['active_names'] and set(stored)==set(e['active_weights'])
  prefix=[n for n in stored if n not in names];assert len(prefix)==5 and all(torch.equal(model[n],warm[n]) for n in prefix)
  assert e['storage_scope']==dict(serialized_override_names=stored,trainable_optimizer_names=names,frozen_override_names=prefix)
  assert support.state_sha({k:model[k] for k in prefix})==s['frozen_head_prefix_SHA']==p['storage_delta_scope']['frozen_prefix_SHA']
  assert support.state_sha({k:v for k,v in model.items() if k not in names})==s['frozen_except_last2_SHA']==support.state_sha({k:v for k,v in warm.items() if k not in names})
  assert torch.equal(s['symbol_weights'],parent['symbol_weights'])
  assert s['schema']=='fdt_guarded_readout_v80_fullstate' and s['objective']==OBJECTIVE and s['objective_report']==p['objective_report'] and s['plan_sha256']==support.sha(run/'plan.json')
  assert s['cache_cursor']==len(s['cache_refs'])==cursor and s['attempts']==s['cursor']==attempt and s['updates']==(result['updates'] if attempt else 0) and s['rejections']==(result['rejected_attempts'] if attempt else 0) and s['skips']==0 and s['inflight'] is None and s['new_dev_refs']==[] and s['gradients']=={}
  assert s['orders']==orders and [h['rows'] for h in s['history']]==orders[:attempt]
  assert support.state_sha({k:v for k,v in model.items() if k not in stored})==s['embedding_sha']==parent['embedding_sha']==result['frozen_upstream_SHA']
  assert len(s['optimizer']['state'])==(2 if attempt else 0) and all(int(v['step'])==s['updates'] for v in s['optimizer']['state'].values())
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
   assert s['calls']==dict(encode=1088,readout=1088,reference_linear=1,stream_chunk=0,denoise=0)
  else:final=s
  states.append(dict(directory=folder,SHA=meta['sha256'],algorithm_SHA=meta['algorithm_sha256'],bytes=f.stat().st_size,attempt=attempt,teacher_cursor=cursor,Adam_states=len(s['optimizer']['state']),scaler=s['scaler'],standalone=False))
 assert all(not torch.equal(final['model'][k],warm[k]) for k in final['optimizer_names'][0])
 assert final['calls']==result['calls']==dict(encode=3076,readout=3418,reference_linear=817,stream_chunk=258,denoise=0) and final['long_block_work']==2322 and final['exposure']==result['exposure']
 counts=collections.Counter();per_source={};inverse={m['row']:m for h in final['history'] for m in h['micros']}
 for i,(row,label,ref) in enumerate(zip(rows,labels,final['cache_refs'])):
  assert set(ref)=={'targets','targets_SHA','cache_key','source_id'} and support.sha(ref['targets'])==ref['targets_SHA']
  t=torch.load(ref['targets'],map_location='cpu',weights_only=True);b=row['prompt'].encode();y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256
  assert torch.equal(y.flatten(),expected) and t['source_SHA']==support.hashlib.sha256(b).hexdigest() and t['source_id']==row['id'] and t['embedding_sha']==final['embedding_sha'] and t['teacher_sha']==final['teacher_sha']
  key=support.state_sha((final['embedding_sha'],row['id'],support.sha(d/'manifest.json'),support.hashlib.sha256(b).hexdigest()))
  assert t['cache_key']==ref['cache_key']==key and t['schema']=='V80_TRAIN_loss_only_teacher_margin'
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
 assert len(final['anchor_refs'])==100 and final['reference_final_ref']==result['TRAIN_final_reference']
 assert final['updates']+final['skips']+final['rejections']==384 and final['reference_patch_work']==817*7487
 reference_checks=sum(len(h['update_receipt']['reference_checks']) for h in final['history']);assert reference_checks+2==final['calls']['reference_linear']
 assert sum(h['rejected'] for h in final['history'])==137 and sum(h['update_receipt']['accepted'] for h in final['history'])==247
 for h in final['history']:
  receipt=h['update_receipt'];checks=receipt['reference_checks'];assert receipt['Adam_proposals']==1 and receipt['Adam_steps_committed']==int(receipt['accepted'])
  assert [c['alpha'] for c in checks]==[1.,.5,.25,.125][:len(checks)] and all(not c['accept'] for c in checks[:-1])
  if receipt['accepted']:assert checks[-1]['accept'] and checks[-1]['measurements']['new_critical_flips']==checks[-1]['measurements']['new_noncritical_EOS_flips']==0
  else:assert len(checks)==4 and not checks[-1]['accept']
 assert sum(h['update_receipt']['accepted'] for h in final['history'])==final['updates']
 bank=final['bank_ref'];assert support.sha(bank['features'])==bank['features_SHA'] and support.sha(bank['targets'])==bank['targets_SHA']
 features=torch.load(bank['features'],map_location='cpu',weights_only=True);target=torch.load(bank['targets'],map_location='cpu',weights_only=True)
 assert set(features)=={'schema','hidden','source_indices','embedding_sha','frozen_prefix_sha'} and 'labels' not in features
 assert features['source_indices']==target['source_indices']==final['anchor_indices'] and len(features['hidden'])==7487
 hs=[];ys=[];cs=[];flags=[]
 mapping={v['index']:v for v in final['anchor_refs']}
 for index in final['anchor_indices']:
  ref=mapping[index];assert support.sha(ref['features'])==ref['SHA'] and support.sha(ref['targets'])==ref['targets_SHA']
  f=torch.load(ref['features'],map_location='cpu',weights_only=True);t=torch.load(ref['targets'],map_location='cpu',weights_only=True)
  assert set(f)=={'schema','hidden','source_id','cache_key','embedding_sha','frozen_prefix_sha'} and f['cache_key']==t['cache_key'] and f['source_id']==rows[index]['id']
  assert f['embedding_sha']==final['embedding_sha'] and f['frozen_prefix_sha']==final['frozen_head_prefix_SHA'];hs.append(f['hidden']);ys.append(t['labels']);cs.append(t['critical_mask']);flags.append(t['teacher_correct'])
 assert torch.equal(features['hidden'],torch.cat(hs)) and torch.equal(target['labels'],torch.cat(ys)) and torch.equal(target['critical_mask'],torch.cat(cs))
 initial_correct=(target['initial_argmax'].long()==target['labels'])&(target['labels']!=257);assert torch.equal(initial_correct,target['initial_correct']) and int(initial_correct.sum())==117600
 finalref=final['reference_final_ref'];assert support.sha(finalref['path'])==finalref['SHA'];fp=torch.load(finalref['path'],map_location='cpu',weights_only=True)
 assert fp['features_SHA']==bank['features_SHA'] and fp['targets_SHA']==bank['targets_SHA'] and int(((fp['argmax'].long()!=target['labels'])&initial_correct).sum())==fp['initial_protected_flips']==0
 bank_proof=dict(TRAIN_sources=100,hidden_rows=7487,protected_positions=117600,protected_critical=int((initial_correct&target['critical_mask']).sum()),initial_rowwise_correct_flags_disagree=int(((torch.cat(flags)!=initial_correct)&(target['labels']!=257)).sum()),final_protected_flips=0,initial_correct_from_saved_argmax_only_NOT_recomputed_full_logits=True,reference_checks=reference_checks,actual_reference_linear_calls=817)
 receipts=dict(collections.Counter(str(h['update_receipt']['alpha']) for h in final['history']))
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
  before=unpack(torch.load(run/f'raw_new_baseline_{i:02d}.pt',map_location='cpu',weights_only=True))
  after=unpack(torch.load(run/f'raw_new_English_{i:02d}.pt',map_location='cpu',weights_only=True))
  assert torch.equal(before['fuzzy_commit'],after['fuzzy_commit']);samecommit+=1
 parity=io.load_json(run/'execution_parity.json');assert support.sha(run/'raw_execution_parity.pt')==parity['raw_SHA']
 raw=unpack(torch.load(run/'raw_execution_parity.pt',map_location='cpu',weights_only=True))
 assert torch.equal(raw['native']['logits'].argmax(-1),raw['stream']['logits'].argmax(-1)) and parity['commit_relative_L2']<=1e-4
 assert all(v['commit_relative_L2']<=1e-4 and v['current_head_all_argmax_identical'] for v in io.load_json(run/'fresh_long_cache_parity.json'))
 owned=sum(f.stat().st_size for f in run.rglob('*') if f.is_file());assert owned<p['max_output_bytes'] and not torch.cuda.is_initialized()
 proof=dict(status='ROOT_V80_SAVED_REVIEW_PASS',all_pins_rehashed=len(p['files']),all3_parentlinked_states_independently_reconstructed=states,
  V75_warm_all_model_exact_initial_digest=initial_digest,mandatory_standalone_V74_storage_parent=True,all_frozen_upstream_identical=True,
  only2_final_readout_tensors_changed_five_prefix_frozen=True,fresh2Adam_steps247_enabledscaler_tracker384_rejected137_ALLRNG_phase_cursor_exposure=True,frozen_V75_symbol_weights_all3states_exact=True,all384groups_1920micro_history_rows_verified=True,
  teacher1024_loss_only_packets_and_capped_loss_normalization_verified=True,TRAIN_anchor_counts=dict(counts),
  teacher_limitation='p(gold)/correct/margin/competitor saved with source checks; full teacher logits not saved, no independent full teacher argmax/margin recomputation',
  all470raw_targets_argmax_NLL_independent_UTF8_FSM_and_outputs_verified=True,same_new64_Commit_bitwise=samecommit,
  TRAIN_reference_bank=bank_proof,actual_update_alpha_histogram=receipts,all_new_raw_lossless_carrier_unpacked_before_NLL_UTF8_FSM=True,native_stream_parity_and_fresh_two262KiB_source_verified=True,owned_output_bytes=owned,panels=panels,extra_model_calls=0,
  CUDA_initialized=False,Exact_entries=0,copy_bytes=0,denoise_calls=0)
 io.atomic_json(run/'root_saved_review.json',proof)
 print(json.dumps(dict(status=proof['status'],states=states,anchors=dict(counts),TRAIN_reference_bank=bank_proof,alpha_histogram=receipts,owned_output_bytes=owned,panels={k:{a:b for a,b in v.items() if a!='details'} for k,v in panels.items()})))
if __name__=='__main__':main()
