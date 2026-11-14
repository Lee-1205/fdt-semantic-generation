"""ROOT independent state/teacher/bank/decision-trace review; extra NN0."""
import collections,gc,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.readout_decision_trace_v88 import decode as decode_trace
from research.lossless_raw_tensor_v80 import unpack
from research.critical_bank_storage_v87 import load as load_bank
from review_v80_semantic_severity_cpu import numeric_severity

def review_trace(info,record,p):
    f=Path(record['trace']);assert support.sha(f)==record['SHA'];t=torch.load(f,map_location='cpu',weights_only=True);old=unpack(torch.load(info['raw'],map_location='cpu',weights_only=True));y=old['targets'].flatten();valid=y!=257
    source=info['row']['prompt'].encode();expected=torch.full_like(y,257);expected[:len(source)]=torch.tensor(list(source));expected[len(source)]=256
    assert torch.equal(y,expected) and t['source_id']==record['id']==info['row']['id'] and t['source_SHA']==hashlib.sha256(source).hexdigest() and t['Exact_entries']==0
    assert t['argmax'].shape==t['gold_log_probability'].shape==y.shape and torch.isfinite(t['gold_log_probability']).all()
    decoded=decode_trace(t);assert torch.equal(decoded['ids'],t['decoded_ids']) and decoded['EOS']==record['EOS'] and decoded['UTF8']==record['UTF8'] and (decoded['text']==info['row']['prompt'])==record['exact_auxiliary']
    assert record['symbols']==int(valid.sum()) and record['correct']==int(((t['argmax'].long()==y)&valid).sum()) and record['NLL_sum']==float(-t['gold_log_probability'][valid].double().sum())
    # Explicit legal sets independently constructed from UTF8 ranges. These
    # verify sampled full scores; complete258-class logits are not saved.
    sets=[]
    for slots in (1,2,3,4):sets.append(list(range(128))+[256]+(list(range(0xC2,0xE0)) if slots>=2 else [])+(list(range(0xE0,0xF0)) if slots>=3 else [])+(list(range(0xF0,0xF5)) if slots>=4 else []))
    sets += [list(range(lo,hi+1)) for lo,hi in ((0x80,0xBF),(0xA0,0xBF),(0x90,0xBF),(0x80,0x9F),(0x80,0x8F))]
    checked=0
    for sample,patch in zip(t['sample_full_logits'],t['sample_patch_indices'].tolist()):
        assert sample.dtype==torch.float32 and sample.shape==(16,258) and torch.isfinite(sample).all();a=patch*16;b=a+16
        assert torch.equal(sample.argmax(-1),t['argmax'][a:b].long())
        for state,allowed in enumerate(sets):
            ids=torch.tensor(sorted(allowed));winner=ids[sample[:,ids].argmax(-1)];assert torch.equal(winner,t['choices'][state,a:b].long())
        lp=sample.gather(-1,y[a:b,None]).squeeze(-1)-torch.logsumexp(sample,-1)
        assert torch.allclose(lp,t['gold_log_probability'][a:b],atol=3e-5,rtol=1e-6);checked+=16
    out=decoded['text'].encode();errors=sum(x!=z for x,z in zip(source,out))+abs(len(source)-len(out))
    return dict(id=record['id'],output=decoded['text'],decoded_errors=errors,sample_bytes_verified=checked)

def main():
    torch.set_num_threads(4);run=ROOT/'runs/critical_repair_20261113_v88';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json');term=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert term['AllOwnedGone'] and not term['PythonProcesses'] and term['BeforeAnalysis'] and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
    for f,s in p['files'].items():assert support.sha(f)==s,f
    storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True);sourceenv=torch.load(p['source'],map_location='cpu',weights_only=True);warm=dict(storage['model']);warm.update(sourceenv['active_weights']);states=[];initial=final=None
    for folder,cursor,attempt in [('initial',0,0),('teacher_complete',1024,0),('model',1024,108)]:
        f=run/folder/'runtime.pt';meta=io.load_json(f.with_suffix('.json'));assert support.sha(f)==meta['SHA'];e=torch.load(f,map_location='cpu',weights_only=True);model=dict(storage['model']);model.update(e['active_weights']);s=dict(e['runtime'],model=model)
        assert support.state_sha(s)==meta['reconstructed_fullstate_SHA']==e['reconstructed_fullstate_SHA'] and s['precision_policy']==p['precision_policy'] and s['plan_sha256']==support.sha(run/'plan.json')
        assert len(e['active_names'])==7 and len(s['optimizer_names'][0])==2 and s['cache_cursor']==len(s['cache_refs'])==cursor and s['attempts']==s['cursor']==attempt and s['inflight'] is None
        names=s['optimizer_names'][0];assert names==p['trainable_names'] and s['storage_delta_names']==e['active_names']
        assert all(torch.equal(v,warm[k]) for k,v in model.items() if k not in names)
        assert support.state_sha({k:v for k,v in model.items() if k not in names})==s['frozen_except_last2_SHA']
        assert support.state_sha({k:v for k,v in model.items() if k not in e['active_names']})==s['embedding_sha']==sourceenv['runtime']['embedding_sha']
        assert torch.equal(s['symbol_weights'],sourceenv['runtime']['symbol_weights']) and s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==attempt
        assert s['Exact_entries']==s['copy_bytes']==s['denoise_calls']==0 and len(s['cuda_rng'])==1 and s['python_rng'] and s['numpy_rng']['keys'] and s['torch_rng'].numel()
        if not attempt:assert not s['optimizer']['state'] and all(torch.equal(v,warm[k]) for k,v in model.items())
        else:assert len(s['optimizer']['state'])==2 and all(int(v['step'])==16 for v in s['optimizer']['state'].values())
        states.append(dict(phase=folder,file_SHA=meta['SHA'],algorithm_SHA=support.state_sha(s),bytes=f.stat().st_size,all_except_last2_bitwise_source=True))
        if folder=='initial':initial=s
        if folder=='model':final=s
        del e;gc.collect()
    assert final['updates']==16 and final['rejections']==92 and final['skips']==0 and len(final['history'])==108 and final['bank_ready'] and final['bank_state'] is None
    assert all(not h['update_receipt']['accepted'] for h in final['history'][-32:]) and final['history'][-33]['update_receipt']['accepted']
    checks=sum(len(h['update_receipt']['reference_checks']) for h in final['history']);assert checks+2==final['calls']['reference_linear']==393
    assert sum(h['update_receipt']['accepted'] for h in final['history'])==16 and sum(not h['AMP_skip'] and not h['update_receipt']['accepted'] for h in final['history'])==92
    rows=io.load_json(Path(p['data'])/'train_features.json');targets=io.load_json(Path(p['data'])/'train_targets.json');orders=io.load_json(Path(p['data'])/'orders.json')[:192];assert final['orders']==orders and [h['rows'] for h in final['history']]==orders[:108]
    exposure=final['exposure'];assert exposure['head_rows']==exposure['head_encoder_rows']==540 and exposure['teacher_rows']==exposure['encoder_rows']==1024 and exposure['head_UTF8_bytes']==exposure['head_encoder_UTF8_bytes']==sum(len(rows[i]['prompt'].encode()) for group in orders[:108] for i in group)
    bank=load_bank(final['bank_ref'],ROOT,support.sha);assert support.state_sha(bank)==final['bank_ref']['bank_runtime_SHA'];cal=bank['calibration'];broad=set(bank['config']['broad_indices']);weak=0;weak_in_broad=0
    for i,(ref,part) in enumerate(zip(final['cache_refs'],bank['parts'])):
        assert support.sha(ref['targets'])==ref['targets_SHA'];t=torch.load(ref['targets'],map_location='cpu',weights_only=True);raw=rows[i]['prompt'].encode();y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:len(raw)]=torch.tensor(list(raw));expected[len(raw)]=256
        assert torch.equal(y.flatten(),expected) and t['source_id']==rows[i]['id']==ref['source_id'] and t['source_SHA']==hashlib.sha256(raw).hexdigest() and t['precision_policy']==p['precision_policy']
        mask=torch.zeros(y.numel(),dtype=torch.bool)
        for span in targets[i]['critical_spans']:mask[span['start']:span['end']]=True
        assert torch.equal(mask.reshape(y.shape),t['critical_mask']);wm=t['critical_mask']&t['teacher_correct']&(t['teacher_margin']<=1.)&(y!=257)
        keep=list(range(len(y))) if i in broad else wm.any(-1).nonzero().flatten().tolist();assert part['source_patch_indices']==keep and part['hidden'].dtype==torch.float32
        idx=torch.tensor(keep,dtype=torch.long)
        for key in ('labels','critical_mask','teacher_correct','teacher_margin'):assert torch.equal(part[key],t[key][idx])
        assert torch.equal(part['weak_critical_mask'],wm[idx]);weak+=int(wm.sum());weak_in_broad+=int(wm.sum()) if i in broad else 0
    assert weak==bank['weak_count']==result['weak_critical_positions']==152 and bank['patch_count']==7623 and cal['teacher_correct_disagreements']==0 and cal['max_row_batch_margin_difference']<=.001
    assert torch.equal(cal['top1_mask'],(cal['initial_argmax'].long()==cal['labels'])&(cal['labels']!=257))
    finalref=torch.load(final['reference_final']['path'],map_location='cpu',weights_only=True);assert support.sha(final['reference_final']['path'])==final['reference_final']['SHA']
    flips=(finalref['argmax'].long()!=cal['labels'])&cal['top1_mask'];erosion=(finalref['margin']<cal['required_margin'])&cal['weak_floor_mask'];assert not flips.any() and not erosion.any()
    items=io.load_json(p['eval_inputs']);look={(x['panel'],x['index']):x for x in items};panels={};cards=[];samples=0
    for phase in ('baseline','final'):
        records=io.load_json(run/f'assessment/{phase}/reconstructions.json');decoded=[]
        for record in records:
            d=review_trace(look[record['panel'],record['index']],record,p);decoded.append(dict(d,panel=record['panel'],index=record['index']));samples+=d['sample_bytes_verified']
        for panel in sorted({x['panel'] for x in decoded}):panels[phase+'_'+panel]=[x for x in decoded if x['panel']==panel]
    base_new=io.load_json(Path(p['baseline_new_run'])/'reconstructions.json');panels['baseline_new']=[dict(id=r['id'],output=r['output']) for r in base_new]
    base_long=io.load_json(ROOT/'runs/update_retention_20261113_v80/long_reconstructions.json');panels['baseline_long']=[dict(id=r['id'],output=r['output']) for r in base_long]
    diffs={}
    for panel in ('new','observed75','observed74','long'):
        changes=[];old=panels['baseline_'+panel];new=panels['final_'+panel]
        assert len(old)==len(new)
        for before,after in zip(old,new):
            assert before['id']==after['id'];row=look[panel,after['index']]['row'];source=row['prompt'].encode();a=before['output'].encode();b=after['output'].encode()
            if a!=b:
                positions=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y];changes.append(dict(id=after['id'],index=after['index'],changed_offsets=positions,old_bytes=len(a),new_bytes=len(b),
                    contexts=[dict(offset=j,source=source[max(0,j-55):j+65].decode(errors='replace'),before=a[max(0,j-55):j+65].decode(errors='replace'),after=b[max(0,j-55):j+65].decode(errors='replace')) for j in positions]))
        diffs[panel]=changes
    newlabels=io.load_json(Path(io.load_json(ROOT/'runs/update_retention_20261113_v80/plan.json')['eval_data'])/'targets.json')
    severity={k:numeric_severity(newlabels,panels[k+'_new']) for k in ('baseline','final')}
    io.atomic_json(run/'root_decoded_panels.json',panels)
    report=dict(status='ROOT_V88_SAVED_STATE_TRACE_TEACHER_BANK_REVIEW_COMPLETE',worker_status=result['status'],owner_status=owner['status'],actual108attempt16update92reject=True,whole192_not_completed=True,
        states=states,actual_calls=final['calls'],actual_total_calls=sum(final['calls'].values()),reference_checks=checks,initial_teacher1024_SHA_source_mask_verified=True,
        all_frozen_except2_bitwise=True,old_source_AdamRNG_preserved=True,enabled_scaler_tracker108=True,all_stored_RNG_present=True,CUDA_restart_NOT_tested=True,
        bank=dict(patches=7623,all_TRAIN_weak_critical=weak,weak_in_broad100=weak_in_broad,weak_outside_broad100=weak-weak_in_broad,protected_top1=int(cal['top1_mask'].sum()),final_top1_flips=0,final_weak_margin_violations=0,CPU_GPU_initial_margin_max_diff=cal['max_row_batch_margin_difference']),
        reconstructed_trace_sample_byte_positions=samples,trace_limit='Allstate UTF8 choices/argmax/goldLP replayed, sampled full258scores verified. NOT full258 raw argmax or NLL reconstruction on unsaved logits.',
        changed_outputs=diffs,new_numeric_severity=severity,extra_CPU_GPU_model_calls=0,training_complete=False,wholelanguage_complete=False,automatic_promotion=False)
    io.atomic_json(run/'root_saved_review.json',report);print(json.dumps(dict(status=report['status'],bank=report['bank'],changed_output_rows={k:len(v) for k,v in diffs.items()},new_numeric={k:v['counts'] for k,v in severity.items()},long_changes=diffs['long'],calls=final['calls'],sample_verified=samples),ensure_ascii=False))

if __name__=='__main__':main()
