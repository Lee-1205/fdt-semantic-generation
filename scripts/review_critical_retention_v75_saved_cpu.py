"""Independently reconstruct parent-linked states and verify ALL saved outputs."""
import collections,gc,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import critical_mask
from research.critical_retention_objective_v75 import OBJECTIVE,COEFFICIENTS
from review_factual_cycle_v70_saved_cpu import verify_output

def raw_panel(p,run,label,rows,records):
    n=correct=NLL=errors=decoded=0;classes=collections.Counter();details=[]
    for i,(row,record) in enumerate(zip(rows,records)):
        file=run/f'raw_{label}_{i:02d}.pt';assert support.sha(file)==record['raw_sha256'];a=torch.load(file,map_location='cpu',weights_only=True);z,y=a['logits'],a['targets'];b=row['prompt'].encode()
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
    torch.set_num_threads(4);run=ROOT/'runs/critical_retention_20261112_v75';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json');term=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert term['all4owned_and_proxy_gone'] and not term['PythonProcesses'] and term['before_result_analysis'] and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone']
    for f,sha in p['files'].items():assert support.sha(f)==sha,f
    parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True);assert support.state_sha(parent)==p['cycle_algorithm_sha'] and len(parent['optimizer']['state'])==7
    d=Path(p['data']);rows=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');orders=io.load_json(d/'orders.json');states=[];first=final=None
    for folder,attempt,cursor in [('initial',0,0),('teacher_complete',0,640),('model',640,640)]:
        file=run/folder/'runtime.pt';meta=io.load_json(run/folder/'runtime.json');assert support.sha(file)==meta['sha256'];e=torch.load(file,map_location='cpu',weights_only=True);ref=e['parent_reference']
        assert ref['path']==p['cycle_source'] and ref['SHA']==p['cycle_sha'] and ref['model_SHA']==support.state_sha(parent['model']) and ref['model_key']=='model' and not e['standalone']
        model=dict(parent['model']);model.update(e['active_weights']);s=dict(e['runtime'],model=model);assert support.state_sha(s)==meta['algorithm_sha256']==e['reconstructed_fullstate_SHA']
        names=sum(s['optimizer_names'],[]);assert names==e['active_names'] and set(names)==set(e['active_weights']) and len(names)==7 and all(k.startswith('readout.readout.') for k in names)
        assert s['schema']=='fdt_live_critical_retention_v75_fullstate' and s['objective']==OBJECTIVE and s['objective_report']==p['objective_report'] and s['plan_sha256']==support.sha(run/'plan.json')
        assert s['cache_cursor']==len(s['cache_refs'])==cursor and s['attempts']==s['updates']==s['cursor']==attempt and s['skips']==0 and s['inflight'] is None and s['new_dev_refs']==[] and s['gradients']=={}
        assert s['orders']==orders and [h['rows'] for h in s['history']]==orders[:attempt] and s['embedding_sha']==result['frozen_upstream_SHA']
        assert support.state_sha({k:v for k,v in model.items() if k not in names})==s['embedding_sha'] and s['embedding_sha']==parent['embedding_sha']
        assert len(s['optimizer']['state'])==(7 if attempt else 0) and all(int(v['step'])==attempt for v in s['optimizer']['state'].values())
        assert s['optimizer']['param_groups'][0]['lr']==1e-5 and s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==attempt and s['torch_rng'].numel() and len(s['cuda_rng'])==1 and s['python_rng'] and s['numpy_rng']['keys']
        size=sum(len(rows[o[0]]['prompt'].encode()) for o in orders[:attempt]);critical=sum(sum(v['end']-v['start'] for v in labels[o[0]]['critical_spans']) for o in orders[:attempt])
        assert s['exposure']==dict(encoder_rows=cursor,teacher_rows=cursor,encoder_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows[:cursor]),head_rows=attempt,head_UTF8_bytes=size,head_symbols=size+attempt,head_critical_positions=critical,head_encoder_rows=attempt,head_encoder_UTF8_bytes=size)
        for h in s['history']:
            loss=h['weighted_byte_CE']+.1*h['EOS_CE']+.1*h['hard_body_CE']+.1*h['tail_false_EOS_UL']+sum(c*h[k] for k,c in COEFFICIENTS.items())
            assert math.isclose(h['loss'],loss,rel_tol=1e-6,abs_tol=1e-6) and not h['AMP_skip'] and h['GPU_peak_bytes']<=12*1024**3
        if folder=='initial':
            assert support.state_sha(model)==support.state_sha(parent['model']);first=s
        elif folder=='teacher_complete':
            for k in ('model','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):assert support.state_sha(s[k])==support.state_sha(first[k])
            assert s['calls']==dict(encode=704,readout=704,stream_chunk=0,denoise=0)
        else:final=s
        states.append(dict(directory=folder,SHA=meta['sha256'],algorithm_SHA=meta['algorithm_sha256'],bytes=file.stat().st_size,attempt=attempt,teacher_cursor=cursor,Adam_states=len(s['optimizer']['state']),scaler=s['scaler'],standalone=False))
    assert all(not torch.equal(final['model'][k],parent['model'][k]) for k in final['optimizer_names'][0])
    assert final['calls']==result['calls']==dict(encode=1412,readout=1690,stream_chunk=258,denoise=0) and final['long_block_work']==2322 and result['exposure']==final['exposure']
    counts=collections.Counter();inverse={v[0]:i for i,v in enumerate(orders)}
    for i,(row,label,ref) in enumerate(zip(rows,labels,final['cache_refs'])):
        assert set(ref)=={'targets','targets_SHA','cache_key','source_id'} and support.sha(ref['targets'])==ref['targets_SHA'];t=torch.load(ref['targets'],map_location='cpu',weights_only=True);b=row['prompt'].encode();y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256
        assert torch.equal(y.flatten(),expected) and t['source_SHA']==hashlib.sha256(b).hexdigest() and t['source_id']==row['id'] and t['embedding_sha']==final['embedding_sha'] and t['teacher_sha']==final['teacher_sha']
        expectedkey=support.state_sha((final['embedding_sha'],row['id'],support.sha(d/'manifest.json'),hashlib.sha256(b).hexdigest()));assert t['cache_key']==ref['cache_key']==expectedkey
        mask=critical_mask(y,label['critical_spans'],len(b));assert torch.equal(mask,t['critical_mask']) and torch.isfinite(t['teacher_gold_probability']).all() and ((t['teacher_gold_probability']>=0)&(t['teacher_gold_probability']<=1)).all()
        critical_correct=int((mask&t['teacher_correct']).sum());critical_wrong=int((mask&~t['teacher_correct']).sum());noncritical=int(((y<256)&~mask&t['teacher_correct']).sum());h=final['history'][inverse[i]]
        assert h['correct_critical_positions']==critical_correct and h['wrong_critical_positions']==critical_wrong and h['noncritical_anchor_positions']==noncritical
        counts.update(correct_critical=critical_correct,wrong_critical=critical_wrong,noncritical_anchor=noncritical)
    assert not list(run.rglob('*.features.pt'))
    del parent,first,final,s,model;gc.collect();panels={};new=io.load_json(d/'dev_features.json')
    specs=[('new_baseline',new),('OWT',io.load_json(p['retention_OWT'])),('retained_English',io.load_json(Path(p['parent_data'])/'retained_English.json')),
        ('old_untrained128',io.load_json(Path(p['old_eval_data'])/'short_features.json')),('old_expression',io.load_json(Path(p['old_expression_data'])/'dev_features.json')),
        ('observed_V74_DEV',io.load_json(Path(p['observed_V74_data'])/'dev_features.json')),('new_English',new),('long',io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2])]
    n=0
    for label,source in specs:
        records=io.load_json(run/(label+'_reconstructions.json'));score=raw_panel(p,run,label,source,records);panels[label]=score;n+=len(records)
        assert abs(score['NLL']-result['evaluations'][label]['NLL'])<4*torch.finfo(torch.float32).eps and score['byte_EOS_accuracy']==result['evaluations'][label]['byte_EOS_accuracy']
    assert n==406 and len(list(run.glob('raw*.pt')))==407
    parity=io.load_json(run/'execution_parity.json');assert support.sha(run/'raw_execution_parity.pt')==parity['raw_SHA'];raw=torch.load(run/'raw_execution_parity.pt',map_location='cpu',weights_only=True)
    assert torch.equal(raw['native']['logits'].argmax(-1),raw['stream']['logits'].argmax(-1)) and parity['commit_relative_L2']<=1e-4
    for v in io.load_json(run/'fresh_long_cache_parity.json'):assert v['commit_relative_L2']<=1e-4 and v['current_head_all_argmax_identical']
    owned=sum(f.stat().st_size for f in run.rglob('*') if f.is_file());assert owned<1024**3 and not torch.cuda.is_initialized()
    proof=dict(status='ROOT_V75_SAVED_REVIEW_PASS',all616pins_rehashed=True,all3_parentlinked_runtime_states_independently_reconstructed=states,parent_required_NOT_standalone=True,
        all_frozen_upstream_identical=True,all7_final_readout_tensors_changed=True,fresh7Adam_steps640_scaler1024_tracker640_ALLRNG_phase_cursor_order_exposure_verified=True,
        teacher640_loss_only_packets_verified=True,teacher_limitation='p(gold)/correct flags only; full teacher logits NOT saved; no independent teacher argmax recomputation',TRAIN_anchor_counts=dict(counts),
        all406_raw_targets_argmax_NLL_independent_UTF8_FSM_and_outputs_verified=True,old_Commit_readonly_references_pinned_and_targets_match=True,native_stream_parity_raw_verified=True,
        fresh2maxlong_source_Commit_parity_current_head_argmax_pass=True,no_feature_cache_written=True,owned_output_bytes=owned,panels=panels,extra_model_calls=0,CUDA_initialized=False,Exact_entries=0,copy_bytes=0,denoise_calls=0)
    io.atomic_json(run/'root_saved_review.json',proof);print(json.dumps(dict(status=proof['status'],states=states,anchors=dict(counts),owned_output_bytes=owned,panels={k:{a:b for a,b in v.items() if a!='details'} for k,v in panels.items()})))
if __name__=='__main__':main()
