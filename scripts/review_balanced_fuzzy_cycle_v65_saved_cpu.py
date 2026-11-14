"""Saved-only review of live language/fuzzy/language trajectory and output text."""
import collections,difflib,gc,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.codec_commit_stop_v56 import reconstruct_commit_logits

def main():
    torch.set_num_threads(4);run=ROOT/'runs/balanced_fuzzy_cycle_20261112_v65'
    p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json');terminal=io.load_json(run/'final_status.json');native=io.load_json(run/'ROOT_terminal_native.json')
    assert terminal['status']=='COMPLETE' and terminal['exit']['all_owned_processes_gone'] and native['allFourOwnedGone'] and native['nativePythonCount']==0
    for name,digest in p['files'].items():assert support.sha(name)==digest,name
    initial=torch.load(run/'initial/checkpoints/state_0000.pt',weights_only=True,map_location='cpu')
    active=sum(initial['optimizer_names'],[]);active_set=set(active);frozen={k:v for k,v in initial['model'].items() if k not in active_set}
    warm=torch.load(p['cycle_source'],weights_only=True,map_location='cpu')
    assert support.state_sha(initial['model'])==support.state_sha(warm['model']) and len(warm['optimizer']['state'])==30 and not initial['optimizer']['state']
    del warm;gc.collect()
    train=io.load_json(Path(p['data'])/'train.json');dev=io.load_json(Path(p['data'])/'dev.json');orders=io.load_json(Path(p['data'])/'orders.json')
    from research.fuzzy_cycle_objective_v65 import training_weights,OBJECTIVE
    expected_weights,objective=training_weights(train)
    assert objective==p['objective_report']
    checks=[];changes={};last=None
    for step in (0,1,256,512):
        path=run/('initial' if step==0 else 'model')/'checkpoints'/f'state_{step:04d}.pt';meta=io.load_json(path.with_suffix('.json'))
        assert support.sha(path)==meta['sha256'];s=torch.load(path,weights_only=True,map_location='cpu');assert support.state_sha(s)==meta['algorithm_sha256']
        assert s['objective']==OBJECTIVE and s['objective_report']==objective and torch.equal(s['symbol_weights'],torch.tensor(expected_weights))
        assert s['schema']=='fdt_live_fuzzy_roundtrip_v65_fullstate' and s['plan_sha256']==support.sha(run/'plan.json')
        assert s['attempts']==s['updates']==s['cursor']==step and s['skips']==0
        assert s['orders']==orders and [h['rows'] for h in s['history']]==orders[:step] and s['optimizer_names']==initial['optimizer_names']
        assert s['exact_policy']==p['exact_policy'] and s['Exact_entries']==s['copy_bytes']==s['denoise_calls']==0 and not s['gradients']
        assert s['frozen_sha']==initial['frozen_sha'] and support.state_sha({k:v for k,v in s['model'].items() if k not in active_set})==support.state_sha(frozen)
        assert s['parent_reference']==initial['parent_reference']
        selected=[train[ids[0]] for ids in orders[:step]];size=sum(len(r['prompt'].encode()) for r in selected)
        assert s['exposure']==dict(rows=step,target_UTF8_bytes=size,active_symbols=size+step)
        assert len(s['optimizer']['state'])==(len(active) if step else 0)
        assert all(int(v['step'])==step for v in s['optimizer']['state'].values())
        assert s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==step
        assert s['torch_rng'].numel() and len(s['cuda_rng'])==1 and s['python_rng'] and s['numpy_rng']['keys']
        assert all(not h['AMP_skip'] and h['Exact_entries']==0 and h['GPU_peak_bytes']<=12*1024**3 and math.isfinite(h['loss']) for h in s['history'])
        if step==512:
            assert s['calls']==dict(encode=528,readout=532,denoise=0) and s['exposure']==result['exposure']
            for prefix in ('encoder.core.language.blocks.','encoder.core.value.','encoder.core.condition.','readout.'):
                keys=[k for k in active if k.startswith(prefix)]
                changes[prefix]=dict(active_tensors=len(keys),changed_tensors=sum(not torch.equal(s['model'][k],initial['model'][k]) for k in keys),
                    max_parameter_delta=max(float((s['model'][k]-initial['model'][k]).abs().max()) for k in keys))
                assert changes[prefix]['changed_tensors']>0
            last=dict(final_SHA=meta['sha256'],final_algorithm_SHA=meta['algorithm_sha256'],final_bytes=meta['bytes'],
                GPU_peak_bytes=max(h['GPU_peak_bytes'] for h in s['history']),last16_mean_byte_CE=sum(h['byte_CE'] for h in s['history'][-16:])/16)
        checks.append(dict(step=step,SHA=meta['sha256'],bytes=meta['bytes'],Adam_states=len(s['optimizer']['state']),scaler=s['scaler']))
        del s;gc.collect()
    readings={};scores={};predictions={};controls=io.load_json(run/'zero_fuzzy_content_controls.json')
    for label in ('baseline','final'):
        saved=io.load_json((Path(p['baseline_run'])/'final_reconstructions.json') if label=='baseline' else run/'final_reconstructions.json');readings[label]=[];NLL=correct=count=0;class_errors=collections.Counter();class_totals=collections.Counter();predictions[label]=[]
        for i,row in enumerate(dev):
            raw=row['prompt'].encode();raw_path=Path(p['baseline_run'])/f'raw_final_{i:02d}.pt' if label=='baseline' else run/f'raw_final_{i:02d}.pt';s=torch.load(raw_path,weights_only=True,map_location='cpu');z,y=s['logits'],s['targets']
            expected=torch.full_like(y.flatten(),257);expected[:len(raw)]=torch.tensor(list(raw));expected[len(raw)]=256;assert torch.equal(y.flatten(),expected)
            assert s['Exact_entries']==0 and s['Exact_policy']==p['exact_policy'] and s['fuzzy_commit'].count_nonzero()
            ce=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'));active_mask=y!=257;pred=z.argmax(-1)
            n=int(active_mask.sum());ok=int(((pred==y)&active_mask).sum());NLL+=ce;correct+=ok;count+=n
            decoded=reconstruct_commit_logits(z);out=decoded['text'];stored=saved[i]
            # CPU/GPU float32 log-softmax/reduction differs slightly even when
            # every saved logit, argmax and byte is identical. Bound per symbol.
            assert stored['source']==row['prompt'] and stored['output']==out and stored['ids']==decoded['ids'] and stored['correct']==ok and abs(stored['NLL_sum']-ce)<1e-7*n+1e-5
            predictions[label].append(pred[y<256].clone())
            for a,b in zip(y[y<256].tolist(),pred[y<256].tolist()):
                kind='upper' if 65<=a<=90 else 'lower' if 97<=a<=122 else 'digit' if 48<=a<=57 else 'other';class_totals[kind]+=1;class_errors[kind]+=int(a!=b)
            a=collections.Counter(re.findall(r'\w+',row['prompt'].lower()));b=collections.Counter(re.findall(r'\w+',out.lower()))
            differences=[dict(kind=t,source=row['prompt'][x:xx],output=out[u:uu],source_offset=x,
                context=row['prompt'][max(0,x-55):min(len(row['prompt']),xx+55)]) for t,x,xx,u,uu in difflib.SequenceMatcher(None,row['prompt'],out,autojunk=False).get_opcodes() if t!='equal']
            readings[label].append(dict(id=row['id'],source=row['prompt'],output=out,source_bytes=len(raw),exact=out==row['prompt'],
                EOS=decoded['EOS'],UTF8=decoded['UTF8'],errors=n-ok,character_similarity=difflib.SequenceMatcher(None,row['prompt'],out,autojunk=False).ratio(),
                word_overlap_F1=2*sum((a&b).values())/max(1,sum(a.values())+sum(b.values())),changes=differences))
        metric=dict(NLL=NLL/count,byte_EOS_accuracy=correct/count,exact_sentences=sum(r['exact'] for r in readings[label]),rows=16,
            UTF8=sum(r['UTF8'] for r in readings[label]),EOS=sum(r['EOS'] for r in readings[label]),symbols=count,Exact_entries=0)
        assert abs(metric['NLL']-result[label]['NLL'])<1e-7 and metric['byte_EOS_accuracy']==result[label]['byte_EOS_accuracy']
        for k in ('exact_sentences','UTF8','EOS','symbols'):assert metric[k]==result[label][k]
        scores[label]=dict(metric,errors=count-correct,class_errors=dict(class_errors),class_totals=dict(class_totals))
    corrected=new_errors=0
    for i,row in enumerate(dev):
        y=torch.tensor(list(row['prompt'].encode()));a=predictions['baseline'][i]!=y;b=predictions['final'][i]!=y
        corrected+=int((a&~b).sum());new_errors+=int((~a&b).sum())
    report=dict(status='INDEPENDENT_V65_SAVED_REVIEW_PASS',checkpoints=checks,all_parent_tensors_exact=True,parent30Adam_unchanged=True,baseline_reused=True,
        fresh_joint_Adam_states=len(active),actual_active_encoder_and_decoder_changes=changes,all_frozen_tensors_including_prior_identical=True,
        fullstate_scaler_ALLRNG_orders_cursor_history_exposure_verified=True,raw32_logits_targets_decoded_text_verified=True,
        Exact_policy=p['exact_policy'],Exact_entries=0,copy_bytes=0,denoise_calls=0,baseline=scores['baseline'],final=scores['final'],
        corrected_byte_positions=corrected,new_byte_errors=new_errors,control4_worker_metrics_not_independent_raw_logits=controls,
        **last,extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
        limitations='observed correlated DEV; coarse Commit topology and EOS guarding retained; character similarity is not semantic equivalence; structural fuzzy codec not abstract meaning certification')
    io.atomic_json(run/'root_saved_review.json',report);io.atomic_json(run/'root_reading_similarity.json',readings)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoints','control4_worker_metrics_not_independent_raw_logits')}))
if __name__=='__main__':main()
