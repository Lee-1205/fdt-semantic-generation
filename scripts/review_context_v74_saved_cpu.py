"""Resolve linked initial weights independently; verify all saved V74 evidence."""
import gc,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import training_weights,critical_mask
from research.frozen_verbalizer_objective_v72 import OBJECTIVE
from context_engine_v74 import SCHEMA
from review_factual_cycle_v70_saved_cpu import raw_panel
def main():
    torch.set_num_threads(4);run=ROOT/'runs/context_verbalizer_20261112_v74';p=io.load_json(run/'plan.json');d=Path(p['data']);result=io.load_json(run/'result.json');parentrun=Path(p['baseline_run'])
    term=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json');assert term['all4owned_and_proxy_gone'] and not term['PythonProcesses'] and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone']
    for file,sha in p['files'].items():assert support.sha(file)==sha,file
    parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True);assert support.state_sha(parent)==p['cycle_algorithm_sha'] and len(parent['optimizer']['state'])==23
    meta=io.load_json(run/'initial_runtime.json');assert support.sha(run/'initial_runtime.pt')==meta['SHA'];initial=torch.load(run/'initial_runtime.pt',map_location='cpu',weights_only=True);ref=initial.pop('model_reference')
    assert ref==meta['model_reference'] and ref['path']==p['cycle_source'] and ref['SHA']==p['cycle_sha'] and ref['key']=='model' and ref['model_digest']==support.state_sha(parent['model'])
    initial['model']=parent['model'];assert support.state_sha(initial)==meta['reconstructed_full_algorithm_SHA'] and not initial['optimizer']['state'] and initial['attempts']==0 and not initial['inflight']
    names=sum(initial['optimizer_names'],[]);assert len(names)==7 and all(n.startswith('readout.readout.') for n in names)
    frozen=support.state_sha({k:v for k,v in parent['model'].items() if k not in names});assert frozen==result['frozen_upstream_SHA']
    rows=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');byid={t['id']:t for t in labels};orders=io.load_json(d/'orders.json');weights,obj=training_weights(rows);obj.update(objective=OBJECTIVE,anchor_coefficient=.05,decoder_LR=3e-5)
    assert obj==p['objective_report'];proofs=[];complete=final=None
    for directory,step in [('cache_complete',0),('model',640)]:
        file=run/directory/'checkpoints'/f'state_{step:04d}.pt';m=io.load_json(file.with_suffix('.json'));assert support.sha(file)==m['sha256'];s=torch.load(file,map_location='cpu',weights_only=True);assert support.state_sha(s)==m['algorithm_sha256']
        assert s['schema']==SCHEMA and s['objective']==OBJECTIVE and s['objective_report']==obj and s['plan_sha256']==support.sha(run/'plan.json') and torch.equal(s['symbol_weights'],torch.tensor(weights))
        assert s['attempts']==s['updates']==s['cursor']==step and s['skips']==0 and s['inflight'] is None and not s['gradients'] and s['cache_cursor']==len(s['cache_refs'])==640 and len(s['new_dev_refs'])==64
        assert s['orders']==orders and s['optimizer_names']==initial['optimizer_names'] and [h['rows'] for h in s['history']]==orders[:step]
        assert support.state_sha({k:v for k,v in s['model'].items() if k not in names})==frozen and s['embedding_sha']==initial['embedding_sha'] and s['teacher_sha']==initial['teacher_sha']
        assert len(s['optimizer']['state'])==(7 if step else 0) and all(int(v['step'])==step for v in s['optimizer']['state'].values())
        assert s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==step and s['torch_rng'].numel() and len(s['cuda_rng'])==1 and s['numpy_rng']['keys'] and s['python_rng']
        selected=[rows[o[0]] for o in orders[:step]];size=sum(len(r['prompt'].encode()) for r in selected);critical=sum(sum(v['end']-v['start'] for v in byid[r['id']]['critical_spans']) for r in selected)
        assert s['exposure']==dict(encoder_rows=640,teacher_rows=640,encoder_UTF8_bytes=1039965,head_rows=step,head_UTF8_bytes=size,head_symbols=size+step,head_critical_positions=critical)
        for h in s['history']:
            value=h['weighted_byte_CE']+.1*h['EOS_CE']+.1*h['hard_body_CE']+.1*h['tail_false_EOS_UL']+.25*h['critical_CE']+.05*h['correct_noncritical_anchor_KL']
            assert math.isclose(h['loss'],value,rel_tol=1e-6,abs_tol=1e-6) and not h['AMP_skip'] and h['GPU_peak_bytes']<=12*1024**3
        if step==0:
            for key in ('model','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):assert support.state_sha(s[key])==support.state_sha(initial[key])
            complete=s
        else:final=s
        proofs.append(dict(directory=directory,attempt=step,SHA=m['sha256'],algorithm_SHA=m['algorithm_sha256'],bytes=m['bytes'],Adam_states=len(s['optimizer']['state']),scaler=s['scaler']))
    assert final['calls']==result['calls']==dict(encode=708,readout=1626,stream_chunk=258,denoise=0) and final['long_block_work']==2322 and complete['calls']==dict(encode=704,readout=704,stream_chunk=0,denoise=0)
    assert all(not torch.equal(final['model'][n],parent['model'][n]) for n in names) and complete['cache_refs']==final['cache_refs'] and complete['new_dev_refs']==final['new_dev_refs']
    inverse={o[0]:i for i,o in enumerate(orders)};anchors=0
    for i,(row,r) in enumerate(zip(rows,final['cache_refs'])):
        assert support.sha(r['features'])==r['features_SHA'] and support.sha(r['targets'])==r['targets_SHA'];f=torch.load(r['features'],map_location='cpu',weights_only=True);t=torch.load(r['targets'],map_location='cpu',weights_only=True);b=row['prompt'].encode()
        assert set(f)=={'schema','semantic','memory_revisions','embedding_sha','source_id','source_version','source_SHA','cache_key'}
        key=support.state_sha((frozen,row['id'],support.sha(d/'manifest.json'),hashlib.sha256(b).hexdigest()));assert f['cache_key']==t['cache_key']==r['cache_key']==key and f['source_SHA']==hashlib.sha256(b).hexdigest() and f['source_id']==row['id']
        y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256;assert torch.equal(y.flatten(),expected)
        mask=critical_mask(y,byid[row['id']]['critical_spans'],len(b));assert torch.equal(mask,t['critical_mask']);prob=t['teacher_gold_probability'];assert torch.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        eligible=(y<256)&~mask&t['teacher_correct'];n=int(eligible.sum());assert final['history'][inverse[i]]['anchor_positions']==n;anchors+=n
    del parent,initial,complete,final;gc.collect();panels={};data=Path(p['parent_data']);old_eval=Path(p['old_eval_data']);new=io.load_json(d/'dev_features.json')
    specs=[('new_baseline',new),('OWT',io.load_json(p['retention_OWT'])),('retained_English',io.load_json(data/'retained_English.json')),('old_untrained128',io.load_json(old_eval/'short_features.json')),
        ('old_expression',io.load_json(Path(p['old_expression_data'])/'dev_features.json')),('new_English',new),('cached_long',io.load_json(old_eval/'long_features.json')[:2]),('long',io.load_json(old_eval/'long_features.json')[:2])]
    count=0
    for label,source in specs:
        records=io.load_json(run/(label+'_reconstructions.json'));files=[run/f'raw_{label}_{i:02d}.pt' for i in range(len(source))]
        for file,r in zip(files,records):assert support.sha(file)==r['raw_sha256']
        score=raw_panel(run,source,records,files,None if 'long' in label else 16385);panels[label]=score;count+=len(records)
        assert abs(score['NLL']-result['evaluations'][label]['NLL'])<4*torch.finfo(torch.float32).eps and score['byte_EOS_accuracy']==result['evaluations'][label]['byte_EOS_accuracy']
    assert count==344 and len(list(run.glob('raw*.pt')))==345
    for v in io.load_json(run/'fresh_long_cache_parity.json'):assert v['commit_relative_L2']<=1e-4 and v['all_argmax_identical'] and v['same_decoded_output']
    pr=io.load_json(run/'execution_parity.json');assert support.sha(run/'raw_execution_parity.pt')==pr['raw_SHA'];raw=torch.load(run/'raw_execution_parity.pt',map_location='cpu',weights_only=True);assert torch.equal(raw['native']['logits'].argmax(-1),raw['stream']['logits'].argmax(-1))
    assert not torch.cuda.is_initialized()
    report=dict(status='ROOT_V74_SAVED_REVIEW_PASS',source_linked_initial_state_full_recursive_digest_independently_resolved=True,initial_model_NOT_standalone_parent_required=True,
        two_standalone_fullstates=proofs,all_V73_nonhead_tensors_frozen=True,all7_readout_tensors_changed=True,parent23Adam_scalerALLRNG_preserved=True,fresh7Adam=True,
        cache640_pairs_SHA_source_epoch_labels_masks_verified=True,anchor_positions=anchors,all_state_Adam_scaler_ALLRNG_phase_order_cursor_exposure_callbudget_verified=True,
        all344_raw_targets_argmax_NLL_independentUTF8FSM_verified=True,execution_parity_raw_verified=True,two_fresh_source_long_cache_parity_verified=True,
        teacher_limitation='p(gold)/correct flags saved; full teacher logits not saved, no independent teacher argmax recomputation',panels=panels,extra_model_calls=0,CUDA_initialized=False,Exact_entries=0,copy_bytes=0,denoise_calls=0,limitations=p['limitations'])
    io.atomic_json(run/'root_saved_review.json',report);print(json.dumps(dict(status=report['status'],states=proofs,anchor_positions=anchors,panels={k:{a:b for a,b in v.items() if a!='details'} for k,v in panels.items()})))
if __name__=='__main__':main()
