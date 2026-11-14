"""Saved-only independent V73 fullstate/teacher/raw review; no NN forwards."""
import gc,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import training_weights,critical_mask
from fuzzy_extraction_engine_v73 import SCHEMA,OBJECTIVE
from review_factual_cycle_v70_saved_cpu import raw_panel
def main():
    torch.set_num_threads(4);run=ROOT/'runs/fuzzy_extraction_20261112_v73';p=io.load_json(run/'plan.json');d=Path(p['data']);e=Path(p['eval_data']);result=io.load_json(run/'result.json')
    assert io.load_json(run/'ROOT_terminal_native.json')['all4owned_and_proxy_gone'] and not io.load_json(run/'ROOT_terminal_native.json')['PythonProcesses']
    owner=io.load_json(run/'final_status.json');assert owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone']
    for file,sha in p['files'].items():assert support.sha(file)==sha,file
    rows=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');byid={t['id']:t for t in labels};orders=io.load_json(d/'orders.json');weights,obj=training_weights(rows);obj.update(objective=OBJECTIVE,anchor_coefficient=.05,LR_groups=[1e-6,3e-6,3e-6]);assert obj==p['objective_report']
    initial=torch.load(run/'initial/checkpoints/state_0000.pt',map_location='cpu',weights_only=True);parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True)
    assert support.state_sha(parent)==p['cycle_algorithm_sha'] and support.state_sha(parent['model'])==support.state_sha(initial['model']) and len(parent['optimizer']['state'])==7
    names=sum(initial['optimizer_names'],[]);assert len(names)==23 and all(not n.startswith('readout.readout.') for n in names) and 'readout.role_projection.weight' in names
    frozen=support.state_sha({k:v for k,v in parent['model'].items() if k not in names});assert frozen==result['fixed_tensors_SHA_before_after']
    del parent;gc.collect();proofs=[];final=None;transition=None
    for directory,step,teacher in [('initial',0,0),('teacher_complete',0,960),('model',1,960),('model',480,960),('model',960,960)]:
        file=run/directory/'checkpoints'/f'state_{step:04d}.pt';m=io.load_json(file.with_suffix('.json'));assert support.sha(file)==m['sha256'];s=torch.load(file,map_location='cpu',weights_only=True);assert support.state_sha(s)==m['algorithm_sha256']
        assert s['schema']==SCHEMA and s['objective']==OBJECTIVE and s['objective_report']==obj and s['plan_sha256']==support.sha(run/'plan.json') and torch.equal(s['symbol_weights'],torch.tensor(weights))
        assert s['attempts']==s['updates']==s['cursor']==step and s['skips']==0 and s['inflight'] is None and not s['gradients']
        assert s['teacher_cursor']==len(s['teacher_refs'])==teacher and s['orders']==orders and s['optimizer_names']==initial['optimizer_names'] and [h['rows'] for h in s['history']]==orders[:step]
        assert support.state_sha({k:v for k,v in s['model'].items() if k not in names})==frozen and s['teacher_model_sha']==initial['teacher_model_sha']
        assert s['parent_reference']==initial['parent_reference'] and s['Exact_entries']==s['copy_bytes']==s['denoise_calls']==0
        assert len(s['optimizer']['state'])==(23 if step else 0) and all(int(x['step'])==step for x in s['optimizer']['state'].values())
        assert s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==step and s['torch_rng'].numel() and len(s['cuda_rng'])==1 and s['numpy_rng']['keys'] and s['python_rng']
        chosen=[rows[o[0]] for o in orders[:step]];size=sum(len(r['prompt'].encode()) for r in chosen);critical=sum(sum(span['end']-span['start'] for span in byid[r['id']]['critical_spans']) for r in chosen)
        assert s['exposure']==dict(teacher_rows=teacher,teacher_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows[:teacher]),live_rows=step,live_UTF8_bytes=size,live_symbols=size+step,live_critical_positions=critical)
        for h in s['history']:
            expected=h['weighted_byte_CE']+.1*h['EOS_CE']+.1*h['hard_body_CE']+.1*h['tail_false_EOS_UL']+.25*h['critical_CE']+.05*h['correct_noncritical_anchor_KL']
            assert math.isclose(h['loss'],expected,rel_tol=1e-6,abs_tol=1e-6) and not h['AMP_skip'] and h['GPU_peak_bytes']<=12*1024**3
        if not step:
            for key in ('model','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):assert support.state_sha(s[key])==support.state_sha(initial[key])
        if directory=='teacher_complete':transition=s
        if step==960:final=s
        proofs.append(dict(directory=directory,attempt=step,teacher_cursor=teacher,bytes=m['bytes'],SHA=m['sha256'],algorithm_SHA=m['algorithm_sha256'],Adam_states=len(s['optimizer']['state']),scaler=s['scaler']))
        if s is not final and s is not transition:del s
        gc.collect()
    assert all(not torch.equal(final['model'][n],initial['model'][n]) for n in names)
    assert final['calls']==result['calls']==dict(encode=2304,readout=2304,stream_chunk=738,denoise=0) and final['long_block_work']==6642
    assert transition['calls']==dict(encode=1088,readout=1088,stream_chunk=0,denoise=0) and transition['teacher_refs']==final['teacher_refs']
    anchors=0;inverse={order[0]:i for i,order in enumerate(orders)}
    for i,(row,ref) in enumerate(zip(rows,final['teacher_refs'])):
        assert support.sha(ref['file'])==ref['SHA'];t=torch.load(ref['file'],map_location='cpu',weights_only=True);b=row['prompt'].encode();n=len(b)
        assert set(t)=={'schema','labels','critical_mask','teacher_gold_probability','teacher_correct','source_id','source_SHA','source_bytes','source_version','teacher_model_SHA','teacher_key'}
        key=support.state_sha((initial['teacher_model_sha'],row['id'],support.sha(d/'manifest.json'),hashlib.sha256(b).hexdigest()))
        assert t['source_id']==row['id'] and t['source_bytes']==n and t['teacher_model_SHA']==initial['teacher_model_sha'] and t['teacher_key']==ref['key']==key
        y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:n]=torch.tensor(list(b));expected[n]=256;assert torch.equal(y.flatten(),expected)
        mask=critical_mask(y,byid[row['id']]['critical_spans'],n);assert torch.equal(mask,t['critical_mask'])
        probability=t['teacher_gold_probability'];assert torch.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all() and t['teacher_correct'].dtype==torch.bool
        eligible=(y<256)&~mask&t['teacher_correct'];count=int(eligible.sum());assert final['history'][inverse[i]]['anchor_positions']==count;anchors+=count
    del initial,transition,final;gc.collect();panels={};data=Path(p['parent_data'])
    specs=[('new_baseline',io.load_json(e/'short_features.json')),('OWT',io.load_json(p['retention_OWT'])),('retained_English',io.load_json(data/'retained_English.json')),
        ('prior_English',io.load_json(data/'dev_features.json')),('old_expression',io.load_json(Path(p['old_expression_data'])/'dev_features.json')),('new_English',io.load_json(e/'short_features.json')),('long',io.load_json(e/'long_features.json'))]
    total=0
    for label,source in specs:
        records=io.load_json(run/(label+'_reconstructions.json'));paths=[run/f'raw_{label}_{i:02d}.pt' for i in range(len(source))]
        for f,r in zip(paths,records):assert support.sha(f)==r['raw_sha256']
        score=raw_panel(run,source,records,paths,None if label=='long' else 16385);panels[label]=score;total+=len(records)
        assert abs(score['NLL']-result['evaluations'][label]['NLL'])<4*torch.finfo(torch.float32).eps and score['byte_EOS_accuracy']==result['evaluations'][label]['byte_EOS_accuracy']
    assert total==382
    pr=io.load_json(run/'execution_parity.json');assert support.sha(run/'raw_execution_parity.pt')==pr['raw_SHA'];s=torch.load(run/'raw_execution_parity.pt',map_location='cpu',weights_only=True)
    assert torch.equal(s['native']['logits'].argmax(-1),s['stream']['logits'].argmax(-1)) and pr['all_argmax_identical'] and pr['commit_relative_L2']<=1e-4
    assert len(list(run.glob('raw*.pt')))==383 and not torch.cuda.is_initialized()
    report=dict(status='ROOT_V73_SAVED_REVIEW_PASS',checkpoints=proofs,all_parent_model_tensors_exact=True,fresh23Adam=True,old7Adam_scalerALLRNG_preserved=True,
        all_nonactive_tensors_including_final_readout7_prior_reasoner_identical=True,all_active23_upstream_tensors_changed=True,
        teacher960_packet_SHA_source_target_mask_epoch_checked=True,teacher_only_loss_packets_no_Commit_H_features=True,anchor_positions=anchors,
        teacher_probability_limitation='full teacher logits not saved; correct flags/probability ranges and eligibility checked, no independent teacher argmax recomputation',
        all_five_fullstates_Adam_scaler_ALLRNG_phase_cursor_order_exposure_calls_verified=True,all382_raw_targets_argmax_NLL_independentUTF8FSM_verified=True,execution_parity_raw_verified=True,
        panels=panels,extra_model_calls=0,CUDA_initialized=False,Exact_entries=0,copy_bytes=0,denoise_calls=0,limitations=p['limitations'])
    io.atomic_json(run/'root_saved_review.json',report)
    print(json.dumps(dict(status=report['status'],checkpoints=proofs,anchor_positions=anchors,panels={k:{a:b for a,b in v.items() if a!='details'} for k,v in panels.items()})))
if __name__=='__main__':main()
