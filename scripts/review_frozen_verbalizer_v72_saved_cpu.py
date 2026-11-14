"""Independent saved-only fullstates/cache/raw review. No model forwards."""
import collections,gc,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import critical_mask,training_weights
from research.frozen_verbalizer_objective_v72 import OBJECTIVE
from review_factual_cycle_v70_saved_cpu import raw_panel

def main():
    torch.set_num_threads(4);run=ROOT/'runs/frozen_verbalizer_20261112_v72';p=io.load_json(run/'plan.json');d=Path(p['data']);old=Path(p['baseline_run'])
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json');result=io.load_json(run/'result.json')
    assert terminal['all4owned_and_proxy_gone'] and not terminal['PythonProcesses'] and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone']
    for name,digest in p['files'].items():assert support.sha(name)==digest,name
    rows=io.load_json(d/'train_features.json');targets=io.load_json(d/'train_targets.json');orders=io.load_json(d/'orders.json');byid={t['id']:t for t in targets}
    weights,obj=training_weights(rows);obj.update(objective=OBJECTIVE,anchor_coefficient=.05,decoder_LR=3e-5);assert obj==p['objective_report']
    initial=torch.load(run/'initial/checkpoints/state_0000.pt',map_location='cpu',weights_only=True)
    parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True);assert support.state_sha(parent)==p['cycle_algorithm_sha']
    assert support.state_sha(parent['model'])==support.state_sha(initial['model']) and len(parent['optimizer']['state'])==30
    names=sum(initial['optimizer_names'],[]);assert len(names)==7 and all(n.startswith('readout.readout.') for n in names)
    frozen=support.state_sha({k:v for k,v in parent['model'].items() if k not in names});assert frozen==result['embedding_SHA_before_after']
    del parent;gc.collect();checkpoints=[];final=None;transition=None
    specs=[('initial',0,0),('cache_complete',0,640),('model',1,640),('model',320,640),('model',640,640)]
    for directory,step,cachecursor in specs:
        file=run/directory/'checkpoints'/f'state_{step:04d}.pt';meta=io.load_json(file.with_suffix('.json'));assert support.sha(file)==meta['sha256']
        s=torch.load(file,map_location='cpu',weights_only=True);assert support.state_sha(s)==meta['algorithm_sha256']
        assert s['schema']=='fdt_frozen_verbalizer_v72_fullstate' and s['objective']==OBJECTIVE and s['objective_report']==obj and s['plan_sha256']==support.sha(run/'plan.json')
        assert s['orders']==orders and s['optimizer_names']==initial['optimizer_names'] and torch.equal(s['symbol_weights'],torch.tensor(weights))
        assert s['attempts']==s['updates']==s['cursor']==step and s['skips']==0 and s['cache_cursor']==cachecursor==len(s['cache_refs'])
        assert s['Exact_entries']==s['copy_bytes']==s['denoise_calls']==0 and not s['gradients'] and s['parent_reference']==initial['parent_reference']
        assert s['embedding_sha']==frozen and s['teacher_sha']==initial['teacher_sha']
        assert support.state_sha({k:v for k,v in s['model'].items() if k not in names})==frozen
        assert len(s['optimizer']['state'])==(7 if step else 0) and all(int(v['step'])==step for v in s['optimizer']['state'].values())
        assert s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==step and s['torch_rng'].numel() and len(s['cuda_rng'])==1 and s['numpy_rng']['keys'] and s['python_rng']
        selected=[rows[a[0]] for a in orders[:step]];size=sum(len(r['prompt'].encode()) for r in selected)
        critical=sum(sum(v['end']-v['start'] for v in byid[r['id']]['critical_spans']) for r in selected)
        assert s['exposure']==dict(encoder_rows=cachecursor,teacher_rows=cachecursor,encoder_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows[:cachecursor]),
            head_rows=step,head_UTF8_bytes=size,head_symbols=size+step,head_critical_positions=critical)
        assert [h['rows'] for h in s['history']]==orders[:step]
        for h in s['history']:
            n=h['bytes'];assert h['hard_k']==min(64,max(1,math.ceil(.01*n))) and not h['AMP_skip'] and h['GPU_peak_bytes']<=12*1024**3
            expected=h['weighted_byte_CE']+.1*h['EOS_CE']+.1*h['hard_body_CE']+.1*h['tail_false_EOS_UL']+.25*h['critical_CE']+.05*h['correct_noncritical_anchor_KL']
            assert math.isclose(h['loss'],expected,rel_tol=1e-6,abs_tol=1e-6)
        if not step:
            for key in ('model','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):
                assert support.state_sha(s[key])==support.state_sha(initial[key])
        if directory=='cache_complete':transition=s
        if step==640:final=s
        checkpoints.append(dict(directory=directory,attempt=step,cache_cursor=cachecursor,SHA=meta['sha256'],algorithm_SHA=meta['algorithm_sha256'],bytes=meta['bytes'],Adam_states=len(s['optimizer']['state']),scaler=s['scaler']))
        if s is not transition and s is not final:del s
        gc.collect()
    assert all(not torch.equal(final['model'][n],initial['model'][n]) for n in names)
    assert final['calls']==result['calls']==dict(encode=676,readout=1434,denoise=0) and transition['calls']==dict(encode=672,readout=672,denoise=0)
    assert final['new_dev_refs']==transition['new_dev_refs'] and len(final['new_dev_refs'])==32
    anchors=criticaltotal=0
    for i,(row,ref) in enumerate(zip(rows,final['cache_refs'])):
        assert ref==transition['cache_refs'][i] and support.sha(ref['features'])==ref['features_SHA'] and support.sha(ref['targets'])==ref['targets_SHA']
        f=torch.load(ref['features'],map_location='cpu',weights_only=True);t=torch.load(ref['targets'],map_location='cpu',weights_only=True);b=row['prompt'].encode()
        assert set(f)=={'schema','semantic','memory_revisions','embedding_sha','source_id','source_version','source_SHA','cache_key'}
        digest=hashlib.sha256(b).hexdigest();key=support.state_sha((frozen,row['id'],support.sha(d/'manifest.json'),digest))
        assert f['source_id']==row['id'] and f['source_SHA']==digest and f['embedding_sha']==frozen and f['cache_key']==t['cache_key']==ref['cache_key']==key
        y=t['labels'];expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256;assert torch.equal(y.flatten(),expected)
        mask=critical_mask(y,byid[row['id']]['critical_spans'],len(b));assert torch.equal(mask,t['critical_mask'])
        prob=t['teacher_gold_probability'];correct=t['teacher_correct'];assert torch.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all() and correct.dtype==torch.bool
        eligible=(y<256)&~mask&correct;index=next(j for j,r in enumerate(orders) if r==[i])
        assert final['history'][index]['anchor_positions']==int(eligible.sum());anchors+=int(eligible.sum());criticaltotal+=int(mask.sum())
    assert criticaltotal==43992
    del final,transition,initial;gc.collect();panels={};dev=io.load_json(d/'dev_features.json');retained=io.load_json(Path(p['parent_data'])/'retained_English.json')
    specs=[('new_baseline',dev),('OWT',io.load_json(p['retention_OWT'])),('retained_English',retained),('prior_English',io.load_json(Path(p['parent_data'])/'dev_features.json')),
        ('long',io.load_json(Path(p['long_data'])/'features.json')[-2:]),('new_expression',dev),('fresh_spot',[retained[4],retained[32],dev[0],dev[1]])]
    totalraw=0
    for label,source in specs:
        records=io.load_json(run/(label+'_reconstructions.json'));paths=[run/f'raw_{label}_{i:02d}.pt' for i in range(len(source))]
        for file,r in zip(paths,records):assert support.sha(file)==r['raw_sha256']
        panels[label]=raw_panel(run,source,records,paths,None if label=='long' else 16385);totalraw+=len(paths)
        assert abs(panels[label]['NLL']-result['evaluations'][label]['NLL'])<4*torch.finfo(torch.float32).eps
        assert panels[label]['byte_EOS_accuracy']==result['evaluations'][label]['byte_EOS_accuracy']
    assert totalraw==154
    assert all(v['commit_relative_L2']==0 and v['all_argmax_agreement']==1 and v['same_decoded_text'] for v in io.load_json(run/'fresh_cache_parity.json'))
    assert not torch.cuda.is_initialized()
    report=dict(status='ROOT_V72_SAVED_REVIEW_PASS',checkpoints=checkpoints,full_parent_warm_exact=True,fresh7Adam=True,old30Adam_preserved=True,
        all_nonhead_tensors_encoder_binding_prior_reasoner_identical=True,all7_final_readout_tensors_changed=True,cache640_pairs_SHA_source_epoch_target_mask_verified=True,
        teacher_anchor_positions=anchors,critical_positions=criticaltotal,teacher_probability_validation='saved p(gold)/correct flags range+shape+TRAIN-only eligibility; full teacher logits not saved, no independent teacher argmax recomputation',
        five_fullstates_all_Adam_scaler_ALLRNG_phase_order_cursor_exposure_callbudget_verified=True,all154_raw_SHA_targets_argmax_NLL_UTF8_FSM_verified=True,
        four_actual_fresh_fullpath_commit_argmax_output_equal=True,panels=panels,extra_model_calls=0,CUDA_initialized=False,Exact_entries=0,copy_bytes=0,denoise_calls=0,
        limitations=p['limitations'])
    io.atomic_json(run/'root_saved_review.json',report)
    print(json.dumps(dict(status=report['status'],anchors=anchors,checkpoints=checkpoints,panels={k:{a:b for a,b in v.items() if a!='details'} for k,v in panels.items()})))
if __name__=='__main__':main()
