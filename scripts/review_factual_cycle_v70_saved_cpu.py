"""Saved-only fullstate and true decoded-output review of ONE V70 trial."""
import collections,gc,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import OBJECTIVE,training_weights

def verify_output(z,ids,cap):
    a=z.reshape(-1,258).numpy();n=min(len(a),cap);guard_limit=len(a)-32;need=0;lo=128;hi=191;out=[]
    for i,chosen in enumerate(ids):
        if i>=n:assert chosen==257;continue
        legal=np.zeros(258,dtype=bool)
        if need:legal[lo:hi+1]=True
        else:
            legal[:128]=True;legal[256]=i>=guard_limit
            if n-i>=2:legal[194:224]=True
            if n-i>=3:legal[224:240]=True
            if n-i>=4:legal[240:245]=True
        best=int(np.where(legal,a[i],-np.inf).argmax());assert chosen==best,(i,chosen,best)
        if chosen==256:break
        out.append(chosen)
        if need:need-=1
        else:need=1 if 194<=chosen<224 else 2 if 224<=chosen<240 else 3 if 240<=chosen<245 else 0
        lo=160 if chosen==224 else 144 if chosen==240 else 128;hi=159 if chosen==237 else 143 if chosen==244 else 191
    assert not need
    return bytes(out).decode('utf8')

def raw_panel(run,rows,records,paths,cap):
    n=correct=NLL=decoded_errors=0;byclass=collections.Counter();details=[]
    for i,(row,r,path) in enumerate(zip(rows,records,paths)):
        a=torch.load(path,weights_only=True,map_location='cpu');z,y=a['logits'],a['targets'];b=row['prompt'].encode()
        expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256;assert torch.equal(y.flatten(),expected)
        assert a['Exact_entries']==0 and a['fuzzy_commit'].count_nonzero()
        assert verify_output(z,r['ids'],cap if cap else z.numel()//258)==r['output']
        pred=z.argmax(-1);active=y!=257;count=int(active.sum());ok=int(((pred==y)&active).sum())
        ce=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'))
        assert ok==r['correct'] and math.isclose(ce,r['NLL_sum'],rel_tol=1e-6,abs_tol=4*torch.finfo(torch.float32).eps*count+1e-5)
        out=r['output'].encode();errors=sum(x!=yy for x,yy in zip(b,out))+abs(len(b)-len(out));decoded_errors+=errors
        changed=[]
        for j,(x,yy) in enumerate(zip(b,out)):
            if x!=yy:
                kind='digit' if 48<=x<=57 else 'upper' if 65<=x<=90 else 'lower' if 97<=x<=122 else 'other';byclass[kind]+=1
                if len(changed)<32:changed.append(dict(byte=j,source=b[max(0,j-32):j+42].decode('utf8',errors='replace'),output=out[max(0,j-32):j+42].decode('utf8',errors='replace')))
        n+=count;correct+=ok;NLL+=ce;details.append(dict(id=row['id'],source_bytes=len(b),output_bytes=len(out),decoded_errors=errors,exact=row['prompt']==r['output'],changes=changed))
    return dict(rows=len(rows),NLL=NLL/n,byte_EOS_accuracy=correct/n,raw_errors=n-correct,decoded_errors=decoded_errors,decoded_error_classes=dict(byclass),
        exact_auxiliary=sum(r['prompt']==s['output'] for r,s in zip(rows,records)),details=details)

def main():
    torch.set_num_threads(4);run=ROOT/'runs/factual_cycle_20261112_v70';p=io.load_json(run/'plan.json');data=Path(p['data']);result=io.load_json(run/'result.json')
    native=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert native['all_known_pids_gone'] and native['Python_native_count']==0 and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone']
    for name,digest in p['files'].items():assert support.sha(name)==digest,name
    train=io.load_json(data/'train_features.json');targets=io.load_json(data/'train_targets.json');orders=io.load_json(data/'orders.json');target_map={t['id']:t for t in targets}
    expected_weights,obj=training_weights(train);assert obj==p['objective_report']
    initial=torch.load(run/'initial/checkpoints/state_0000.pt',weights_only=True,map_location='cpu');parent=torch.load(p['cycle_source'],weights_only=True,map_location='cpu')
    assert support.state_sha(parent)==p['cycle_algorithm_sha'] and len(parent['optimizer']['state'])==30
    assert support.state_sha(initial['model'])==support.state_sha(parent['model']) and not initial['optimizer']['state'];del parent;gc.collect()
    names=sum(initial['optimizer_names'],[]);frozen={k:v for k,v in initial['model'].items() if k not in names};frozen_digest=support.state_sha(frozen);states=[];final_info=None;changes={}
    for step in (0,1,256,512):
        file=run/('initial' if step==0 else 'model')/'checkpoints'/f'state_{step:04d}.pt';meta=io.load_json(file.with_suffix('.json'));assert support.sha(file)==meta['sha256']
        s=torch.load(file,weights_only=True,map_location='cpu');assert support.state_sha(s)==meta['algorithm_sha256']
        assert s['schema']=='fdt_live_fuzzy_roundtrip_v70_fullstate' and s['objective']==OBJECTIVE and s['objective_report']==obj and s['plan_sha256']==support.sha(run/'plan.json')
        assert torch.equal(s['symbol_weights'],torch.tensor(expected_weights)) and s['optimizer_names']==initial['optimizer_names'] and s['orders']==orders
        assert s['attempts']==s['updates']==s['cursor']==step and s['skips']==0 and [h['rows'] for h in s['history']]==orders[:step]
        selected=[train[a[0]] for a in orders[:step]];size=sum(len(r['prompt'].encode()) for r in selected)
        assert s['exposure']==dict(rows=step,target_UTF8_bytes=size,active_symbols=size+step)
        assert s['parent_reference']==initial['parent_reference'] and not s['gradients'] and s['Exact_entries']==s['copy_bytes']==s['denoise_calls']==0
        assert support.state_sha({k:v for k,v in s['model'].items() if k not in names})==frozen_digest
        assert len(s['optimizer']['state'])==(30 if step else 0) and all(int(v['step'])==step for v in s['optimizer']['state'].values())
        assert s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==step and s['torch_rng'].numel() and len(s['cuda_rng'])==1 and s['numpy_rng']['keys'] and s['python_rng']
        for h in s['history']:
            row=train[h['rows'][0]];critical=sum(v['end']-v['start'] for v in target_map[row['id']]['critical_spans'])
            assert h['critical_positions']==critical and h['hard_k']==min(64,max(1,math.ceil(.01*h['bytes']))) and not h['AMP_skip']
            expected=h['weighted_byte_CE']+.1*h['EOS_CE']+.1*h['hard_body_CE']+.1*h['tail_false_EOS_UL']+.25*h['critical_CE']
            assert math.isclose(h['loss'],expected,rel_tol=1e-6,abs_tol=1e-6) and h['GPU_peak_bytes']<=12*1024**3
        if step==512:
            assert s['calls']==dict(encode=598,readout=598,stream_chunk=256,denoise=0) and s['long_block_work']==2304 and sum(h['critical_positions'] for h in s['history'])==38480
            for prefix in ('encoder.core.language.blocks.','encoder.core.value.','encoder.core.condition.','readout.'):
                keys=[k for k in names if k.startswith(prefix)];changes[prefix]=dict(active=len(keys),changed=sum(not torch.equal(s['model'][k],initial['model'][k]) for k in keys));assert changes[prefix]['changed']>0
            final_info=dict(SHA=meta['sha256'],algorithm_SHA=meta['algorithm_sha256'],bytes=meta['bytes'],training_GPU_peak_bytes=max(h['GPU_peak_bytes'] for h in s['history']))
        states.append(dict(step=step,SHA=meta['sha256'],Adam_states=len(s['optimizer']['state']),scaler=s['scaler']));del s;gc.collect()
    panels={};specs=[('new_English',io.load_json(data/'dev_features.json')),('final',io.load_json(p['retention_OWT'])),('retained_English',io.load_json(data/'retained_English.json'))]
    for label,rows in specs:
        records=io.load_json(run/(label+'_reconstructions.json'));score=raw_panel(run,rows,records,[run/f'raw_{label}_{i:02d}.pt' for i in range(len(rows))],16385)
        assert abs(score['NLL']-result[label]['NLL'])<4*torch.finfo(torch.float32).eps and score['byte_EOS_accuracy']==result[label]['byte_EOS_accuracy'];panels[label]=score
    rows=io.load_json(Path(p['long_data'])/'features.json')[-2:];records=io.load_json(run/'long_reconstructions.json')
    for i,r in enumerate(records):assert support.sha(run/f'raw_long_{i}.pt')==r['raw_sha256']
    panels['long']=raw_panel(run,rows,records,[run/f'raw_long_{i}.pt' for i in (0,1)],None)
    report=dict(status='ROOT_V70_SAVED_REVIEW_PASS',checkpoints=states,all_source_model_tensors_exact_warm=True,fresh30Adam=True,
        all_frozen_tensors_including_prior_identical=True,actual_encoder_decoder_changes=changes,all_Adam_scaler_ALLRNG_orders_cursor_exposure_criticalcounts_verified=True,
        all86raw_logits_targets_argmax_independent_UTF8_FSM_verified=True,panels=panels,final_state=final_info,Exact_entries=0,copy_bytes=0,denoise_calls=0,
        extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),limitations='meaning reading separate; similarity/byte counts not abstract semantics; observed retained dev and prospective correlated authored data')
    assert not torch.cuda.is_initialized();io.atomic_json(run/'root_saved_review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoints','panels')}));print(json.dumps({k:{a:b for a,b in v.items() if a!='details'} for k,v in panels.items()}))
if __name__=='__main__':main()
