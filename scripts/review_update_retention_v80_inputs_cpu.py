"""One finite TRAIN-only coverage and existing raw precision review; no NN."""
import collections,hashlib,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    began=time.monotonic();torch.set_num_threads(4)
    run=ROOT/'runs/last_linear_20261113_v79';p=io.load_json(run/'plan.json')
    out=ROOT/'research/artifacts/update_retention_20261113_v80_cpu_v1';out.mkdir(exist_ok=False)
    rows=io.load_json(Path(p['data'])/'train_features.json');targets=io.load_json(Path(p['data'])/'train_targets.json');orders=io.load_json(Path(p['balance_data'])/'orders.json')
    assert len(rows)==len(targets)==1024 and len(orders)==384
    counts=collections.Counter();roles=collections.defaultdict(collections.Counter);values=collections.defaultdict(set);alignment=collections.defaultdict(set);teacher=collections.defaultdict(collections.Counter)
    multiplicity=collections.Counter(i for g in orders for i in g)
    refs=torch.load(run/'model/runtime.pt',map_location='cpu',weights_only=True)['runtime']['cache_refs']
    for i,(row,t,ref) in enumerate(zip(rows,targets,refs)):
        if time.monotonic()-began>100:raise TimeoutError('fixed CPU input review100sec')
        assert set(row)=={'id','prompt'} and t['id']==row['id']==ref['source_id'] and support.sha(ref['targets'])==ref['targets_SHA']
        packet=torch.load(ref['targets'],map_location='cpu',weights_only=True)
        b=row['prompt'].encode();mode=t.get('mode','OWT');counts[mode+'_unique_rows']+=1;counts[mode+'_head_rows']+=multiplicity[i];counts[mode+'_unique_bytes']+=len(b);counts[mode+'_head_bytes']+=len(b)*multiplicity[i]
        counts[mode+'_critical_bytes']+=int(packet['critical_mask'].sum())
        body=packet['labels']<256;correct=packet['teacher_correct'];weak=packet['teacher_margin']<=3
        teacher[mode].update(correct_critical=int((body&packet['critical_mask']&correct).sum()),weak_correct_critical=int((body&packet['critical_mask']&correct&weak).sum()),wrong_critical=int((body&packet['critical_mask']&~correct).sum()))
        if mode=='OWT':continue
        for key in ('A','B'):
            s=t[key+'_span'];assert b[s['start']:s['end']].decode()==s['text']
            sign='+' if s['text'].startswith('+') else '-' if s['text'].startswith('-') else 'implicit_positive'
            unit=t['unit'];roles[key+'|'+unit][sign]+=1;values[key+'|'+unit].add(t[key]);alignment[key+'|'+unit].add(s['start']%16)
            start,end=s['start'],s['end'];gold=packet['labels'].flatten()[start:end];okay=correct.flatten()[start:end];m=packet['teacher_margin'].flatten()[start:end]
            teacher[key+'|'+unit].update(symbols=len(gold),wrong=int((~okay).sum()),weak_correct=int((okay&(m<=3)).sum()))
    # Exhaustive file scan reads stored values only. Conversion is accepted only
    # with exact bitwise roundtrip, never just unchanged argmax or small L2.
    raw_stats=collections.Counter();failures=[]
    for f in sorted(run.glob('raw*.pt')):
        if time.monotonic()-began>100:raise TimeoutError('fixed CPU raw precision review100sec')
        packet=torch.load(f,map_location='cpu',weights_only=True);raw_stats['files']+=1
        candidates=([('logits',packet['logits'])] if 'logits' in packet else [(k+'.logits',v['logits']) for k,v in packet.items() if isinstance(v,dict) and 'logits' in v])
        for key,z in candidates:
            assert z.dtype==torch.float32 and torch.isfinite(z).all()
            mismatch=int((z.to(torch.bfloat16).float().view(torch.int32)!=z.view(torch.int32)).sum());raw_stats['logit_tensors']+=1;raw_stats['float32_logit_bytes']+=z.numel()*4
            raw_stats['BF16_exact_tensors']+=int(mismatch==0);raw_stats['BF16_nonexact_tensors']+=int(mismatch!=0);raw_stats['nonexact_float32_values']+=mismatch
            if mismatch:failures.append(dict(file=f.name,key=key,nonexact_values=mismatch,values=z.numel(),max_abs_difference=float((z-z.to(torch.bfloat16).float()).abs().max())))
    assert raw_stats['files']==471
    report=dict(status='ROOT_V80_TRAIN_COVERAGE_AND_RAW_PRECISION_CPU_COMPLETE',rows=dict(counts),A_B_sign_cells={k:dict(v) for k,v in roles.items()},
        numeric_integer_parts={k:dict(min=min(v),max=max(v),unique=sorted(v),single_digit_parts=sum(0<=a<=9 for a in v)) for k,v in values.items()},
        field_start_roles={k:sorted(v) for k,v in alignment.items()},initial_V75_teacher_by_source_or_field={k:dict(v) for k,v in teacher.items()},
        raw_precision=dict(raw_stats),nonexact_raw=failures,BF16_blind_compression_rejected=bool(failures),
        inferred_data_gap='Authored unsigned is only steel-rings count, not unsigned kg/temperature/distance; signed secondary B is always +. No authored one-digit unsigned quantity. EOS/start-role balance does not balance these content conditions.',
        causal_limit='Measured TRAIN coverage and saved-value precision, not proof of all generalization errors or actual task-gradient conflict. Existing OWT may contain one-digit numbers; claim restricted to authored A/B quantities.',
        source_files_SHA={str(f):support.sha(f) for f in [Path(p['data'])/'train_features.json',Path(p['data'])/'train_targets.json',Path(p['balance_data'])/'orders.json',run/'plan.json',Path(__file__)]},
        actual_model_forwards=0,actual_optimizer_updates=0,CUDA_initialized=torch.cuda.is_initialized(),E2_final_registry_access=0,new_TRAIN_rows=0,new_GPU=False,seconds=time.monotonic()-began)
    io.atomic_json(out/'root_input_review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_files_SHA','nonexact_raw','field_start_roles','initial_V75_teacher_by_source_or_field')}))


if __name__=='__main__':main()
