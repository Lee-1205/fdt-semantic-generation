"""Complete numerical-expression TRAIN coverage for prospective fixed encoder."""
import hashlib,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from research.frozen_verbalizer_objective_v72 import complete_expression_spans

def extra(count,names,seed):
    rng=random.Random(seed);features=[];targets=[]
    for i in range(count):
        actor,recipient=rng.sample(names,2);hour=rng.choice([3,8,11,14,17,21]);minute=rng.choice([7,17,26,32,41,52]);clock=f'{hour:02d}:{minute:02d}'
        q=rng.randint(3,97);decimal=f'{rng.randint(1,29)}.{rng.randint(1,99):02d}';sign=rng.choice(['-','+']);currency=f'${rng.randint(1,999)},{rng.randint(0,999):03d}.{rng.randint(0,99):02d}'
        variants=[
            f'At {clock}, {actor} logged {sign}{decimal} degrees C while {recipient} weighed {q}.{rng.randint(1,9)} kg of flour.',
            f'{actor} paid {currency} to {recipient} on 2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}; the refund was -${decimal}.',
            f'{recipient} received {q}% of the mixture from {actor}, and the remaining sample weighed {decimal} kg at {clock}.',
            f'Between {q} and {q+3} minutes after {clock}, {actor} moved a 3-7 kg package to {recipient} without changing its label.',
        ]
        text=variants[i%4]
        if i%2:text+=' '+rng.choice(['The final reading was not an estimate.','The order of the measurements was recorded.','Neither person discarded the remaining sample.'])
        rid='expr72:'+hashlib.sha256(text.encode()).hexdigest()[:24]
        namesp=[dict(start=m.start(),end=m.end(),kind='name',text=m.group()) for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b',text)]
        features.append(dict(id=rid,prompt=text));targets.append(dict(id=rid,critical_spans=complete_expression_spans(text,namesp),source_kind='new_authored_complete_expressions'))
    return features,targets

def main():
    out=ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu';out.mkdir(exist_ok=False)
    old=ROOT/'research/artifacts/factual_cycle_20261112_v70_cpu_v2';train=io.load_json(old/'train_features.json');oldtargets=io.load_json(old/'train_targets.json');meta=io.load_json(old/'manifest.json')
    targets=[dict(id=r['id'],critical_spans=complete_expression_spans(r['prompt'],t['critical_spans']),source_kind='preserved_V70_TRAIN') for r,t in zip(train,oldtargets)]
    added,newtargets=extra(128,meta['TRAIN_names'],72112);train+=added;targets+=newtargets
    dev,devtargets=extra(32,meta['DEV_names'],72113)
    assert len(train)==len(targets)==640 and len({r['prompt'] for r in train})==640
    assert not {r['prompt'] for r in dev}&{r['prompt'] for r in train}
    forbidden=io.load_json(ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json')+io.load_json(ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu/features.json')
    assert not any(f['prompt'] in t['prompt'] for f in forbidden for t in train)
    order=list(range(640));random.Random(72112).shuffle(order)
    manifest=dict(schema='frozen_verbalizer_v72_CPU_data',TRAIN_rows=640,preserved_V70_TRAIN=512,new_complete_expression_rows=128,new_DEV_rows=32,
        finite_epochs=1,proposed_head_attempts=640,target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in train),
        critical_bytes=sum(s['end']-s['start'] for t in targets for s in t['critical_spans']),max_input_bytes=max(len(r['prompt'].encode()) for r in train),
        features=['id','prompt'],gold_spans_teacher_correct_filter_TRAIN_only=True,NN_inference_feature_change=False,
        intended_boundary='freeze encoder and upstream role_projection; train only seven final readout tensors on actual bound Commit; dynamic input memories retained',
        teacher_anchor='initial decoder p(gold) and correct flag on TRAIN only, excluded on critical positions; no filtering of evaluation',
        actual_runner_plan_optimizer_GPU=False,E2_registry_access=0,new_downloads=0,
        limitations='prospective loss/data only; syntactic numeric-span heuristic is not semantic truth/NER; family overlap and observed design; no head training or capability result yet')
    for name,value in [('train_features',train),('train_targets',targets),('dev_features',dev),('dev_targets',devtargets),('orders',[[i] for i in order]),('manifest',manifest)]:io.atomic_json(out/(name+'.json'),value)
    print(json.dumps(manifest))
if __name__=='__main__':main()
