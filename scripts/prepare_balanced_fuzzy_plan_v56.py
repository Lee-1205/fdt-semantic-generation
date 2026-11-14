"""TRAIN-only class counts, fixed new finite budget, unchanged data and Commit."""
import ast,collections,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_balanced_fuzzy_20261111_v56 import validate,SOURCE_SHA,PARENT_SHA

def main():
    artifact=ROOT/'research/artifacts/balanced_fuzzy_20261111_v56_cpu';path=artifact/'plan_unsealed.json'
    if path.exists():raise FileExistsError('new immutable plan')
    old_run=ROOT/'runs/ordered_fuzzy_20261111_v55';old=json.loads((old_run/'plan.json').read_text(encoding='utf-8-sig'))
    source=Path(old['source']);head=old_run/'model/checkpoints/state_3624.pt'
    train=json.loads((Path(old['data'])/'train.json').read_text(encoding='utf-8-sig'));counts=collections.Counter()
    def cls(i):return 'digit' if 48<=i<=57 else 'upper' if 65<=i<=90 else 'lower' if 97<=i<=122 else 'other'
    for row in train:counts.update(cls(i) for i in row['prompt'].encode())
    total=sum(counts.values());weights={k:math.sqrt(total/(4*v)) for k,v in counts.items()}
    norm=sum(counts[k]*v for k,v in weights.items())/total;weights={k:v/norm for k,v in weights.items()}
    lookup=[weights[cls(i)] for i in range(256)]+[1.,0.]
    pins={k:v for k,v in old['files'].items() if Path(k).suffix=='.py'}
    paths=[ROOT/'scripts/balanced_fuzzy_engine_v56.py',ROOT/'scripts/run_balanced_fuzzy_20261111_v56.py',
        ROOT/'scripts/supervise_balanced_fuzzy_v56.py',ROOT/'research/codec_commit_stop_v56.py',
        ROOT/'scripts/test_balanced_fuzzy_v56_cpu.py',Path(__file__),artifact/'root_CPU_contract.json',
        old_run/'root_saved_review.json',old_run/'root_error_localization.json',old_run/'root_commit_stop_candidate.json',
        source,head,Path(old['data'])/'train.json',Path(old['data'])/'dev.json']
    paths.extend(old_run/name for name in ('train_features.pt','train_targets.pt','dev_features.pt','dev_targets.pt'))
    for file in paths:pins[str(file)]=support.sha(file)
    for name,digest in pins.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    output=ROOT/'runs/balanced_fuzzy_20261111_v56'
    p=dict(schema='fdt_balanced_fuzzy_v56',authorized=False,root=str(ROOT),output=str(output),data=old['data'],
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=PARENT_SHA,
        files=pins,capacity=16384,attempts=3624,epochs=6,batch=256,amp_dtype='bfloat16',
        wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=3800,max_output_bytes=512*1024**2,
        class_counts_TRAIN=dict(counts),class_weights_TRAIN=weights,class_weights=lookup,early_patch_multiplier=4,
        loss='normalized inverse-sqrt TRAIN byte-class CE times first32-position4, EOSweight1 PAD0',
        class_weight_selection='from TRAIN only; not DEV-fit optimum',raw_indices=[0,4,8,12],
        cost={'encode':0,'readout':3640,'denoise':0},exposure=old['exposure'],source_encoder_optimizer_updates=0,
        recovery='V55 same8 active Adam/scaler/allRNG inherited exactly; NEW orders/cursor/clock. Source96Adam untouched in V53 parent; no trajectory rewind.',
        codec_stop='EOS only in final local slot/structural tail, no exact output byte count and no forced EOS',
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    for name in ('train_features','train_targets','dev_features','dev_targets'):p[name]=str(old_run/(name+'.pt'))
    path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(path,output)
    print(json.dumps(dict(actual_schema_PASS=True,class_counts_TRAIN=dict(counts),class_weights_TRAIN=weights,cost=p['cost'])))
if __name__=='__main__':main()
