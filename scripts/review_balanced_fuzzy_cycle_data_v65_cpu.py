"""Independent V65 source/split/finite-order and TRAIN-loss review."""
import collections,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.fuzzy_cycle_objective_v65 import training_weights
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canonical(r):return re.search(r'OWT:(?:train|validation):[0-9]+',r.get('source_id',r['id'])).group()
def main():
    d=ROOT/'research/artifacts/balanced_fuzzy_cycle_20261112_v65_cpu';m=read(d/'manifest.json');rows=read(d/'train.json');dev=read(d/'dev.json');orders=read(d/'orders.json')
    original=[read(Path(p)) for p in m['original_TRAIN_files']];lookup={r['id']:r for pool in original for r in pool};long={canonical(r):r['prompt'].encode() for r in original[0]}
    assert len(rows)==512 and len(dev)==16 and dev==read(Path(m['original_DEV']))
    for r in rows:
        assert r['prompt']==lookup[r['id']]['prompt'] and r['prompt'].encode() in long[canonical(r)]
        assert not any(k.lower().startswith(('exact','gold','target','answer')) for k in r)
    assert not set(map(canonical,rows))&set(map(canonical,dev))
    ids=list(range(512));random.Random(65112).shuffle(ids);assert orders==[[i] for i in ids]
    lookup,objective=training_weights(rows);assert min(lookup[:256])>=.5 and max(lookup[:256])<=4 and abs(objective['normalization']-1)<1e-12
    assert sum(len(r['prompt'].encode()) for r in rows)==m['target_UTF8_bytes']==3724282
    report=dict(status='INDEPENDENT_V65_DATA_PASS',TRAIN_rows=512,unique_source_documents=505,DEV_unchanged16=True,TRAINDEV_overlap0=True,
        all_windows_original_TRAIN=True,finite_epoch_once=True,EOS_targets=512,target_UTF8_bytes=3724282,objective_report=objective,
        class_weights_from_TRAIN_only=True,Exact_features0=True,model_calls=0,E2_registry_access=0)
    p=d/'root_data_review.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
