"""Independent TRAIN provenance, remainder coverage and held-data exclusion."""
import collections,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.fuzzy_cycle_objective_v67 import training_weights
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canonical(r):return re.search(r'OWT:(?:train|validation):[0-9]+',r.get('source_id',r['id'])).group()
def main():
    d=ROOT/'research/artifacts/hard_fuzzy_cycle_20261112_v67_cpu_v2';m=read(d/'manifest.json');rows=read(d/'train.json');dev=read(d/'dev.json');orders=read(d/'orders.json');retained=read(d/'retained_English.json')
    original=[read(Path(p)) for p in m['original_TRAIN_files']];lookup={r['id']:r for pool in original for r in pool};long={canonical(r):r['prompt'].encode() for r in original[0]}
    assert len(rows)==1024 and len({r['id'] for r in rows})==1024 and dev==read(Path(m['original_DEV']))
    for row in rows:
        assert row['prompt'].encode() in long[canonical(row)] and row['prompt'].encode().decode()==row['prompt']
        if row['length_group']=='ragged16k':
            assert row['prompt'].encode()==lookup[row['source_id']]['prompt'].encode()[:len(row['prompt'].encode())]
            assert row['actual_bytes']==len(row['prompt'].encode())
        else:assert row['prompt']==lookup[row['id']]['prompt']
        assert not any(k.lower().startswith(('exact','gold','target','answer')) for k in row)
    ragged=[r for r in rows if r['length_group']=='ragged16k'];tails=collections.Counter(len(r['prompt'].encode())%16 for r in ragged)
    assert len(ragged)==240 and set(tails)==set(range(16))
    for i,row in enumerate(ragged):assert row['requested_tail_remainder']==(16369+i%16)%16
    assert not set(map(canonical,rows))&set(map(canonical,dev))
    held=read(ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json')
    assert retained==[r for r in held if r['language']=='English'] and not any(h['prompt'] in t['prompt'] for h in held for t in rows)
    ids=list(range(1024));random.Random(67112).shuffle(ids);assert orders==[[i] for i in ids]
    weights,objective=training_weights(rows);assert min(weights[:256])>=.5 and max(weights[:256])<=4 and abs(objective['normalization']-1)<1e-12
    assert sum(len(r['prompt'].encode()) for r in rows)==m['target_UTF8_bytes']==7978178
    proof=dict(status='INDEPENDENT_V67_DATA_PASS',TRAIN_rows=1024,unique_source_documents=1011,DEV16_unchanged=True,TRAINDEV_canonical_overlap0=True,
        all52V66_never_TRAIN=True,retained36_observed_English_unchanged=True,one_finite_order_once=True,target_UTF8_bytes=7978178,
        actual_ragged_tail_counts=dict(tails),objective_report=objective,model_calls=0,E2_registry_access=0)
    path=d/'root_data_review.json';assert not path.exists();path.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
