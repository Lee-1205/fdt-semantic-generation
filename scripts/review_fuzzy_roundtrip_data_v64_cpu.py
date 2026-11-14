"""Independent provenance/split/exposure review; no model or held registry."""
import collections,hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canonical(r):return re.search(r'OWT:(?:train|validation):[0-9]+',r.get('source_id',r['id'])).group()
def main():
    data=ROOT/'research/artifacts/fuzzy_roundtrip_20261112_v64_cpu';m=read(data/'manifest.json')
    train,dev,orders=(read(data/(name+'.json')) for name in ('train','dev','orders'))
    originals=[read(Path(p)) for p in m['original_TRAIN_files']]
    lookup={r['id']:r for pool in originals for r in pool}
    assert len(train)==128 and len(dev)==16 and dev==read(Path(m['original_DEV']))
    for row in train:
        old=lookup[row['id']]
        assert row['prompt']==old['prompt'] and canonical(row)==canonical(old)
        assert row['prompt'].encode().decode()==row['prompt']
        assert not any(k.lower().startswith(('exact','gold','target','answer')) for k in row)
    old_long={canonical(r):r['prompt'].encode() for r in originals[0]}
    for row in train:
        # Authored windows must be actual contiguous bytes of their TRAIN source.
        assert row['prompt'].encode() in old_long[canonical(row)]
    assert not set(map(canonical,train))&set(map(canonical,dev))
    assert len(orders)==256
    for epoch in range(2):
        expected=list(range(128));random.Random(64112+epoch).shuffle(expected)
        assert orders[epoch*128:(epoch+1)*128]==[[i] for i in expected]
    counts=collections.Counter(r['length_group'] for r in train)
    assert sorted(counts.values())==[16,16,16,16,32,32]
    size=sum(len(r['prompt'].encode()) for r in train)
    assert size==558504 and max(len(r['prompt'].encode()) for r in train)==16384
    result=dict(status='INDEPENDENT_V64_DATA_PASS',rows=128,unique_source_documents=len(set(map(canonical,train))),
        DEV_unchanged16=True,TRAINDEV_canonical_overlap0=True,all_TRAIN_windows_in_original_source=True,
        no_Exact_or_gold_feature_fields=True,finite_epochs=2,row_exposure=256,target_UTF8_bytes=size*2,
        EOS_targets=256,group_counts=dict(counts),model_calls=0,E2_registry_access=0,
        limitations='previously observed DEV; 124 source documents and correlated windows; EOS denotes supplied input-window end')
    out=data/'root_data_review.json'
    if out.exists():raise FileExistsError('immutable independent review')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
