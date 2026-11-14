"""Same-input OWT reconstruction, source-document split preserved."""
import hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def canonical(row):return re.search(r'OWT:(?:train|validation):[0-9]+',row.get('source_id',row['id'])).group()


def main():
    old=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu'
    windows=ROOT/'research/artifacts/semantic_binding_20261111_v59_cpu/OWT_train_reconstruction_windows.json'
    out=ROOT/'research/artifacts/fuzzy_roundtrip_20261112_v64_cpu';out.mkdir(exist_ok=True)
    if (out/'train.json').exists():raise FileExistsError('immutable roundtrip data')
    train=json.loads((old/'train.json').read_text(encoding='utf-8-sig'));dev=json.loads((old/'dev.json').read_text(encoding='utf-8-sig'))
    short=json.loads(windows.read_text(encoding='utf-8-sig'));selected=[]
    choose=lambda rows:sorted(rows,key=lambda r:hashlib.sha256(('v64:'+r['id']).encode()).hexdigest())[:32]
    for label in ('short','mid'):
        pool=[r for r in short if r['selection']==label];assert len(pool)>=32
        for r in choose(pool):selected.append(dict(r,actual_bytes=len(r['prompt'].encode()),length_group=label))
    for low,high in ((769,2048),(2049,8192),(8193,16383),(16384,16384)):
        pool=sorted([r for r in train if low<=len(r['prompt'].encode())<=high],key=lambda r:hashlib.sha256(('v64:'+r['id']).encode()).hexdigest())[:16]
        assert len(pool)==16
        for r in pool:selected.append(dict(r,length_group=f'{low}-{high}',source_id=r['id']))
    assert len(selected)==128 and not set(map(canonical,selected))&set(map(canonical,dev))
    orders=[]
    for epoch in range(2):
        ids=list(range(128));random.Random(64112+epoch).shuffle(ids);orders.extend([[i] for i in ids])
    for name,value in (('train',selected),('dev',dev),('orders',orders)):
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    manifest=dict(schema='fuzzy_roundtrip_v64_data',TRAIN_rows=128,DEV_rows=16,epochs=2,attempts=256,
        TRAIN_unique_source_documents=len(set(map(canonical,selected))),TRAINDEV_source_overlap=0,
        min_TRAIN_bytes=min(len(r['prompt'].encode()) for r in selected),max_TRAIN_bytes=max(len(r['prompt'].encode()) for r in selected),
        UTF8_input_target_bytes_per_epoch=sum(len(r['prompt'].encode()) for r in selected),
        original_TRAIN_files=[str(old/'train.json'),str(windows)],original_DEV=str(old/'dev.json'),
        contract='SAME supplied UTF8 input window -> fuzzy Commit -> SAME full window + predicted EOS; no semantic-role target, continuation, target prefix or Exact content',
        Exact_policy='DORMANT_UNTIL_FUTURE_EXPLICIT_INTERNAL_COPY_REQUEST',new_downloads=0,E2_access=0,
        limitations='existing observed local OWT split; correlated source windows; input-window EOF not full original-document EOF; reconstruction alone does not uniquely establish abstract semantics')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');print(json.dumps(manifest))

if __name__=='__main__':main()
