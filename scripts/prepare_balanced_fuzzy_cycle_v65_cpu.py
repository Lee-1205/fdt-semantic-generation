"""More existing TRAIN windows; same held rows, no Exact or semantic labels."""
import hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canonical(r):return re.search(r'OWT:(?:train|validation):[0-9]+',r.get('source_id',r['id'])).group()
def main():
    old=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu';windows=ROOT/'research/artifacts/semantic_binding_20261111_v59_cpu/OWT_train_reconstruction_windows.json'
    out=ROOT/'research/artifacts/balanced_fuzzy_cycle_20261112_v65_cpu';out.mkdir(exist_ok=False)
    long,short,dev=read(old/'train.json'),read(windows),read(old/'dev.json');rows=[]
    choose=lambda pool,n:sorted(pool,key=lambda r:hashlib.sha256(('v65:'+r['id']).encode()).hexdigest())[:n]
    for label in ('short','mid'):
        rows.extend(dict(r,length_group=label) for r in choose([r for r in short if r['selection']==label],32))
    for low,high in ((769,2048),(2049,8192),(8193,16383),(16384,16384)):
        pool=choose([r for r in long if low<=len(r['prompt'].encode())<=high],112);assert len(pool)==112
        rows.extend(dict(r,length_group=f'{low}-{high}',source_id=r['id']) for r in pool)
    assert len(rows)==512 and not set(map(canonical,rows))&set(map(canonical,dev))
    ids=list(range(512));random.Random(65112).shuffle(ids);orders=[[i] for i in ids]
    manifest=dict(schema='balanced_fuzzy_cycle_v65_data',TRAIN_rows=512,DEV_rows=16,epochs=1,attempts=512,
        TRAIN_unique_source_documents=len(set(map(canonical,rows))),TRAINDEV_source_overlap=0,
        target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),EOS_targets=512,max_bytes=16384,
        original_TRAIN_files=[str(old/'train.json'),str(windows)],original_DEV=str(old/'dev.json'),
        contract='same input -> fuzzy Commit -> same window; Exact dormant; target only loss; no target prefix, H or output length at language boundary',
        new_downloads=0,E2_registry_access=0,limitations='observed DEV; correlated source windows; input-window EOS; no abstract meaning uniqueness proof')
    for name,value in (('train',rows),('dev',dev),('orders',orders),('manifest',manifest)):
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest))
if __name__=='__main__':main()
