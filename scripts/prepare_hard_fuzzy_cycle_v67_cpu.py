"""More TRAIN sources plus all sixteen high-capacity EOS tail offsets."""
import hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canonical(r):return re.search(r'OWT:(?:train|validation):[0-9]+',r.get('source_id',r['id'])).group()
def main():
    old=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu';windows=ROOT/'research/artifacts/semantic_binding_20261111_v59_cpu/OWT_train_reconstruction_windows.json'
    out=ROOT/'research/artifacts/hard_fuzzy_cycle_20261112_v67_cpu_v2';out.mkdir(exist_ok=False)
    long,short,dev=read(old/'train.json'),read(windows),read(old/'dev.json');rows=[]
    choose=lambda pool,n:sorted(pool,key=lambda r:hashlib.sha256(('v67:'+r['id']).encode()).hexdigest())[:n]
    for label in ('short','mid'):
        rows.extend(dict(r,length_group=label) for r in choose([r for r in short if r['selection']==label],32))
    for low,high in ((769,2048),(2049,8192),(8193,16383)):
        pool=choose([r for r in long if low<=len(r['prompt'].encode())<=high],240);assert len(pool)==240
        rows.extend(dict(r,length_group=f'{low}-{high}',source_id=r['id']) for r in pool)
    pool=choose([r for r in long if len(r['prompt'].encode())==16384],240);assert len(pool)==240
    for i,row in enumerate(pool):
        n=16369+i%16;raw=row['prompt'].encode()
        while True:
            try:text=raw[:n].decode('utf8');break
            except UnicodeDecodeError:n-=1
        rows.append(dict(row,id=row['id']+f':ragged67:{n}',source_id=row['id'],prompt=text,actual_bytes=n,length_group='ragged16k',requested_tail_remainder=(16369+i%16)%16))
    assert len(rows)==1024 and not set(map(canonical,rows))&set(map(canonical,dev))
    held=read(ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json');retained=[r for r in held if r['language']=='English']
    assert len(retained)==36 and not any(h['prompt'] in t['prompt'] for h in retained for t in rows)
    ids=list(range(1024));random.Random(67112).shuffle(ids);orders=[[i] for i in ids]
    manifest=dict(schema='hard_fuzzy_cycle_v67_data',TRAIN_rows=1024,unique_source_documents=len(set(map(canonical,rows))),epochs=1,attempts=1024,
        target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),EOS_targets=1024,min_bytes=min(len(r['prompt'].encode()) for r in rows),max_bytes=16384,
        original_TRAIN_files=[str(old/'train.json'),str(windows)],original_DEV=str(old/'dev.json'),retained_English_rows=36,
        retained_rows_are_observed_eval_not_blind=True,all_V66_52_remain_eval_only=True,new_downloads=0,E2_registry_access=0,
        contract='same input -> fuzzy Commit -> same full supplied window; Exact dormant; no target prefix/H/GT output length at reader',
        limitations='new loss plus wider TRAIN and ragged-tail coverage jointly change; observed DEV/English held reused as retention; no Korean training or abstract-meaning certificate')
    for name,value in (('train',rows),('dev',dev),('orders',orders),('retained_English',retained),('manifest',manifest)):
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest))
if __name__=='__main__':main()
