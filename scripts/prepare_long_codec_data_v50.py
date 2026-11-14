"""Existing OpenWebText document splits, finite mixed-length reconstruction windows."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/artifacts/long_codec_20261111_v50'
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('immutable data already exists')
    corpus=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final'
    documents=json.loads((corpus/'documents.json').read_text(encoding='utf-8'))
    lengths=[128,256,512,1024,2048,4096,8192,16384]
    counts=[64,64,64,64,64,64,32,16]
    result={};used={}
    for split in ('train','dev'):
        rows=[];seen=set()
        for length,count in zip(lengths,counts if split=='train' else [2]*8):
            candidates=sorted(documents[split],key=lambda d:hashlib.sha256((str(length)+d['id']).encode()).hexdigest())
            for doc in candidates:
                raw=doc['text'].encode()
                if len(raw)<length or doc['id'] in seen:continue
                text=raw[:length].decode('utf-8',errors='ignore')
                rows.append(dict(id=f"{split}:{length}:{doc['id']}",prompt=text,document_id=doc['id'],
                    requested_bytes=length,actual_bytes=len(text.encode()),source_byte_start=0,
                    source_byte_end=len(text.encode()),input_window_EOS=True,document_EOF=False,
                    normalized_document_sha=doc['normalized_sha256']))
                seen.add(doc['id']);count-=1
                if count==0:break
            if count:raise ValueError('not enough distinct original '+split+' documents')
        result[split]=rows;used[split]=seen
    if used['train']&used['dev']:raise ValueError('document split overlap')
    if {r['prompt'] for r in result['train']}&{r['prompt'] for r in result['dev']}:
        raise ValueError('literal train/dev overlap')
    orders=[]
    for epoch in range(4):
        order=list(range(len(result['train'])))
        if epoch:random.Random(50000+epoch).shuffle(order)
        orders.extend([[i] for i in order])
    manifest=dict(capacity=16384,chunk_bytes=16,lengths=lengths,train_counts=counts,train_rows=len(result['train']),
        dev_rows=len(result['dev']),epochs=4,attempts=len(orders),batch=1,
        target_bytes_per_epoch=sum(r['actual_bytes'] for r in result['train']),
        features='id,prompt only at encoder; metadata/targets offline',source=str(corpus),
        corpus_scope='same approved local OWT source/split; no download; no E2',
        boundary='UTF8-safe input window; EOS denotes that input, not full document or sentence boundary',
        raw_indices=[0,4,8,12],raw_max_symbols=16385,copy_bytes=0,denoise_calls=0)
    OUT.mkdir()
    for name,rows in result.items():write(name+'.json',rows)
    write('orders.json',orders);write('manifest.json',manifest)
    print(json.dumps(manifest))
if __name__=='__main__':main()
