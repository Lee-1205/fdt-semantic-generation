"""One finite full TRAIN-document pass; existing held document split unchanged."""
import json,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu'
    if out.exists():raise FileExistsError('immutable artifact')
    source=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final/documents.json'
    docs=json.loads(source.read_text(encoding='utf-8'));rows=[]
    held={d['normalized_sha256'] for d in docs['dev']}
    for doc in docs['train']:
        if doc['normalized_sha256'] in held:raise ValueError('original document split overlap')
        raw=doc['text'].encode();text=raw[:16384].decode('utf-8',errors='ignore')
        rows.append(dict(id='broad:'+doc['id'],prompt=text,document_id=doc['id'],actual_bytes=len(text.encode()),
                         source_byte_start=0,source_byte_end=len(text.encode()),document_EOF=len(text.encode())==len(raw),input_window_EOS=True))
    assert len(rows)==8192 and len({r['document_id'] for r in rows})==8192
    dev=json.loads((ROOT/'research/artifacts/long_codec_20261111_v50/dev.json').read_text())
    if {r['prompt'] for r in rows}&{r['prompt'] for r in dev}:raise ValueError('literal split overlap')
    order=list(range(8192));random.Random(53111).shuffle(order);orders=[[i] for i in order]
    manifest=dict(source=str(source),train_documents=8192,train_rows=8192,epochs=1,attempts=8192,capacity=16384,
        target_UTF8_bytes=sum(r['actual_bytes'] for r in rows),context_lengths=sorted(r['actual_bytes'] for r in rows),
        document_EOF=sum(r['document_EOF'] for r in rows),seed=53111,
        contract='SAME UTF8 input-window reconstruction; no artificial full-document EOF claim; original held split retained',
        new_downloads=0,source_new_to_project=False,E2_final_access=False,copy_bytes=0,denoise_calls=0)
    out.mkdir()
    for name,value in [('train',rows),('dev',dev),('orders',orders),('manifest',manifest)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='context_lengths'}))
if __name__=='__main__':main()
