"""Append512 training documents, retain every V12 held row; bounded source fetch."""
import copy,hashlib,json,re,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import learning_io_v2 as io
PARENT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12'
OUT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v13'
REVISION='79d93d786212f7344586290adb811d4ae6a1762c'
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode('utf-8')).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    if OUT.exists():raise FileExistsError('new artifact only')
    parent=load(PARENT/'manifest.json')
    for name,h in parent['files'].items():
        if sha((PARENT/name).read_bytes())!=h:raise ValueError('old artifact changed')
    docs=copy.deepcopy(load(PARENT/'documents.json'));f=copy.deepcopy(load(PARENT/'features.json'));y=copy.deepcopy(load(PARENT/'targets.json'))
    seen_docs={d['normalized_sha256'] for rows in docs.values() for d in rows};seen_sentences={r['id'] for rows in f.values() for r in rows}
    OUT.mkdir();(OUT/'source_pages').mkdir();requests=[];downloaded=0;started=time.monotonic()
    def consume(page):
        for row in page['rows']:
            if len(docs['train'])==1024:return
            if row.get('truncated_cells'):continue
            text=row['row']['text'];dh=sha(' '.join(text.split()))
            if dh in seen_docs or int(dh[:8],16)%9==0:continue
            seen_docs.add(dh);candidates=[];local=set()
            for raw in re.split(r'(?<=[.!?])\s+',text):
                s=' '.join(raw.split());h=sha(s)
                if not 48<=len(s.encode('utf-8'))<=128 or not s or s[-1] not in '.!?' or '\ufffd' in s:continue
                if any(ord(c)<32 for c in s) or sum(c.isalpha() for c in s)<16 or h in seen_sentences or h in local:continue
                local.add(h);candidates.append((h,s))
            if len(candidates)<4:continue
            docid=f'OWT:train:{row["row_idx"]}'
            docs['train'].append({'id':docid,'row_idx':row['row_idx'],'normalized_sha256':dh,'raw_sha256':sha(text),'text':text})
            for h,s in sorted(candidates)[:4]:
                seen_sentences.add(h);f['train'].append({'id':h,'prompt':s})
                y['train'].append({'id':h,'document_id':docid,'target':s,'target_ids':list(s.encode('utf-8'))+[256],
                    'sentence_boundary':True,'document_EOF_claimed':False,'normalized_source_sentence_sha256':h})
    for source in sorted((PARENT/'source_pages').glob('*.json')):consume(load(source))
    reused_new=len(docs['train'])-512
    last=max(r['offset']+r['length'] for r in parent['requests'])
    for offset in range(last,last+2400,100):
        if len(docs['train'])==1024:break
        if time.monotonic()-started>120 or downloaded+2*1024**2>8*1024**2:raise RuntimeError('bounded source fetch cap')
        url=f'https://datasets-server.huggingface.co/rows?dataset=Skylion007%2Fopenwebtext&config=plain_text&split=train&offset={offset}&length=100'
        with urllib.request.urlopen(url,timeout=20) as response:
            b=response.read(2*1024**2+1);revision=response.headers.get('x-revision')
        if len(b)>2*1024**2 or revision!=REVISION:raise RuntimeError('source size/revision')
        downloaded+=len(b);(OUT/'source_pages'/f'rows_{offset:05d}.json').write_bytes(b)
        requests.append({'offset':offset,'length':100,'bytes':len(b),'sha256':sha(b),'revision':revision});consume(json.loads(b))
    if len(docs['train'])!=1024:raise RuntimeError('fixed expansion pool not filled')
    assert docs['dev']==load(PARENT/'documents.json')['dev'] and f['dev']==load(PARENT/'features.json')['dev'] and y['dev']==load(PARENT/'targets.json')['dev']
    assert not {d['normalized_sha256'] for d in docs['train']}&{d['normalized_sha256'] for d in docs['dev']}
    for n,v in [('documents.json',docs),('features.json',f),('targets.json',y)]:io.atomic_json(OUT/n,v)
    manifest={'schema':'owt_fuzzy_AE_expansion_v13_cpu','dataset':'Skylion007/openwebtext','revision':REVISION,
        'parent_artifact':str(PARENT),'parent_manifest_sha256':sha((PARENT/'manifest.json').read_bytes()),
        'documents':{k:len(v) for k,v in docs.items()},'rows':{k:len(v) for k,v in f.items()},'old_training_documents':512,'new_training_documents':512,
        'new_documents_recovered_from_old_pages':reused_new,'old_train_rows_unchanged_prefix':True,'dev_documents_and_rows_unchanged':True,
        'sentence_UTF8_bytes':{k:sum(len(r['prompt'].encode('utf-8')) for r in v) for k,v in f.items()},
        'old_train_sentence_UTF8_bytes':sum(len(r['prompt'].encode('utf-8')) for r in f['train'][:2048]),
        'new_train_sentence_UTF8_bytes':sum(len(r['prompt'].encode('utf-8')) for r in f['train'][2048:]),
        'exact_normalized_document_cross_split_overlap':0,'exact_selected_sentence_cross_split_overlap':0,
        'near_duplicate_checked_beyond_source_curation':False,'pretraining_overlap':'UNKNOWN','E2_final_accessed':False,
        'EOS_means_sentence_boundary_not_document_EOF':True,'fetch_bytes':downloaded,'fetch_cap_bytes':8*1024**2,
        'fetch_seconds':time.monotonic()-started,'requests':requests,
        'source_rights_note':parent['source_rights_note'],'modifications':parent['modifications']+';512newtrain-docs appended;old64helddoc unchanged',
        'files':{p.name:sha(p.read_bytes()) for p in OUT.iterdir() if p.is_file()}}
    io.atomic_json(OUT/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k in ('documents','rows','sentence_UTF8_bytes','new_documents_recovered_from_old_pages','fetch_bytes','fetch_seconds')}))
if __name__=='__main__':main()
