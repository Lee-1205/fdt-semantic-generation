"""Append2048 prospective training docs, retain2048old train and64held exactly."""
import copy,hashlib,json,re,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import learning_io_v2 as io
PARENT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v15'
INTERMEDIATE=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v13'
ORIGINAL=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12'
OUT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v20'
REVISION='79d93d786212f7344586290adb811d4ae6a1762c'
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode('utf-8')).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    if OUT.exists():raise FileExistsError('new immutable artifact only')
    for artifact in (PARENT,INTERMEDIATE,ORIGINAL):
        m=load(artifact/'manifest.json')
        if m['revision']!=REVISION:raise ValueError('fixed original revision')
        for name,h in m['files'].items():
            if sha((artifact/name).read_bytes())!=h:raise ValueError('parent artifact changed')
    parent=load(PARENT/'manifest.json');original=load(ORIGINAL/'manifest.json')
    docs=copy.deepcopy(load(PARENT/'documents.json'));features=copy.deepcopy(load(PARENT/'features.json'));targets=copy.deepcopy(load(PARENT/'targets.json'))
    seen_docs={d['normalized_sha256'] for v in docs.values() for d in v};seen_sentences={r['id'] for v in features.values() for r in v}
    OUT.mkdir();(OUT/'source_pages').mkdir();requests=[];downloaded=0;started=time.monotonic()
    def consume(page):
        for row in page['rows']:
            if len(docs['train'])==4096:return
            if row.get('truncated_cells'):continue
            text=row['row']['text'];dh=sha(' '.join(text.split()))
            if dh in seen_docs or int(dh[:8],16)%9==0:continue
            seen_docs.add(dh);candidates=[];local=set()
            for raw in re.split(r'(?<=[.!?])\s+',text):
                s=' '.join(raw.split());h=sha(s)
                if not 48<=len(s.encode('utf-8'))<=128 or s[-1] not in '.!?' or '\ufffd' in s:continue
                if any(ord(c)<32 for c in s) or sum(c.isalpha() for c in s)<16 or h in seen_sentences or h in local:continue
                local.add(h);candidates.append((h,s))
            if len(candidates)<4:continue
            docid=f'OWT:train:{row["row_idx"]}'
            docs['train'].append({'id':docid,'row_idx':row['row_idx'],'normalized_sha256':dh,'raw_sha256':sha(text),'text':text})
            for h,s in sorted(candidates)[:4]:
                seen_sentences.add(h);features['train'].append({'id':h,'prompt':s})
                targets['train'].append({'id':h,'document_id':docid,'target':s,'target_ids':list(s.encode('utf-8'))+[256],
                    'sentence_boundary':True,'document_EOF_claimed':False,'normalized_source_sentence_sha256':h})
    reused=[]
    for artifact in (ORIGINAL,INTERMEDIATE,PARENT):
        for page in sorted((artifact/'source_pages').glob('*.json')):
            consume(load(page));reused.append({'path':str(page),'sha256':sha(page.read_bytes())})
    recovered=len(docs['train'])-2048
    last=max(r['offset']+r['length'] for r in parent['requests']+load(INTERMEDIATE/'manifest.json')['requests']+original['requests'])
    for offset in range(last,last+8000,100):
        if len(docs['train'])==4096:break
        if time.monotonic()-started>180 or downloaded+2*1024**2>24*1024**2:raise RuntimeError('bounded data fetch cap reached')
        url=f'https://datasets-server.huggingface.co/rows?dataset=Skylion007%2Fopenwebtext&config=plain_text&split=train&offset={offset}&length=100'
        with urllib.request.urlopen(url,timeout=min(20,max(1,180-(time.monotonic()-started)))) as response:
            b=response.read(2*1024**2+1);revision=response.headers.get('x-revision')
        if len(b)>2*1024**2 or revision!=REVISION:raise RuntimeError('page size or immutable source revision mismatch')
        downloaded+=len(b);(OUT/'source_pages'/f'rows_{offset:05d}.json').write_bytes(b)
        requests.append({'offset':offset,'length':100,'bytes':len(b),'sha256':sha(b),'revision':revision});consume(json.loads(b))
    if len(docs['train'])!=4096:raise RuntimeError('new fixed pool incomplete; preserve partial pages')
    for panel in ('train','dev'):
        previous_docs=load(PARENT/'documents.json')[panel];previous_f=load(PARENT/'features.json')[panel];previous_y=load(PARENT/'targets.json')[panel]
        assert docs[panel][:len(previous_docs)]==previous_docs and features[panel][:len(previous_f)]==previous_f and targets[panel][:len(previous_y)]==previous_y
        if panel=='dev':assert len(docs[panel])==64 and len(features[panel])==256
    assert not {d['normalized_sha256'] for d in docs['train']}&{d['normalized_sha256'] for d in docs['dev']}
    assert len({r['id'] for v in features.values() for r in v})==16640
    for name,value in (('documents.json',docs),('features.json',features),('targets.json',targets)):io.atomic_json(OUT/name,value)
    manifest={'schema':'owt_fuzzy_AE_expansion_v20_cpu','dataset':'Skylion007/openwebtext','revision':REVISION,
        'parent_artifact':str(PARENT),'parent_manifest_sha256':sha((PARENT/'manifest.json').read_bytes()),
        'documents':{k:len(v) for k,v in docs.items()},'rows':{k:len(v) for k,v in features.items()},'old_training_documents':2048,'new_training_documents':2048,
        'new_documents_recovered_from_old_pages':recovered,'reused_pages':reused,'old_train_rows_unchanged_prefix':True,'dev_documents_and_rows_unchanged':True,
        'sentence_UTF8_bytes':{k:sum(len(r['prompt'].encode('utf-8')) for r in v) for k,v in features.items()},
        'old_train_sentence_UTF8_bytes':sum(len(r['prompt'].encode('utf-8')) for r in features['train'][:8192]),
        'new_train_sentence_UTF8_bytes':sum(len(r['prompt'].encode('utf-8')) for r in features['train'][8192:]),
        'exact_normalized_document_cross_split_overlap':0,'exact_selected_sentence_cross_split_overlap':0,
        'near_duplicate_checked_beyond_source_curation':False,'pretraining_overlap':'UNKNOWN','E2_final_accessed':False,
        'EOS_means_sentence_boundary_not_document_EOF':True,'fetch_bytes':downloaded,'fetch_cap_bytes':24*1024**2,'fetch_cap_seconds':180,
        'fetch_seconds':time.monotonic()-started,'requests':requests,'source_rights_note':parent['source_rights_note'],
        'modifications':parent['modifications']+';2048more training documents appended; original64held unchanged',
        'files':{str(p.relative_to(OUT)):sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}}
    io.atomic_json(OUT/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('documents','rows','sentence_UTF8_bytes','new_documents_recovered_from_old_pages','fetch_bytes','fetch_seconds')}))
if __name__=='__main__':main()
