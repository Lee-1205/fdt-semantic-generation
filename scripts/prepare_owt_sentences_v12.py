"""Bounded OWT sentence sample, document-disjoint prospective split, no full download."""
import hashlib,json,re,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import learning_io_v2 as io
REVISION='79d93d786212f7344586290adb811d4ae6a1762c'
ARTIFACT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12'

def digest(s):return hashlib.sha256(s if isinstance(s,bytes) else s.encode('utf-8')).hexdigest()

def main():
    if ARTIFACT.exists():raise FileExistsError('new artifact only')
    ARTIFACT.mkdir();pages=ARTIFACT/'source_pages';pages.mkdir();started=time.monotonic();downloaded=0
    documents={'train':[],'dev':[]};features={'train':[],'dev':[]};targets={'train':[],'dev':[]}
    doc_seen=set();sentence_seen=set();requests=[]
    def fetch(url,maximum):
        nonlocal downloaded
        if time.monotonic()-started>180 or downloaded+maximum>32*1024**2:raise RuntimeError('fixed source fetch cap')
        with urllib.request.urlopen(url,timeout=20) as response:
            b=response.read(maximum+1)
            if len(b)>maximum:raise RuntimeError('response cap')
            downloaded+=len(b);return b,dict(response.headers)
    for name,url in [('hub_metadata.json','https://huggingface.co/api/datasets/Skylion007/openwebtext'),
                     ('README.md',f'https://huggingface.co/datasets/Skylion007/openwebtext/raw/{REVISION}/README.md'),
                     ('author_source.html','https://skylion007.github.io/OpenWebTextCorpus/')]:
        b,headers=fetch(url,1024**2);(ARTIFACT/name).write_bytes(b)
    if json.loads((ARTIFACT/'hub_metadata.json').read_bytes())['sha']!=REVISION:raise RuntimeError('source revision changed')
    for offset in range(0,4000,100):
        url=f'https://datasets-server.huggingface.co/rows?dataset=Skylion007%2Fopenwebtext&config=plain_text&split=train&offset={offset}&length=100'
        b,headers=fetch(url,2*1024**2)
        revision=next((v for k,v in headers.items() if k.lower()=='x-revision'),None)
        if revision!=REVISION:raise RuntimeError('row source revision differs')
        (pages/f'rows_{offset:05d}.json').write_bytes(b);page=json.loads(b)
        requests.append({'offset':offset,'length':100,'bytes':len(b),'sha256':digest(b),'revision':revision})
        for row in page['rows']:
            if row.get('truncated_cells'):continue
            text=row['row']['text'];normal=' '.join(text.split());dh=digest(normal)
            if dh in doc_seen:continue
            doc_seen.add(dh);panel='dev' if int(dh[:8],16)%9==0 else 'train'
            cap=64 if panel=='dev' else 512
            if len(documents[panel])>=cap:continue
            candidates=[];local=set()
            for raw in re.split(r'(?<=[.!?])\s+',text):
                s=' '.join(raw.split());h=digest(s);n=len(s.encode('utf-8'))
                if not 48<=n<=128 or not s or s[-1] not in '.!?' or '\ufffd' in s:continue
                if any(ord(c)<32 for c in s) or sum(c.isalpha() for c in s)<16 or h in sentence_seen or h in local:continue
                local.add(h);candidates.append((h,s))
            if len(candidates)<4:continue
            selected=sorted(candidates)[:4];docid=f'OWT:train:{row["row_idx"]}'
            documents[panel].append({'id':docid,'row_idx':row['row_idx'],'normalized_sha256':dh,'raw_sha256':digest(text),'text':text})
            for h,s in selected:
                sentence_seen.add(h);features[panel].append({'id':h,'prompt':s})
                targets[panel].append({'id':h,'document_id':docid,'target':s,'target_ids':list(s.encode('utf-8'))+[256],
                    'sentence_boundary':True,'document_EOF_claimed':False,'normalized_source_sentence_sha256':h})
        if len(documents['train'])==512 and len(documents['dev'])==64:break
    if {k:len(v) for k,v in features.items()}!={'train':2048,'dev':256}:raise RuntimeError('bounded fetch did not fill fixed pools')
    assert not {d['normalized_sha256'] for d in documents['train']}&{d['normalized_sha256'] for d in documents['dev']}
    for n,obj in [('documents.json',documents),('features.json',features),('targets.json',targets)]:io.atomic_json(ARTIFACT/n,obj)
    manifest={'schema':'owt_sentence_fuzzy_autoencode_v12_cpu','dataset':'Skylion007/openwebtext','revision':REVISION,
        'source_split':'train','prospective_document_split':'normalized document SHA256 prefix modulo9: dev0, otherwise train; first eligible512/64documents',
        'documents':{k:len(v) for k,v in documents.items()},'rows':{k:len(v) for k,v in features.items()},
        'source_document_UTF8_bytes':{k:sum(len(d['text'].encode('utf-8')) for d in v) for k,v in documents.items()},
        'sentence_UTF8_bytes':{k:sum(len(r['prompt'].encode('utf-8')) for r in v) for k,v in features.items()},
        'selected_sentences_per_document':4,'max_sentence_UTF8_bytes':128,'EOS_means_sentence_boundary_not_document_EOF':True,
        'exact_selected_sentence_cross_split_overlap':0,'exact_normalized_document_cross_split_overlap':0,'near_duplicate_checked_beyond_source_curation':False,
        'pretraining_overlap':'UNKNOWN','broad_LM_tokens_claimed':False,'E2_final_accessed':False,'old_holdout_accessed':False,
        'fetch_bytes':downloaded,'fetch_cap_bytes':32*1024**2,'fetch_seconds':time.monotonic()-started,'requests':requests,
        'source_rights_note':'Author licenses packaging CC0, does not claim ownership of underlying web text; source README/author notice retained. User-directed local research; no publication authorization or blanket underlying-text clearance claimed.',
        'modifications':'sentence segmentation, whitespace normalization,48-128UTF8byte selection,4hash-selected sentences/document, prospective document split',
        'files':{p.name:digest(p.read_bytes()) for p in ARTIFACT.iterdir() if p.is_file()}}
    io.atomic_json(ARTIFACT/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k in ('revision','documents','rows','sentence_UTF8_bytes','fetch_bytes','fetch_seconds')}))
if __name__=='__main__':main()
