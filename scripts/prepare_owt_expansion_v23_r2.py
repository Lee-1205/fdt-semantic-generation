"""Append4096 train docs; old4096 docs and held64 remain byte-identical.

Conservative abbreviation/start/length heuristics apply ONLY to new train rows.
These offline selection rules are not semantic gold, runtime roles or NN input.
"""
import copy, hashlib, json, re, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import learning_io_v2 as io
PARENT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v20'
ANCESTORS=[ROOT/f'research/artifacts/owt_fuzzy_reconstruction_20261028_v{v}' for v in (12,13,15,20)]
RECOVERED=[ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23',ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_retry1']
OUT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_r2'
REVISION='79d93d786212f7344586290adb811d4ae6a1762c'
FETCH_CAP=0
WALL=60
RULES={'protect_dotted_initialisms':True,'protect_single_capital_initial':True,
       'abbreviations':['Mr.','Mrs.','Ms.','Dr.','Prof.','Gov.','Rep.','Sen.','St.','No.','Jr.','Sr.','vs.','etc.','Jan.','Feb.','Mar.','Apr.','Jun.','Jul.','Aug.','Sep.','Sept.','Oct.','Nov.','Dec.'],
       'new_rows_start':'after opening quote/bracket, ASCII uppercase or digit',
       'new_rows_min_words':6,'new_rows_bytes':[48,128],'semantic_completeness_guaranteed':False,
       'old_rows_or_held_modified':False,'runtime_regex_or_gold_features':False}

def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode('utf-8')).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def abbreviation_end(s):
    token=s.rsplit(' ',1)[-1]
    return token.lower() in {x.lower() for x in RULES['abbreviations']} or bool(re.fullmatch(r'(?:[A-Za-z]\.){2,}|[A-Z]\.',token))

def split_spans(text):
    normalized=' '.join(text.split());start=0
    for match in re.finditer(r'(?<=[.!?])\s+',normalized):
        if abbreviation_end(normalized[:match.start()]):continue
        yield normalized[start:match.start()];start=match.end()
    if start<len(normalized):yield normalized[start:]

def acceptable_new_span(s):
    lead=s.lstrip('"\'“”‘’([{')
    return bool(lead) and (lead[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') and \
        48<=len(s.encode('utf-8'))<=128 and s[-1] in '.!?' and not abbreviation_end(s) and \
        '\ufffd' not in s and not any(ord(c)<32 for c in s) and \
        sum(c.isalpha() for c in s)>=16 and len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",s))>=6

def main():
    if OUT.exists():raise FileExistsError('new immutable artifact only')
    for artifact in ANCESTORS:
        m=load(artifact/'manifest.json')
        if m['revision']!=REVISION:raise ValueError('source revision')
        for name,h in m['files'].items():
            if sha((artifact/name).read_bytes())!=h:raise ValueError('parent artifact changed')
    parent=load(PARENT/'manifest.json')
    docs=copy.deepcopy(load(PARENT/'documents.json'));features=copy.deepcopy(load(PARENT/'features.json'));targets=copy.deepcopy(load(PARENT/'targets.json'))
    seen_docs={d['normalized_sha256'] for rows in docs.values() for d in rows}
    seen_spans={r['id'] for rows in features.values() for r in rows}
    OUT.mkdir();(OUT/'source_pages').mkdir();requests=[];downloaded=0;started=time.monotonic()
    def consume(page):
        for row in page['rows']:
            if len(docs['train'])==8192:return
            if row.get('truncated_cells'):continue
            text=row['row']['text'];dh=sha(' '.join(text.split()))
            if dh in seen_docs or int(dh[:8],16)%9==0:continue
            seen_docs.add(dh);candidates=[];local=set()
            for s in split_spans(text):
                h=sha(s)
                if not acceptable_new_span(s) or h in seen_spans or h in local:continue
                local.add(h);candidates.append((h,s))
            if len(candidates)<4:continue
            docid=f'OWT:train:{row["row_idx"]}'
            docs['train'].append({'id':docid,'row_idx':row['row_idx'],'normalized_sha256':dh,'raw_sha256':sha(text),'text':text})
            for h,s in sorted(candidates)[:4]:
                seen_spans.add(h);features['train'].append({'id':h,'prompt':s})
                targets['train'].append({'id':h,'document_id':docid,'target':s,'target_ids':list(s.encode('utf-8'))+[256],
                    'sentence_boundary':True,'document_EOF_claimed':False,'normalized_source_sentence_sha256':h})
    reused=[]
    for artifact in ANCESTORS:
        for page in sorted((artifact/'source_pages').glob('*.json')):
            consume(load(page));reused.append({'path':str(page),'sha256':sha(page.read_bytes())})
    for artifact in RECOVERED:
        expected=load(artifact/'incident.json')['saved_pages_sha256'] if (artifact/'incident.json').exists() else {Path(k).name:v for k,v in load(artifact/'manifest.json')['files'].items() if k.startswith('source_pages')}
        for page in sorted((artifact/'source_pages').glob('*.json')):
            if time.monotonic()-started>WALL:raise TimeoutError('fixed cache-only CPU cap')
            if sha(page.read_bytes())!=expected[page.name]:raise ValueError('cached complete page changed')
            consume(load(page));reused.append({'path':str(page),'sha256':sha(page.read_bytes()),'cache_only_recovered':True})
    recovered=len(docs['train'])-4096
    if downloaded!=0:raise RuntimeError('cache-only no network')
    if len(docs['train'])!=8192:raise RuntimeError('fixed pool incomplete; no automatic retry')
    for name,now in (('documents.json',docs),('features.json',features),('targets.json',targets)):
        old=load(PARENT/name)
        assert now['train'][:len(old['train'])]==old['train'] and now['dev']==old['dev']
        io.atomic_json(OUT/name,now)
    assert len(features['train'])==32768 and len(features['dev'])==256
    assert len({r['id'] for rows in features.values() for r in rows})==33024
    assert not {d['normalized_sha256'] for d in docs['train']}&{d['normalized_sha256'] for d in docs['dev']}
    manifest={'schema':'owt_fuzzy_AE_expansion_v23_cpu','dataset':'Skylion007/openwebtext','revision':REVISION,
        'parent_artifact':str(PARENT),'parent_manifest_sha256':sha((PARENT/'manifest.json').read_bytes()),
        'documents':{k:len(v) for k,v in docs.items()},'rows':{k:len(v) for k,v in features.items()},
        'old_training_documents':4096,'new_training_documents':4096,'new_documents_recovered_from_old_pages':recovered,
        'reused_pages':reused,'old_train_rows_unchanged_prefix':True,'dev_documents_and_rows_unchanged':True,
        'sentence_UTF8_bytes':{k:sum(len(r['prompt'].encode('utf-8')) for r in v) for k,v in features.items()},
        'old_train_sentence_UTF8_bytes':sum(len(r['prompt'].encode('utf-8')) for r in features['train'][:16384]),
        'new_train_sentence_UTF8_bytes':sum(len(r['prompt'].encode('utf-8')) for r in features['train'][16384:]),
        'new_training_span_rules':RULES,'rules_builder_sha256':sha(Path(__file__).read_bytes()),'data_revision':2,'cache_only':True,'superseded_CPU_artifacts_preserved':[str(x) for x in RECOVERED],
        'exact_normalized_document_cross_split_overlap':0,'exact_selected_sentence_cross_split_overlap':0,
        'near_duplicate_checked_beyond_source_curation':False,'pretraining_overlap':'UNKNOWN','E2_final_accessed':False,
        'EOS_means_sentence_boundary_not_document_EOF':True,'sentence_heuristic_not_semantic_gold':True,
        'fetch_bytes':downloaded,'fetch_cap_bytes':FETCH_CAP,'fetch_cap_seconds':WALL,'fetch_seconds':time.monotonic()-started,
        'requests':requests,'source_rights_note':parent['source_rights_note'],
        'modifications':parent['modifications']+';4096more train docs; conservative boundary/start selection on NEWrows only; held unchanged',
        'files':{str(p.relative_to(OUT)):sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}}
    io.atomic_json(OUT/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('documents','rows','sentence_UTF8_bytes','new_documents_recovered_from_old_pages','fetch_bytes','fetch_seconds')}))

if __name__=='__main__':main()
