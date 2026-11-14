"""Authored synthetic raw-text corpus. Templates are offline labels, never inference."""
import hashlib
import itertools
import json
from pathlib import Path
from collections import Counter
from research.text_diffusion_v1 import pack_answers,pack_prompts

OPS=('greater','less','not greater','not less')

def digest(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

def render(a,b,op,reverse,lang,template):
    q,r=(b,a) if reverse else (a,b)
    relation=('less' if op=='greater' else 'greater') if reverse and op in OPS[:2] else op
    if lang=='en':
        phrase={'greater':'larger than','less':'smaller than','not greater':'not larger than','not less':'not smaller than'}[op]
        fact=f'{a} is {phrase} {b}.'
        query=f'How does {q} compare with {r}?'
        prompt=(f'{fact} {query}',f'Fact: {fact} Question: {query}',
                f'{query} Given: {fact}',f'Use the fact. {fact} {query}',
                f'Given {fact} Compare {q} and {r}.')[template]
        answer=f'{q} is {"larger" if relation=="greater" else "smaller"} than {r}.' if op in OPS[:2] else 'It is unknown.'
    else:
        phrase={'greater':'크다','less':'작다','not greater':'크지 않다','not less':'작지 않다'}[op]
        fact=f'{a}는 {b}보다 {phrase}.'
        query=f'{q}와 {r}의 관계는?'
        prompt=(f'{fact} {query}',f'사실: {fact} 질문: {query}',
                f'{query} 조건: {fact}',f'사실로 답하라. {fact} {query}',
                f'주어진 사실은 {fact} {q}와 {r}를 비교하라.')[template]
        answer=f'{q}는 {r}보다 {"큽니다" if relation=="greater" else "작습니다"}.' if op in OPS[:2] else '알 수 없습니다.'
    return prompt,answer,relation if op in OPS[:2] else 'unknown'

def build():
    train=[]; dev=[]; novel=[]
    for a,b in itertools.permutations('ABCDEFGH',2):
        # All directions/operators/query order/templates for one entity pair and
        # language stay in one split. Names and answer strings can still overlap.
        for lang in ('en','ko'):
            group=''.join(sorted((a,b)))+'|'+lang
            split='dev' if int(digest(group)[:8],16)%5==0 else 'train'
            for op in OPS:
                for reverse in (False,True):
                    for template in range(5):
                        prompt,answer,label=render(a,b,op,reverse,lang,template)
                        row={'id':digest(prompt),'group':group,'prompt':prompt,'answer':answer,
                             'language':lang,'label':label,'operator':op,'reverse':reverse,'template':template}
                        if template==4:
                            if split=='train': novel.append(row)
                        elif split=='train': train.append(row)
                        else: dev.append(row)
    def panel(rows,count):
        # Deterministic language/answer-class stratification, before any output.
        selected=[]; strata=sorted(set((r['language'],r['label']) for r in rows))
        queues={s:sorted([r for r in rows if (r['language'],r['label'])==s],key=lambda r:r['id']) for s in strata}
        while len(selected)<count:
            for s in strata:
                if len(selected)==count: break
                selected.append(queues[s].pop(0))
        return selected
    data={'train':train,'train_probe':panel(train,16),'dev_pairs':panel(dev,32),'dev_wording':panel(novel,32)}
    check(data)
    return data

def check(data):
    train_groups={r['group'] for r in data['train']}
    if train_groups & {r['group'] for r in data['dev_pairs']}: raise ValueError('pair split overlap')
    train_prompts={r['prompt'] for r in data['train']}
    for name,rows in data.items():
        for r in rows:
            if r['id']!=digest(r['prompt']): raise ValueError('row identity')
            if name.startswith('dev') and r['prompt'] in train_prompts: raise ValueError('prompt contamination')
            # Independent direction rule, not model-derived labels.
            expected='unknown' if r['operator'].startswith('not ') else r['operator']
            if r['reverse'] and expected!='unknown': expected='less' if expected=='greater' else 'greater'
            if r['label']!=expected: raise ValueError('direction label')
        pack_prompts([r['prompt'] for r in rows]); pack_answers([r['answer'] for r in rows])
    return {'rows':{k:len(v) for k,v in data.items()},
            'support':{k:dict(Counter(r['language']+'|'+r['label'] for r in v)) for k,v in data.items()},
            'prompt_utf8_bytes':{k:sum(len(r['prompt'].encode()) for r in v) for k,v in data.items()},
            'answer_utf8_bytes':{k:sum(len(r['answer'].encode()) for r in v) for k,v in data.items()},
            'new_external_corpus':False,'synthetic':True,'e2_final_accessed':False,
            'evaluation_selection':'fixed hash/strata before model; observed development, not blind final'}

def write_new(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    data=build(); meta=check(data)
    for name,rows in data.items():
        (directory/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    meta['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.json')}
    (directory/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return meta
