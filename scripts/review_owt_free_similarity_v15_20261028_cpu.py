"""Apply unchanged saved-text measures to V15; no new forward or embedding model."""
import hashlib,json,statistics
from pathlib import Path
from review_owt_free_similarity_20261028_cpu import metrics,chrf,REVIEWS
ROOT=Path(__file__).resolve().parents[1]
COMMENTS=[
 'He didnocknowly is closer to the opening spelling; U.S. suggests an American topic, but culpability/deceit/negation not expressed.',
 'Fansting resembles Fans; waving, flower accessories and anticipation absent.',
 'Uppercase fragments and TO repetition; Disney/donut/sundae proposition absent.',
 'Glanday resembles Globally initially; ransom, victim, restoring backups and cause absent.',
 'Death and first one partly resemble the opening; team, confrontation, Slade Wilson and Moscow absent.',
 'No Israel/Syria/Assad/ceasefire relation retained.',
 'She fi resembles the opening; sobriety test failure and DUI charge absent.',
 'I then survives and producted resembles proceeded; leaving without paying for stolen items absent.',
 'But more survives; speech restriction effectiveness, negation and causal relation absent.',
 'Names, scouting radar relation and three-month timing absent.',
 'We and asked resemble the opening, but no second academic/anonymity/article review relation.',
 'People and inaction-as-concern relation absent.',
 'Country/name-like list structure survives; actual six countries, gunmen and held vessels absent.',
 'Invented1999999,2014repeat and19; qualification/team/Sochi relation absent.',
 'Japanises is not Janissa; prevention, assassination and wall absent.',
 'The and string-like morphology only; function names, temporary string and pointer relationship absent.'
]
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    run=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v15';raw=load(run/'final_generation.json');previous=load(run/'parent_reference_generation.json')
    result={'status':'V15_GRADED_FREE_TEXT_REVIEW_COMPLETE','rows':{},'summary':{},'previous_summary':{},
        'source_sha256':hashlib.sha256((run/'final_generation.json').read_bytes()).hexdigest(),
        'ROOT_read16_faithful_paraphrases':0,'qualitative_nonblind_judgment':True,'additional_model_forwards':0,
        'metric_definition_reused_from':'review_owt_free_similarity_20261028_cpu.py','semantic_embedding_scorer_run':False,
        'limitations':'Posthoc same observed16devdocuments; lexical similarity not calibrated semantic probability. ROOT read each proposition separately.'}
    for mode in raw:
        scored=[];oldscored=[]
        for i,(row,old) in enumerate(zip(raw[mode],previous[mode])):
            assert row['id']==old['id'] and row['source']==old['source']
            x={'index':i,'id':row['id'],'source':row['source'],'output':row['generated']['text'],**metrics(row['source'],row['generated']['text'])}
            if mode=='normal':x.update(source_proposition=REVIEWS[i][0],ROOT_reading=COMMENTS[i],faithful_paraphrase=False)
            scored.append(x);oldscored.append(metrics(old['source'],old['generated']['text']))
        result['rows'][mode]=scored
        for rows,key in ((scored,'summary'),(oldscored,'previous_summary')):
            summary={k:statistics.mean(x[k] for x in rows) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
            summary['median_correct_initial_UTF8_bytes']=statistics.median(x['correct_initial_UTF8_bytes'] for x in rows)
            result[key][mode]=summary
        refs=[row['source'] for row in scored];top=0;margin=[]
        for i,row in enumerate(scored):
            scores=[chrf(ref,row['output']) for ref in refs];mx=max(scores)
            top+=scores[i]==mx and scores.count(mx)==1;margin.append(scores[i]-statistics.mean(v for j,v in enumerate(scores) if j!=i))
        result['summary'][mode].update(own_source_unique_top1_of16=top,own_vs_other15_chrF_margin=statistics.mean(margin),any_exact_content_word_rows=sum(bool(x['shared_content_words']) for x in scored))
    with (run/'root_free_similarity.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    lines=['# V15 all16 saved free reconstructions: ROOT reading + unchanged graded similarity','', 'Posthoc; no new inference, no embedding judge.']
    for row in result['rows']['normal']:
        lines.extend(['',f"## {row['index']}",f"Source: {row['source']}",f"Output: {row['output']}",f"ROOT reading: {row['ROOT_reading']}",f"chrF={row['chrF2_1to6']:.3f}; contentF1={row['content_word_F1']:.4f}; prefixbytes={row['correct_initial_UTF8_bytes']}"])
    with (run/'root_all16_free_reading.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'V14':result['previous_summary'],'V15':result['summary']},ensure_ascii=True))
if __name__=='__main__':main()
