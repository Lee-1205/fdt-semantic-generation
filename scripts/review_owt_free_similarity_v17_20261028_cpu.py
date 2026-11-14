"""Apply unchanged saved-text measures to V17; no new forward or embedding model."""
import hashlib,json,statistics
from pathlib import Path
from review_owt_free_similarity_20261028_cpu import metrics,chrf,REVIEWS
ROOT=Path(__file__).resolve().parents[1]
COMMENTS=['He survives, but acknowledging personal culpability, deceit and the American public are absent.', 'Fanstine and expecting resemble fans/anticipation superficially; waving flowers and standing absent.', 'Uppercase fragments only; Disney donut sundae and surprised question are not expressed.', 'Glably resembles Globally; football is invented and ransom/backup restoration cause absent.', 'Panel, team, confrontation, Slade Wilson and Moscow absent.', 'No Israel/Syria/Assad/ceasefire relation; invented proper-name-like fragments.', 'She fi survives; no sobriety test failure or DUI charge, lawson is not that relation.', 'I then opening is lost; pay survives among Apple/officers hallucinations, without-paying relation absent.', 'But and express loosely relate to discourse/speech; original negation and causal restriction claim absent.', 'Robby is now correctly retained, but Fabbri/scouting radar/three-month relation absent.', 'We can searched does not preserve second academic reviewing anonymous article copy.', 'Web is not Wubbels; lawyer, inaction and primary concern relationship absent.', 'Gunshan resembles Gunmen initially; held vessels and actual country list absent.', 'No Jamaican bobsled qualification, Winter Olympics or Sochi relation.', 'Japanisas resembles name spelling; assassination prevention and wall absent.', 'The funting resembles function spelling; names, temporary string and pointers relation absent.']

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    run=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v17';raw=load(run/'final_generation.json');previous=load(run/'parent_reference_generation.json')
    result={'status':'V17_GRADED_FREE_TEXT_REVIEW_COMPLETE','rows':{},'summary':{},'previous_summary':{},
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
    lines=['# V17 all16 saved free reconstructions: ROOT reading + unchanged graded similarity','', 'Posthoc; no new inference, no embedding judge.']
    for row in result['rows']['normal']:
        lines.extend(['',f"## {row['index']}",f"Source: {row['source']}",f"Output: {row['output']}",f"ROOT reading: {row['ROOT_reading']}",f"chrF={row['chrF2_1to6']:.3f}; contentF1={row['content_word_F1']:.4f}; prefixbytes={row['correct_initial_UTF8_bytes']}"])
    with (run/'root_all16_free_reading.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'V15':result['previous_summary'],'V17':result['summary']},ensure_ascii=True))
if __name__=='__main__':main()
