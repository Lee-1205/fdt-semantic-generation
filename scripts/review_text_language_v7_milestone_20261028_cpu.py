"""One concise saved-state milestone calculation; no new model calls or audit loop."""
import json,sys
from pathlib import Path
from collections import Counter,defaultdict
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import text_generation_support_v1 as support
from review_text_language_v6_20261028_cpu import gram_stats

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    torch.set_num_threads(2);r=ROOT/'runs/text_language_continuation_20261028_v7'
    osend=read(r/'supervisor/final_status.json');native=read(r/'code_snapshot/root_native_terminal_check.json')
    assert osend['status']=='COMPLETE' and osend['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    result=read(r/'result.json');latest=result['latest'];p=r/'model/checkpoints'/latest['file']
    assert support.sha(p)==latest['sha256']
    state=torch.load(p,map_location='cpu',weights_only=True);assert support.state_sha(state)==latest['algorithm_sha256']
    assert (state['attempts'],state['updates'],state['cursor'])==(818,818,6542)
    assert state['exposure']['processed_target_bytes']==769350 and len(state['history'])==818
    assert all(h['outcome']=='UPDATED' for h in state['history'])
    batches=state['curriculum']['batches'];assert [h['rows'] for h in state['history']]==batches
    for i in (0,409):assert sorted(j for b in batches[i:i+409] for j in b)==list(range(3271))
    assert len(batches[408])==len(batches[-1])==7
    adam=Counter(int(s['step']) for s in state['optimizer']['state'].values())
    assert adam=={1226:74,1078:70};assert state['scaler']['_growth_tracker']==1226
    review={'status':'V7_EARLY_LEARNING_MILESTONE','worker_result_sha256':support.sha(r/'result.json'),
        'final_fullstate_sha256':latest['sha256'],'final_algorithm_sha256':latest['algorithm_sha256'],
        'final_state_CPU_hash_and_safeload':True,'Adam_step_histogram':dict(adam),'scaler':state['scaler'],
        'new_attempts':818,'new_updates':818,'global_updates':1226,'new_finite_epochs':2,
        'epoch_last7rows_preserved':True,'new_target_byte_exposure':769350,
        'cumulative_backpropagated_target_bytes':383651+769350,'unique_train_documents':512,
        'same_documents_reexposed':True,'all_owned_gone_before_analysis':True,'fresh_native_python_zero':True,
        'baseline_reused':True,'quality_metrics_were_not_stops':True,'additional_model_forwards':0,
        'whole_model_failure_claim':False,'E2_final_accessed':False}
    del state
    targets=read(ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu/targets.json')['dev']
    reports={};documents={}
    for phase in ('baseline','final'):
        saved=torch.load(r/(phase+'_NLL_logits.pt'),map_location='cpu',weights_only=True)
        logits=saved['logits'].double();ids=saved['target_ids'];active=ids!=257;end=active&(ids==256);nonend=active&~end
        losses=logits.logsumexp(-1)-logits.gather(-1,ids[...,None]).squeeze(-1);pred=logits.argmax(-1)
        doc=defaultdict(lambda:[0.,0])
        for i,target in enumerate(targets):
            a=doc[target['document_id']];a[0]+=float(losses[i,active[i]].sum());a[1]+=int(active[i].sum())
        documents[phase]={k:a[0]/a[1] for k,a in doc.items()}
        raw=read(r/(phase+'_generation.json'));generations={}
        for mode,rows in raw.items():
            stats=[]
            for row in rows:
                g=row['generated'];data=bytes.fromhex(g['raw_hex']);assert data.decode('utf-8')==g['text']
                assert data==bytes(x for x in g['token_ids'] if x<256)
                stats.append(gram_stats(g['text']))
            generations[mode]={'rows':len(rows),'UTF8':sum(x['generated']['utf8_valid'] for x in rows),
                'EOS':sum(x['generated']['eos'] for x in rows),'unique_strings':len({x['generated']['text'] for x in rows}),
                'repeated_4gram_fraction_mean':sum(x['repeated_4gram_fraction'] for x in stats)/len(stats),
                'the_word_fraction_mean':sum(x['the_word_fraction'] for x in stats)/len(stats)}
        reports[phase]={'NLL':float(losses[active].mean()),'accuracy':float((pred[active]==ids[active]).double().mean()),
            'EOF_correct':int((pred[end]==256).sum()),'EOF_total':int(end.sum()),
            'EOF_probability_mean':float(logits.softmax(-1)[...,256][end].mean()),
            'nonEOF_falseEOS':int((pred[nonend]==256).sum()),'nonEOF_total':int(nonend.sum()),
            'generation':generations}
    review['metrics']=reports
    review['documents_NLL_improved']=sum(documents['final'][k]<v for k,v in documents['baseline'].items())
    review['documents_total']=64
    review['interpretation']='Continued byte prediction learning on small repeated corpus; greedy repetition remains and worsened. An early training milestone, not whole-model failure or success at general generation.'
    with (r/'root_result.json').open('x',encoding='utf-8') as f:json.dump(review,f,ensure_ascii=False,indent=2)
    print(json.dumps(review,ensure_ascii=True))
if __name__=='__main__':main()
