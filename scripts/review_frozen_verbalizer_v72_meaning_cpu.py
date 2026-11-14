"""Actual decoded source-faithfulness cards; no NN or output repair."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_factual_cycle_v70_meaning_cpu import long_score

def main():
    run=ROOT/'runs/frozen_verbalizer_20261112_v72';p=io.load_json(run/'plan.json');d=Path(p['data']);old=Path(p['baseline_run'])
    assert io.load_json(run/'root_saved_review.json')['status']=='ROOT_V72_SAVED_REVIEW_PASS'
    rows=io.load_json(Path(p['long_data'])/'features.json')[-2:];targets=io.load_json(Path(p['long_data'])/'targets.json')[-2:]
    before,bc=long_score(rows,targets,io.load_json(old/'long_reconstructions.json'));after,ac=long_score(rows,targets,io.load_json(run/'long_reconstructions.json'))
    short=io.load_json(run/'new_expression_reconstructions.json');base=io.load_json(run/'new_baseline_reconstructions.json');labels=io.load_json(d/'dev_targets.json')
    counts=collections.Counter();changes=[];basechanges=[];tails=[]
    for i,(a,b,t) in enumerate(zip(base,short,labels)):
        source=b['source'].encode()
        for stage,r,dest in [('before',a,basechanges),('after',b,changes)]:
            output=r['output'].encode()
            # These ASCII fixed-position readouts produce one selected byte at
            # each source-bearing slot. Some initial outputs add terminal bytes;
            # don't require equal whole-output length or shift gold spans.
            assert source.isascii() and output.isascii() and len(output)>=len(source)
            assert r['ids'][:len(output)]==list(output)
            if len(output)>len(source):tails.append(dict(stage=stage,row=i,source_end=len(source),extra=output[len(source):].decode(),whitespace_only=output[len(source):].isspace()))
            for s in t['critical_spans']:
                if stage=='after':counts[s['kind']+'_fields']+=1
                value=output[s['start']:s['end']].decode()
                if value.casefold()!=s['text'].casefold():dest.append(dict(row=i,kind=s['kind'],source=s['text'],output=value))
    shortcards=[dict(row=i,source=b['source'],before=a['output'],after=b['output']) for i,(a,b) in enumerate(zip(base,short))]
    ret=io.load_json(run/'retained_English_reconstructions.json');oldret=io.load_json(old/'retained_English_reconstructions.json')
    retcards=[dict(row=i,source=b['source'],before=a['output'],after=b['output']) for i,(a,b) in enumerate(zip(oldret,ret)) if len(b['source'].encode())<350 and (a['output']!=b['output'] or b['source']!=b['output'])]
    contexts=[]
    for label,oldlabel in [('OWT','final'),('retained_English','retained_English'),('prior_English','new_English')]:
        new=io.load_json(run/(label+'_reconstructions.json'));oldrecords=io.load_json(old/(oldlabel+'_reconstructions.json'))
        for i,(a,b) in enumerate(zip(oldrecords,new)):
            src=b['source'].encode();ob=a['output'].encode();nb=b['output'].encode();bad=[j for j in range(min(len(src),len(nb))) if src[j]!=nb[j]]
            groups=[]
            for j in bad:
                if not groups or j-groups[-1][-1]>40:groups.append([j])
                else:groups[-1].append(j)
            for group in groups:
                start=max(0,group[0]-45);end=min(len(src),group[-1]+60)
                contexts.append(dict(panel=label,row=i,byte_start=start,source=src[start:end].decode(errors='replace'),before=ob[start:end].decode(errors='replace'),after=nb[start:end].decode(errors='replace')))
    summary=dict(status='ROOT_V72_MEANING_CARDS_READY',long_before=before,long_after=after,new_expression_fields=dict(counts),new_expression_before_changes=basechanges,new_expression_after_changes=changes,
        new_expression_terminal_additions=tails,long_valid_different_numbers_before=[v for v in before['numeric_examples'] if v['output'].isdigit()],long_valid_different_numbers_after=[v for v in after['numeric_examples'] if v['output'].isdigit()],
        extra_model_calls=0,scope='offline source-bound restricted fixed-byte fields on verified actual ASCII decoded text; correlated authored templates/selected observed panels; no general semantic accuracy or abstract-state certificate; no output repair')
    io.atomic_json(run/'root_meaning_review.json',summary);io.atomic_json(run/'root_manual_reading_cards.json',dict(new_DEV32=shortcards,changed_short_English=retcards,long_before=bc,long_after=ac,remaining_contexts=contexts))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('long_before','long_after')}))
    print('LONG_SUMMARY '+json.dumps(dict(before={k:v for k,v in before.items() if k!='numeric_examples'},after={k:v for k,v in after.items() if k!='numeric_examples'})))
    print('RETAINED_SHORT '+json.dumps(retcards));print('LONG_CRITICAL_AFTER '+json.dumps([c for c in ac if any(k in c['changes'] for k in ('quantity','reserve','actor','checker','negative_actor','order','negative'))]))
if __name__=='__main__':main()
