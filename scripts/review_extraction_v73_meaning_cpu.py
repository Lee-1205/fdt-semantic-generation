"""Limited source-fidelity ledgers on actual decoded held English output."""
import collections,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN,fields

def short_fields(rows,labels,records):
    total=collections.Counter();changed=collections.Counter();errors=[];rowcards=[];tail=[]
    for i,(row,t,r) in enumerate(zip(rows,labels,records)):
        source=row['prompt'].encode();output=r['output'].encode();assert r['ids'][:len(output)]==list(output)
        for s in t['critical_spans']:
            kind=s['kind'];total[kind]+=1;actual=output[s['start']:s['end']].decode('utf8',errors='replace')
            if actual.casefold()!=s['text'].casefold():changed[kind]+=1;errors.append(dict(row=i,kind=kind,source=s['text'],output=actual))
        if len(output)!=len(source):tail.append(dict(row=i,source_bytes=len(source),output_bytes=len(output),extra=output[len(source):].decode(errors='replace') if len(output)>len(source) else None))
        if row['prompt']!=r['output']:rowcards.append(dict(row=i,origin=t['origin'],source=row['prompt'],output=r['output']))
    return dict(rows=len(rows),field_totals=dict(total),changed_source_fields=dict(changed),errors=errors,tail_length_changes=tail,nonliteral_rows=len(rowcards)),rowcards

def long_fields(rows,labels,records):
    summaries=[];cards=[];regions=[];patterns=collections.defaultdict(list)
    for i,(row,t,r) in enumerate(zip(rows,labels,records)):
        source=row['prompt'].encode();output=r['output'].encode();assert r['ids'][:len(output)]==list(output);counts=collections.Counter();core=0;other=0;polarity=collections.Counter()
        for j,event in enumerate(t['events']):
            start,end=event['byte_start'],event['byte_end'];ss=source[start:end].decode();oo=output[start:end].decode(errors='replace');match=PATTERN.fullmatch(ss);assert match
            changes={};covered=set()
            for name,value in match.groupdict().items():
                a,b=match.span(name)
                if a<0:continue
                covered.update(range(a,b));actual=output[start+a:start+b].decode(errors='replace')
                if value.casefold().strip()!=actual.casefold().strip():changes[name]=dict(source=value,output=actual);counts[name]+=1
            form=any(a!=b and k not in covered for k,(a,b) in enumerate(zip(source[start:end],output[start:end]))) or len(output[start:end])!=end-start
            core+=not {k:v for k,v in changes.items() if k!='episode'} and not form;other+=form
            pm=PATTERN.fullmatch(oo)
            if pm:polarity['known_changed' if bool(pm.group('negative'))!=event['negative_removal'] else 'known_same']+=1
            else:
                # Only inside this source-delimited event. No regex is allowed
                # to bridge a neighboring record when another word is damaged.
                tail=re.search(r'\bdid\s+(not\s+)?remove\b',oo[-30:],flags=re.I)
                polarity['local_known_changed' if tail and bool(tail.group(1))!=event['negative_removal'] else 'local_known_same' if tail else 'unresolved_manual']+=1
            if changes or form:
                card=dict(row=i,event=j+1,origin=t['origin'],style=t['style'],source_bytes=len(source),changes=changes,other_word_or_form=form,
                    source=ss,output=oo,relative_position=start/len(source),crosses_compute_boundary=start//2048!=(end-1)//2048)
                cards.append(card)
                for name,v in changes.items():patterns[(name,v['source'],v['output'])].append(dict(row=i,event=j+1))
        for j in (0,len(t['events'])//2,len(t['events'])-1):
            event=t['events'][j];a,b=event['byte_start'],event['byte_end'];regions.append(dict(row=i,origin=t['origin'],style=t['style'],region='start' if j==0 else 'end' if j==len(t['events'])-1 else 'middle',source=source[a:b].decode(),output=output[a:b].decode(errors='replace')))
        summaries.append(dict(row=i,origin=t['origin'],style=t['style'],source_bytes=len(source),output_bytes=len(output),multiple_of_TRAINmax=len(source)/16384,
            events=len(t['events']),changed_source_fields=dict(counts),numeric_changed=sum(counts[n] for n in ('quantity','reserve')),name_changed=sum(counts[n] for n in ('actor','checker','negative_actor')),
            exact_core_surface_event_bundles=core,other_word_or_form_events=other,polarity=dict(polarity),EOS=r['EOS'],UTF8=r['UTF8']))
    unique=[dict(field=k[0],source=k[1],output=k[2],occurrences=len(v),locations=v) for k,v in patterns.items()]
    return dict(rows=summaries,number_changes=sum(r['numeric_changed'] for r in summaries),name_changes=sum(r['name_changed'] for r in summaries),unique_changed_field_patterns=unique),cards,regions

def main():
    run=ROOT/'runs/fuzzy_extraction_20261112_v73';p=io.load_json(run/'plan.json');e=Path(p['eval_data']);assert io.load_json(run/'root_saved_review.json')['status']=='ROOT_V73_SAVED_REVIEW_PASS'
    rows=io.load_json(e/'short_features.json');labels=io.load_json(e/'short_targets.json');before,bc=short_fields(rows,labels,io.load_json(run/'new_baseline_reconstructions.json'));after,ac=short_fields(rows,labels,io.load_json(run/'new_English_reconstructions.json'))
    long,lc,regions=long_fields(io.load_json(e/'long_features.json'),io.load_json(e/'long_targets.json'),io.load_json(run/'long_reconstructions.json'))
    old=Path(p['baseline_run']);oldrows=io.load_json(e/'long_features.json')[:2];oldlabels=io.load_json(e/'long_targets.json')[:2]
    oldlong,_,_=long_fields(oldrows,oldlabels,io.load_json(old/'long_reconstructions.json'))
    report=dict(status='ROOT_V73_RESTRICTED_MEANING_CARDS_READY',short_before=before,short_after=after,long=long,observed_long2_before=oldlong,
        extra_model_calls=0,scope='offline fixed-byte/source-bound authored clauses and actual decoded text; all fields are limited fidelity indicators, not abstract meaning/state accuracy; paraphrases/typos manually interpreted; no output repair')
    io.atomic_json(run/'root_meaning_review.json',report);io.atomic_json(run/'root_manual_reading_cards.json',dict(short_before=bc,short_after=ac,long_changed_events=lc,long_start_middle_end=regions))
    print(json.dumps(dict(short_before={k:v for k,v in before.items() if k!='errors'},short_after={k:v for k,v in after.items() if k!='errors'},long_rows=long['rows'],observed_long2_before=oldlong['rows'],unique_field_patterns=len(long['unique_changed_field_patterns']))))
if __name__=='__main__':main()
