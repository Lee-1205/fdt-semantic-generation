"""Source-faithfulness comparison on actual decoded words, not TF alone."""
import collections,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN

def long_score(rows,targets,records):
    result=[];cards=[];critical=[]
    for i,(f,t,r) in enumerate(zip(rows,targets,records)):
        source=f['prompt'].encode();output=r['output'].encode();assert len(source)==len(output)
        fields=collections.Counter();preserved=0;word_form=0;name_events=set();number_events=set()
        for j,e in enumerate(t['events']):
            start,end=e['byte_start'],e['byte_end'];text=source[start:end].decode();out=output[start:end].decode('utf8',errors='replace');m=PATTERN.fullmatch(text);assert m
            changes={};covered=set()
            for name in m.groupdict():
                a,b=m.span(name)
                if a<0:continue
                covered.update(range(a,b));old=text[a:b];new=output[start+a:start+b].decode('utf8',errors='replace')
                if old.casefold().strip()!=new.casefold().strip():
                    changes[name]=dict(source=old,output=new);fields[name]+=1
                    if name in ('quantity','reserve'):number_events.add(j);critical.append(dict(row=i,event=j+1,field=name,source=old,output=new))
                    if name in ('actor','checker','negative_actor'):name_events.add(j)
            other=any(a!=b and k not in covered for k,(a,b) in enumerate(zip(text.encode(),output[start:end])))
            core={k:v for k,v in changes.items() if k!='episode'}
            preserved+=not core and not other;word_form+=bool(other)
            if core or other:cards.append(dict(row=i,event=j+1,changes=changes,other_word_or_form=other,source=text,output=out))
        result.append(dict(row=i,source_bytes=len(source),events=len(t['events']),fields_changed=dict(fields),
            core_relation_surface_preserved=preserved,numeric_risk_events=len(number_events),name_risk_events=len(name_events),word_or_form_review_events=word_form,
            decoded_byte_errors=sum(a!=b for a,b in zip(source,output))))
    return dict(rows=result,numeric_changed_fields=sum(sum(r['fields_changed'].get(k,0) for k in ('quantity','reserve')) for r in result),
        name_changed_fields=sum(sum(r['fields_changed'].get(k,0) for k in ('actor','checker','negative_actor')) for r in result),numeric_examples=critical),cards

def main():
    run=ROOT/'runs/factual_cycle_20261112_v70';p=io.load_json(run/'plan.json');data=Path(p['data']);raw=io.load_json(run/'root_saved_review.json');assert raw['status']=='ROOT_V70_SAVED_REVIEW_PASS'
    rows=io.load_json(Path(p['long_data'])/'features.json')[-2:];targets=io.load_json(Path(p['long_data'])/'targets.json')[-2:]
    before,oldcards=long_score(rows,targets,io.load_json(Path(p['baseline_long_run'])/'reconstructions.json')[-2:])
    after,newcards=long_score(rows,targets,io.load_json(run/'long_reconstructions.json'))
    rows=io.load_json(data/'dev_features.json');labels=io.load_json(data/'dev_targets.json');saved=io.load_json(run/'new_English_reconstructions.json');new_errors=[];counts=collections.Counter()
    for i,(f,t,r) in enumerate(zip(rows,labels,saved)):
        source=f['prompt'].encode();output=r['output'].encode();assert len(source)==len(output)
        for s in t['critical_spans']:
            counts[s['kind']+'_fields']+=1;a,b=s['start'],s['end'];word=output[a:b].decode('utf8',errors='replace')
            if word.casefold()!=s['text'].casefold():new_errors.append(dict(row=i,field=s['kind'],source=s['text'],output=word,source_sentence=f['prompt'],output_sentence=r['output']))
    retained=io.load_json(run/'retained_English_reconstructions.json');old=io.load_json(Path(p['baseline_run'])/'retained_English_reconstructions.json');retention_changes=[]
    for i,(a,b) in enumerate(zip(old,retained)):
        if a['output']!=b['output']:
            sb=b['source'].encode();ob=b['output'].encode();ab=a['output'].encode();diff=[]
            for k in range(min(len(sb),len(ob))):
                if sb[k]!=ob[k] and len(diff)<12:diff.append(dict(position=k,source=sb[max(0,k-40):k+50].decode('utf8',errors='replace'),old=ab[max(0,k-40):k+50].decode('utf8',errors='replace'),new=ob[max(0,k-40):k+50].decode('utf8',errors='replace')))
            retention_changes.append(dict(row=i,id=b['id'],source_bytes=len(sb),new_differences=diff,source=b['source'] if len(sb)<300 else None,output=b['output'] if len(sb)<300 else None))
    report=dict(status='ROOT_V70_MEANING_REVIEW_READY_FOR_READING',long_before=before,long_after=after,new_English_32_critical_field_counts=dict(counts),
        new_English_32_critical_field_changes=new_errors,retained_English_output_changes=retention_changes,
        scope='restricted offline source fields; typo versus meaning harm requires ROOT reading; no abstract-state/general-semantic accuracy claim; evaluated words not repaired',
        extra_model_calls=0,Exact_entries=0,copy_bytes=0,denoise_calls=0)
    io.atomic_json(run/'root_meaning_review.json',report);io.atomic_json(run/'root_long_meaning_risks.json',dict(before=oldcards,after=newcards))
    print(json.dumps(report))
    # All post-training numeric risk contexts, short enough to read completely.
    print(json.dumps([c for c in newcards if any(k in c['changes'] for k in ('quantity','reserve','actor','checker','negative_actor'))]))
if __name__=='__main__':main()
