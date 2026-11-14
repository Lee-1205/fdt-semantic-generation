"""Offline byte-aligned event/field audit; prevents regex crossing events.

Output byte length equality and fixed-position reconstruction were established
from saved raw scores. Source offsets are used ONLY for POSTHOC comparison,
never for model features or output correction. No NN or GPU calls.
"""
import collections,difflib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN,CORE

def main():
    run=ROOT/'runs/long_extrapolation_20261112_v69';data=ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu'
    features=io.load_json(data/'features.json');targets=io.load_json(data/'targets.json');saved=io.load_json(run/'reconstructions.json')
    first=io.load_json(run/'root_saved_review.json');assert first['all_raw_SHA_target_argmax_UTF8_FSM_verified']
    rows=[];cards=[];lexical=collections.Counter();field_counts=collections.Counter();samples=[];all_changes=[];rawerrors=0
    for i,(f,t,r) in enumerate(zip(features,targets,saved)):
        sb=f['prompt'].encode();ob=r['output'].encode();assert len(sb)==len(ob)==t['actual_bytes']
        good=metadata_only=core_risks=form_only=boundary_good=0;region=collections.Counter();char_errors=sum(a!=b for a,b in zip(sb,ob));rawerrors+=char_errors
        for j,e in enumerate(t['events']):
            start,end=e['byte_start'],e['byte_end'];source=sb[start:end].decode();out=ob[start:end].decode('utf8',errors='replace');m=PATTERN.fullmatch(source);assert m
            changes={};field_covered=set()
            for k in m.groupdict():
                a,b=m.span(k)
                if a<0:continue
                field_covered.update(range(a,b))
                old=source[a:b];new=ob[start+a:start+b].decode('utf8',errors='replace')
                if old.casefold().strip()!=new.casefold().strip():
                    changes[k]=dict(source=old,output=new);lexical[(k,old,new)]+=1;field_counts[k]+=1
            original=source.encode();generated=ob[start:end]
            other=[q for q,(a,b) in enumerate(zip(original,generated)) if a!=b and q not in field_covered]
            core={k:v for k,v in changes.items() if k!='episode'}
            label='content_field_risk' if core else 'relation_word_or_form_review' if other else 'metadata_only' if changes else 'core_and_relation_surface_preserved'
            if label in ('core_and_relation_surface_preserved','metadata_only'):good+=1
            if label=='metadata_only':metadata_only+=1
            if label=='content_field_risk':core_risks+=1
            if label=='relation_word_or_form_review':form_only+=1
            cross=start//2048!=(end-1)//2048
            if cross and label in ('core_and_relation_surface_preserved','metadata_only'):boundary_good+=1
            part='first_third' if start<len(sb)/3 else 'middle_third' if start<2*len(sb)/3 else 'last_third'
            region[part+'_events']+=1;region[part+'_risk']+=label in ('content_field_risk','relation_word_or_form_review')
            card=dict(row=i,ordinal=j+1,episode=e['episode'],label=label,changed_fields=changes,other_word_changes=other,
                crosses_compute_boundary=cross,region=part,source=source,output=out)
            cards.append(card)
            if label not in ('core_and_relation_surface_preserved','metadata_only'):all_changes.append(card)
            if j in (0,len(t['events'])//2,len(t['events'])-1):samples.append(card)
        rows.append(dict(row=i,style=t['style'],source_bytes=len(sb),times_training_max=len(sb)/16384,events=len(t['events']),
            core_and_relation_surface_preserved=good,metadata_only=metadata_only,content_field_risks=core_risks,relation_word_or_form_review=form_only,
            compute_boundary_events=t['boundary_crossing_events'],compute_boundary_core_surface_preserved=boundary_good,
            decoded_byte_errors_auxiliary=char_errors,decoded_byte_accuracy_auxiliary=1-char_errors/len(sb),regions=dict(region)))
    pairs=[dict(field=k,source=a,output=b,count=n) for (k,a,b),n in lexical.most_common()]
    report=dict(status='ROOT_V69_MEANING_ALIGNED_V2_REVIEW',rows=rows,events=len(cards),
        core_and_relation_surface_preserved=sum(r['core_and_relation_surface_preserved'] for r in rows),
        content_field_risks=sum(r['content_field_risks'] for r in rows),relation_word_or_form_review=sum(r['relation_word_or_form_review'] for r in rows),
        field_change_counts=dict(field_counts),decoded_byte_errors_auxiliary=rawerrors,all_observed_output_byte_lengths_match=True,
        no_output_correction=True,source_offsets_OFFLINE_SCORE_ONLY=True,extra_model_calls=0,
        earlier_regex_review='raw/SHA/argmax/UTF8/parity checks correct; unbounded object regex could bridge adjacent events on damaged words, inflating event risks. Preserve revision1, use bounded posthoc aligned fields and human reading; no GPU repeat.',
        limitation='surface-field equality is restricted fidelity evidence, not independent general semantic or abstract-memory success; spelling errors require unblinded human reading; variable local memory retained')
    io.atomic_json(run/'root_meaning_review_v2.json',report);io.atomic_json(run/'root_meaning_event_cards_v2.json',cards)
    io.atomic_json(run/'root_reading_changes_v2.json',dict(all_risks=all_changes,unique_field_changes=pairs,region_samples=samples))
    io.atomic_json(run/'root_CPU_reviewer_incident.json',dict(issue='revision1 unconstrained wildcard matched across damaged neighboring records',model_results_unchanged=True,extra_model_calls=0,fix='bounded event byte ranges used only in saved-output scoring; keep all prior files',revision2='root_meaning_review_v2.json'))
    print(json.dumps(report));print(json.dumps(pairs));print(json.dumps([c for c in all_changes if c['label']=='relation_word_or_form_review'][:16]))
if __name__=='__main__':main()
