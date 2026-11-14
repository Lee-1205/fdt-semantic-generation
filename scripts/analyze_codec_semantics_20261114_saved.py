"""Saved text/metadata calculations only; stdlib, no Torch/NN/optimizer."""
import ast
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

P=Path(__file__).resolve().parents[1]
W=P.parent
OUT=W/'outputs/FDT_CODEC_SEMANTICS_DENOISER_FOLLOWUP_20261114_evidence'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    OUT.mkdir(exist_ok=False)
    run=P/'runs/critical_repair_20261113_v88'
    plan=read(run/'plan.json');source=[x for x in read(plan['eval_inputs']) if x['panel']=='long']
    panels=read(run/'root_decoded_panels.json')['final_long']
    # Extract the published, pre-existing scoring grammar without importing its
    # module (which imports Torch). This grammar NEVER becomes an NN feature.
    scorer=P/'scripts/review_long_extrapolation_v69_saved_cpu.py'
    tree=ast.parse(scorer.read_text(encoding='utf-8-sig'))
    expression=next(n.value.args[0].value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PATTERN' for t in n.targets))
    pattern=re.compile(expression,re.I)
    totals=collections.Counter();characters=[];affected=[];tuples=[];documents=[]
    for doc,(entry,saved) in enumerate(zip(source,panels)):
        text=entry['row']['prompt'];out=saved['output']
        assert text.isascii() and out.isascii() and len(text)==len(out) and entry['row']['id']==saved['id']
        events=list(pattern.finditer(text));assert len(events) in (1356,1362)
        doc_errors=[]
        for m in events:
            event_id=int(m.group('episode'));changes={};spans={}
            for field in pattern.groupindex:
                a,b=m.span(field)
                if a<0:continue
                spans[field]=(a,b);totals[field+'_total']+=1
                if text[a:b]!=out[a:b]:
                    changes[field]=dict(source=text[a:b],output=out[a:b]);totals[field+'_bad']+=1
            extra=dict(predicate='moved',closed_predicate='closed',contained_predicate='contained',
                negated_predicate='remove',source_kind='shelf',destination_kind='drawer',gate_kind='gate',box_kind='box',episode_keyword='episode')
            for field,word in extra.items():
                sm=re.search(r'\b'+word+r'\b',m.group(),re.I);assert sm
                a,b=m.start()+sm.start(),m.start()+sm.end();spans[field]=(a,b);totals[field+'_total']+=1
                if text[a:b]!=out[a:b]:changes[field]=dict(source=text[a:b],output=out[a:b]);totals[field+'_bad']+=1
            err=[i for i in range(m.start(),m.end()) if text[i]!=out[i]]
            core_changes={k:v for k,v in changes.items() if k!='episode_keyword'}
            totals['events']+=1;totals['events_with_any_byte_error']+=bool(err)
            totals['events_with_selected_field_error']+=bool(core_changes)
            totals['events_strict_grammar_parseable']+=pattern.fullmatch(out[m.start():m.end()]) is not None
            tuples.append(dict(doc=doc,episode=event_id,selected_fields_match=not core_changes,
                all_event_bytes_match=not err))
            if err:affected.append(dict(doc=doc,episode=event_id,source=m.group(),output=out[m.start():m.end()],changes=changes,offsets=err))
            for pos in err:
                field=next(k for k,(a,b) in spans.items() if a<=pos<b)
                rec=dict(doc=doc,episode=event_id,offset=pos,byte_role=pos%16,field=field,
                    source=text[pos],output=out[pos],source_context=text[max(0,pos-30):pos+32],
                    output_context=out[max(0,pos-30):pos+32])
                characters.append(rec);doc_errors.append(rec)
        owned_offsets={x['offset'] for x in doc_errors}
        for pos,(a,b) in enumerate(zip(text,out)):
            if a!=b and pos not in owned_offsets:
                rec=dict(doc=doc,episode=None,offset=pos,byte_role=pos%16,field='separator',source=a,output=b,
                    source_context=text[max(0,pos-30):pos+32],output_context=out[max(0,pos-30):pos+32])
                characters.append(rec);doc_errors.append(rec)
        assert len(doc_errors)==saved['decoded_errors']
        documents.append(dict(id=saved['id'],bytes=len(text),events=len(events),byte_errors=len(doc_errors),
            source_SHA=hashlib.sha256(text.encode()).hexdigest(),output_SHA=hashlib.sha256(out.encode()).hexdigest()))
    characters.sort(key=lambda x:(x['doc'],x['offset']))
    assert len(characters)==49 and totals['events']==2718
    old=read(run/'root_meaning_and_rejection_review.json')['long_source_bound_fields']['final']
    for field in ('quantity','reserve','actor','item','order','checker','episode'):
        assert totals[field+'_bad']==old.get(field+'_errors',0)
    assert totals['predicate_bad']==old['predicate_errors'] and totals['destination_kind_bad']==old['location_noun_errors']
    values=[]
    for row in affected:
        for field in ('quantity','reserve'):
            if field in row['changes']:
                v=row['changes'][field];s=v['output'];valid=re.fullmatch(r'[+-]?\d+',s) is not None
                values.append(dict(doc=row['doc'],episode=row['episode'],field=field,**v,
                    output_is_integer=valid,changed_integer_value=valid and int(s)!=int(v['source']),
                    unsigned_integer=re.fullmatch(r'\d+',s) is not None))
    grouped=collections.defaultdict(list)
    for x in characters:grouped[(x['field'],x['source'],x['output'],x['byte_role'])].append(x)
    repeated=[dict(field=k[0],source=k[1],output=k[2],byte_role=k[3],count=len(v)) for k,v in grouped.items()]
    repeated.sort(key=lambda x:-x['count'])
    v103=read(P/'research/artifacts/thought_noise_gap_20261114_v103_cpu_v1/cpu_preflight_owned/runtime/result.json')
    v106=read(P/'research/artifacts/thought_rollout_20261114_v106_cpu_v1/cpu_preflight_owned/runtime/result.json')
    noise_groups=collections.defaultdict(list)
    for r in v103['records']:
        noise_groups[(r['path'],r['t'])].append(r)
    noise=[dict(path=k[0],time=k[1],rows=len(v),byte_CE=sum(x['byte_CE'] for x in v)/len(v),
        EOS_CE=sum(x['EOS_CE'] for x in v)/len(v),exact=sum(x['expected']==x['output'] for x in v),
        prefix=sum(x['target_prefix'] for x in v)) for k,v in noise_groups.items()]
    steps=[]
    for n in (1,4,16):
        rr=[r for r in v106['records'] if r['steps']==n]
        steps.append(dict(steps=n,rows=len(rr),byte_CE=sum(r['byte_CE'] for r in rr)/len(rr),
            unknown_exact=sum(r['label']=='unknown' and r['expected']==r['output'] for r in rr),
            positive_exact=sum(r['label']!='unknown' and r['expected']==r['output'] for r in rr)))
    save('all49_error_contexts.json',characters);save('all_affected_events.json',affected)
    save('all2718_selected_field_comparisons.json',tuples)
    save('numeric_error_interpretation.json',values)
    summary=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        schema='SAVED_ONLY_codec_semantics_and_undertrained_thought_analysis',NNcalls=0,optimizer_steps=0,
        Torch_imported=False,timer_remains_deleted=True,training_remains_paused=True,
        documents=documents,total_bytes=sum(d['bytes'] for d in documents),totals=dict(totals),
        byte_error_classes=dict(collections.Counter(x['field'] for x in characters)),
        selected_field_tuple_matching_events=totals['events']-totals['events_with_selected_field_error'],
        selected_field_tuple_match_percent=100*(1-totals['events_with_selected_field_error']/totals['events']),
        exact_event_byte_match_percent=100*(1-totals['events_with_any_byte_error']/totals['events']),
        numeric_changed_unsigned_integer=sum(v['unsigned_integer'] for v in values),
        numeric_changed_signed_integer=sum(v['output_is_integer'] and not v['unsigned_integer'] for v in values),
        numeric_unparseable=sum(not v['output_is_integer'] for v in values),
        repeated_byte_role_errors=repeated,noise_gap_V103=noise,step_comparison_V106=steps,
        fixed_sampler_algebra=dict(delta=math.pi/32,cos_delta=math.cos(math.pi/32),sin_delta=math.sin(math.pi/32)),
        prior_training_seconds=68.6052985+158.50034976005554,
        no_causal_claim_from_surface_tuple_scores=True,no_abstract_semantics_disproof_from_size=True,
        source_bound_scoring_only_not_inference_features=True,
        no_estimated_numeric_probability_for_undertraining=True)
    save('summary.json',summary)
    pins={}
    for f in sorted(OUT.iterdir()):
        if f.is_file():pins[str(f)]=digest(f)
    for f in (run/'root_decoded_panels.json',run/'root_meaning_and_rejection_review.json',
              Path(plan['eval_inputs']),P/'research/memory_thought_diffusion_v94.py',
              P/'research/ordered_fuzzy_readout_v55.py',Path(__file__)):
        pins[str(f)]=digest(f)
    save('evidence_SHA.json',pins)
    print(json.dumps({k:summary[k] for k in ('total_bytes','totals','byte_error_classes',
        'selected_field_tuple_match_percent','numeric_changed_unsigned_integer','numeric_changed_signed_integer',
        'numeric_unparseable','repeated_byte_role_errors','noise_gap_V103','step_comparison_V106')},ensure_ascii=False))


if __name__=='__main__':main()
