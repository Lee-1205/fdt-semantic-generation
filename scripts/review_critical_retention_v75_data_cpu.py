"""Reparse all authored facts/offsets, original OWT identity and finite pairs."""
import collections,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io

UNSIGNED=[
 r'Record (?P<episode>\d+): (?P<actor>\w+) packed (?P<q>\d+) (?P<item>.+?) before (?P<peer>\w+) locked the (?P<color>\w+) cabinet; (?P<reserve>\d+) (?P=item) stayed inside and were not taken\.',
 r'After (?P<peer>\w+) checked inventory (?P<episode>\d+), (?P<actor>\w+) sent (?P<q>\d+) (?P<item>.+?) to the (?P<color>\w+) room, leaving (?P<reserve>\d+) (?P=item) with (?P=peer)\.',
 r'Inventory note (?P<episode>\d+) states that (?P<actor>\w+) delivered (?P<q>\d+) (?P<item>.+?); (?P<peer>\w+) retained (?P<reserve>\d+) (?P=item), none of which were delivered\.',
 r'(?P<peer>\w+) verified entry (?P<episode>\d+) after (?P<actor>\w+) set aside (?P<reserve>\d+) (?P<item>.+?); (?P=actor) then packed (?P<q>\d+) (?P=item) for the (?P<color>\w+) room\.'
]
SIGNED=[
 r'Under entry (?P<episode>\d+), (?P<actor>\w+) measured a shift of (?P<primary>(?P<sign>[+-])(?P<q>\d+)\.75 (?P<unit>kg|degrees C)); (?P<peer>\w+) stored \+(?P<reserve>\d+)\.25 kg of rice and did not move the spare parcels\.',
 r'After (?P<peer>\w+) confirmed \+(?P<reserve>\d+)\.25 kg of rice in stock, (?P<actor>\w+) reported a shift of (?P<primary>(?P<sign>[+-])(?P<q>\d+)\.75 (?P<unit>kg|degrees C)) under entry (?P<episode>\d+); the spare parcels were not moved\.',
 r"(?P<actor>\w+) noted a change of (?P<primary>(?P<sign>[+-])(?P<q>\d+)\.75 (?P<unit>kg|degrees C)), whereas (?P<peer>\w+)'s separate stock weighed \+(?P<reserve>\d+)\.25 kg\."
]

def main():
    d=ROOT/'research/artifacts/critical_retention_20261112_v75_cpu';rows=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');dev=io.load_json(d/'dev_features.json');dt=io.load_json(d/'dev_targets.json');m=io.load_json(d/'manifest.json');orders=io.load_json(d/'orders.json')
    assert len(rows)==len(labels)==640 and len(dev)==len(dt)==64 and sorted(v[0] for v in orders)==list(range(640)) and all(len(o)==1 for o in orders)
    groups=collections.defaultdict(list);positions=collections.defaultdict(set);spans_count=collections.Counter();where={o[0]:i for i,o in enumerate(orders)}
    for i,(row,t) in enumerate(zip(rows+dev,labels+dt)):
        assert set(row)=={'id','prompt'} and row['id']==t['id'];b=row['prompt'].encode();prev=-1
        for span in t['critical_spans']:
            a,e=span['start'],span['end'];assert prev<=a<e<=len(b) and b[a:e].decode()==span['text'];prev=e;spans_count[span['kind']]+=1
        if 512<=i<640:continue
        assert b[t['clause_start']:].decode()==t['clause'] and b[t['primary_start']:t['primary_start']+len(t['primary_text'].encode())].decode()==t['primary_text']
        patterns=UNSIGNED if t['mode']=='unsigned' else SIGNED
        matches=[v for pattern in patterns if (v:=re.fullmatch(pattern,t['clause']))]
        assert len(matches)==1;match=matches[0];g=match.groupdict()
        for field in ('actor','peer','item','color','sign','unit'):
            if field in g:assert g[field]==t[field],(i,field)
        for field,target in [('q','primary_value'),('reserve','reserve'),('episode','episode')]:
            if field in g:assert int(g[field])==t[target]
        offset=match.start('q' if t['mode']=='unsigned' else 'primary');assert t['primary_start']==t['clause_start']+offset and t['primary_start']%16==t['primary_position']
        expected_names={(x.start(),x.end(),x.group()) for x in re.finditer(r'\b(?:'+'|'.join(m['DEV_names'] if i>=640 else m['TRAIN_names'])+r')\b',row['prompt'])}
        actual_names={(s['start'],s['end'],s['text']) for s in t['critical_spans'] if s['kind']=='name'};assert expected_names==actual_names
        expected_numbers={(x.start(),x.end(),x.group()) for x in re.finditer(r'[+-]?\d+(?:\.\d+)?(?: degrees C| kg)?',row['prompt'])}
        actual_numbers={(s['start'],s['end'],s['text']) for s in t['critical_spans'] if s['kind']=='number_expression'};assert expected_numbers==actual_numbers,(i,expected_numbers^actual_numbers)
        if i<512:groups[t['pair_id']].append(i);positions[t['mode']+t['sign']+(t['unit'] if t['mode']=='signed' else '')].add(t['primary_position'])
    for ix in groups.values():
        assert len(ix)==2 and [labels[k]['member'] for k in ix]==[0,1] and abs(where[ix[0]]-where[ix[1]])==1
        a,b=(labels[k] for k in ix);fa,fb=(rows[k]['prompt'] for k in ix)
        assert a['primary_start']==b['primary_start']
        assert fa[:a['primary_start']]+fa[a['primary_start']+len(a['primary_text']):]==fb[:b['primary_start']]+fb[b['primary_start']+len(b['primary_text']):]
        assert (a['primary_value']+1==b['primary_value']) if a['mode']=='unsigned' else (a['primary_value']==b['primary_value'] and (a['sign'],b['sign'])==('+','-'))
    assert len(groups)==256 and all(len(v)==16 for v in positions.values())
    old=ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2';assert rows[512:]==io.load_json(old/'train_features.json')[512:] and labels[512:]==io.load_json(old/'train_targets.json')[512:]
    held=io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/short_features.json')+io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/long_features.json')+io.load_json(old/'dev_features.json')+dev
    assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in held for r in rows)
    forbidden=m['DEV_names']+['Zephyr','Boaz','Ilyaz','Kyrene','Anya','Ezrien','Ariadne','Calista','Dashiell','Esme','Idris','Noemi','Inez','Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette']
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in rows for n in forbidden)
    assert sum(len(r['prompt'].encode()) for r in rows)==m['target_UTF8_bytes']==969013
    assert sum(s['end']-s['start'] for t in labels for s in t['critical_spans'])==m['critical_bytes']==29114
    proof=dict(status='ROOT_V75_MIXED_DATA_CPU_PASS',all704_rows_source_target_UTF8_checked=True,all576_authored_whole_clauses_independently_reparsed=True,
        primary_and_secondary_numbers_actor_peer_unit_episode_checked=True,substring_digit_offset_prebuild_bug_fixed=True,original_failed_builder_preserved='../outputs/FDT_V75_CPU_PREBUILD_REV1_20261112.py',
        all256pairs_only_primary_changes=True,pair_adjacency_and_finite640order=True,primary_context_positions={k:sorted(v) for k,v in positions.items()},
        original128_OWT_TRAIN_features_targets_exact=True,all_actual_old_held_prompts_names_and_newDEV_NOT_TRAIN=True,model_calls=0,E2_registry_access=0,
        files_SHA={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.glob('*.json')},limitations=m['limitations'])
    target=d/'root_data_review.json';assert not target.exists();io.atomic_json(target,proof);print(json.dumps({k:v for k,v in proof.items() if k!='files_SHA'}))
if __name__=='__main__':main()
