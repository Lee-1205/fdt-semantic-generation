"""Prospective TRAIN coverage; no held prompts/gold imported into TRAIN."""
import collections,hashlib,itertools,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
NAMES=['Alden','Brenna','Celeste','Dorian','Emilia','Finley','Greta','Harlan','Isolde','Jasper','Keziah','Lazlo','Meredith','Nolan','Orla','Pascal','Quentin','Rosalie','Silas','Thalia','Ulric','Verena','Willa','Xavier','Yara','Zelda']
DEV_NAMES=['Adara','Beatrix','Corwin','Delphine','Esmond','Iria','Jolene','Zorion']
CONTEXT=['The storage room was quiet. ','The form was ready. ','A clerk checked the door. ','The morning round had ended. ','A lamp lit the desk. ','No other entry had been changed. ','The label remained attached. ','The report was kept for review. ']

def suffixes():
    choices=collections.defaultdict(list)
    for n in range(1,4):
        for parts in itertools.product(CONTEXT,repeat=n):
            s=''.join(parts);choices[len(s.encode())%16].append(s)
    assert len(choices)==16
    return choices

def authored(names,seed,held=False):
    rng=random.Random(seed);byremainder=suffixes();features=[];targets=[]
    for ni,actor in enumerate(names):
        roles=range(4) if held else range(16)
        for requested_role in roles:
            role=(requested_role*5+3)%16 if held else requested_role
            recipient=names[(ni+3)%len(names)];family=ni%4
            long_context=''.join(rng.choice(CONTEXT) for _ in range((ni*3+role)%40))
            remainder=(role-len(long_context.encode()))%16;prefix=long_context+rng.choice(byremainder[remainder])
            assert len(prefix.encode())%16==role
            pair='pair73:'+hashlib.sha256((prefix+actor+recipient+str(seed)).encode()).hexdigest()[:24]
            for member in ((ni+role)%2,) if held else (0,1):
                q=2*((ni*16+role)%50)+member;reserve=2+(ni+role)%6;clock=f'{3+(ni%19):02d}:{7+(role%5)*10:02d}'
                family_clauses=[
                    f'{actor} recorded +{q}.45 degrees C at {clock}; {recipient} kept -{reserve}.80 kg in the green container. Neither person changed the label.',
                    f'{actor} paid ${100+q},{104+role:03d}.25 to {recipient} on 2024-{1+ni%12:02d}-{1+role%28:02d}; the refund was -${reserve}.80. The earlier amount was not paid twice.',
                    f'{actor} moved {q} blue folders to {recipient} before the bell; {recipient} kept {reserve} folders and did not move them.',
                    f'{actor} signed a note: {recipient} kept {q} wooden tokens after the meeting, and {actor} did not take the remaining {reserve} tokens.',
                ]
                clause=family_clauses[family]
                if held:
                    # A new surface wrapper around newly authored facts, not
                    # any previous evaluation prompt or gold imported as TRAIN.
                    text=prefix+clause+' This entry describes the later inspection.'
                else:text=prefix+clause
                identity='extract73:'+hashlib.sha256(text.encode()).hexdigest()[:24]
                spans=[dict(start=m.start(),end=m.end(),kind='name',text=m.group()) for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b',text)]
                features.append(dict(id=identity,prompt=text));targets.append(dict(id=identity,source_kind='prospective_authored',pair_id=pair,member=member,
                    family=family,actor=actor,recipient=recipient,primary_value=q,reserve=reserve,clause_start=len(prefix.encode()),clause=clause,
                    actor_position_role=role,critical_spans=complete_expression_spans(text,spans)))
    return features,targets

def main():
    out=ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2';out.mkdir(exist_ok=False)
    train,labels=authored(NAMES,73112);dev,devlabels=authored(DEV_NAMES,73113,True)
    old=ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu';oldrows=io.load_json(old/'train_features.json');oldlabels=io.load_json(old/'train_targets.json')
    for row,t in zip(oldrows[384:512],oldlabels[384:512]):
        train.append(row);labels.append(dict(id=row['id'],source_kind='preserved_OWT_TRAIN',critical_spans=t['critical_spans']))
    assert len(train)==len(labels)==960 and len(dev)==32 and len({r['prompt'] for r in train})==960
    held=[]
    for file in [ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json',ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu/features.json',
        ROOT/'research/artifacts/factual_cycle_20261112_v70_cpu_v2/dev_features.json',old/'dev_features.json']:
        held.extend(io.load_json(file))
    assert not any(h['prompt'] in r['prompt'] for h in held for r in train)
    assert not {r['prompt'] for r in train}&{r['prompt'] for r in dev}
    assert not any(re.search(r'\b'+name+r'\b',r['prompt']) for r in train for name in DEV_NAMES+['Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette','Inez'])
    groups=collections.defaultdict(list)
    for i,t in enumerate(labels):groups[t.get('pair_id',t['id'])].append(i)
    order=list(groups.values());random.Random(73112).shuffle(order);order=[[i] for group in order for i in group]
    coverage=collections.defaultdict(collections.Counter)
    for r,t in zip(train,labels):
        b=r['prompt'].encode()
        for s in t['critical_spans']:
            for k in range(s['start'],s['end']):coverage[chr(b[k])][str(k%16)]+=1
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZz+:':assert len(coverage[char])==16,char
    manifest=dict(schema='fuzzy_extraction_v73_CPU_data',TRAIN_rows=960,new_authored_rows=832,complete_minimal_pairs=416,OWT_original_TRAIN_rows=128,new_DEV_rows=32,
        finite_epochs=1,proposed_attempts=960,source_UTF8_bytes=sum(len(r['prompt'].encode()) for r in train),max_input_bytes=max(len(r['prompt'].encode()) for r in train),
        critical_bytes=sum(s['end']-s['start'] for t in labels for s in t['critical_spans']),TRAIN_names=NAMES,new_DEV_names=DEV_NAMES,
        all_capital_letters_and_lower_z_plus_clock_colon_in_critical_all16_positions=True,features=['id','prompt'],pair_and_facts_spans_loss_score_only=True,
        ordering='416 pairs adjacent two single-row attempts; 128 old OWT rows once; one finite shuffle',primary_minimal_pair_values='0..99, even/odd pairs; currency leading group100+q',
        actual_candidate_engine_plan_optimizer_GPU=False,E2_registry_access=0,new_downloads=0,
        limitations='authoring after observed V72 problems, correlated templates/pairs and repeated prefix contexts; not832 independent documents or blind benchmark; old TRAIN reused; no performance result')
    for name,value in [('train_features',train),('train_targets',labels),('dev_features',dev),('dev_targets',devlabels),('orders',order),('coverage',coverage),('manifest',manifest)]:io.atomic_json(out/(name+'.json'),value)
    io.atomic_json(out/'builder_snapshot_sha.json',dict(file=str(Path(__file__)),SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(manifest))
if __name__=='__main__':main()
