"""Cross sign/unit/context and name-internal position without held-name leakage."""
import collections,hashlib,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from prepare_fuzzy_extraction_v73_cpu import suffixes,CONTEXT
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
TRAIN_NAMES=['Cyrus','Emrys','Fayruz','Cruz','Azriel','Eliza','Zayden','Yasmin']
DEV_NAMES=['Ilyaz','Kyrene','Anya','Ezrien']

def build(names,seed,held=False):
    rng=random.Random(seed);suffix=suffixes();rows=[];labels=[]
    for ni,a in enumerate(names):
        for role in range(4 if held else 16):
            requested=(role*5+3)%16 if held else role
            context=''.join(rng.choice(CONTEXT) for _ in range((ni*7+role)%20))
            context+=rng.choice(suffix[(requested-len(context.encode()))%16])
            for unit_index,unit in enumerate(('kg','degrees C')):
                q=(ni*29+role*7+unit_index*13)%100;left=1+(ni+role+unit_index)%19;peer=names[(ni+1+unit_index)%len(names)]
                pair='signed74:'+hashlib.sha256((context+a+peer+unit+str(seed)).encode()).hexdigest()[:24]
                for sign in ('+','-'):
                    if held:
                        phrase=f"{a}'s log lists a change of {sign}{q}.50 {unit}. {peer} confirmed the reading, while the stored grain weighed +{left}.25 kg."
                    else:
                        kind='mass' if unit=='kg' else 'temperature'
                        phrase=f'{a} recorded a signed {kind} change of {sign}{q}.50 {unit} after {peer} checked the balance. {peer} kept +{left}.25 kg of grain; the remaining {left+2} parcels were not moved.'
                    text=context+phrase;identity='context74:'+hashlib.sha256(text.encode()).hexdigest()[:24]
                    spans=[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b',text)]
                    rows.append(dict(id=identity,prompt=text));labels.append(dict(id=identity,pair_id=pair,sign=sign,unit=unit,quantity=q,stored_quantity=left,actor=a,peer=peer,
                        clause_start=len(context.encode()),clause=phrase,actor_position_role=requested,source_kind='new_authored_context_cross',critical_spans=complete_expression_spans(text,spans)))
    return rows,labels

def main():
    out=ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2';out.mkdir(exist_ok=False)
    train,targets=build(TRAIN_NAMES,74112);dev,dt=build(DEV_NAMES,74113,True)
    old=ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2';oldrows=io.load_json(old/'train_features.json');oldtargets=io.load_json(old/'train_targets.json')
    train.extend(oldrows[832:]);targets.extend(oldtargets[832:]);assert len(train)==640 and len(dev)==64
    groups=collections.defaultdict(list)
    for i,t in enumerate(targets):groups[t.get('pair_id',t['id'])].append(i)
    groups=list(groups.values());random.Random(74112).shuffle(groups);orders=[[i] for group in groups for i in group]
    held=io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/short_features.json')+io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/long_features.json')
    assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in held for r in train)
    assert len({r['prompt'] for r in train})==640 and not {r['prompt'] for r in train}&{r['prompt'] for r in dev}
    forbidden=DEV_NAMES+['Zephyr','Boaz','Ariadne','Calista','Dashiell','Esme','Idris','Noemi','Inez','Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette']
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in train for n in forbidden)
    primary=collections.Counter(t['sign']+t['unit'] for t in targets[:512]);assert set(primary.values())=={128}
    m=dict(schema='context_verbalizer_v74_CPU_data',TRAIN_rows=640,new_authored512=True,pairs256=True,preserved_OWT_TRAIN128=True,new_DEV_rows=64,finite_epochs=1,
        TRAIN_names=TRAIN_NAMES,DEV_names=DEV_NAMES,primary_signed_unit_counts=dict(primary),
        target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in train),critical_bytes=sum(s['end']-s['start'] for t in targets for s in t['critical_spans']),
        max_input_bytes=max(len(r['prompt'].encode()) for r in train),features=['id','prompt'],gold_span_pair_position_facts_loss_score_only=True,
        intended_boundary='V73 encoder+binding fixed; fresh final7readout learning on actual bound Commit',actual_runner_plan_optimizer_GPU=False,E2_registry_access=0,new_downloads=0,
        limitations='prospective correlated sign pairs and authored names after observed problems; known held prompts/names excluded; no performance result or blind natural benchmark')
    for name,value in [('train_features',train),('train_targets',targets),('dev_features',dev),('dev_targets',dt),('orders',orders),('manifest',m)]:io.atomic_json(out/(name+'.json'),value)
    print(json.dumps(m))
if __name__=='__main__':main()
