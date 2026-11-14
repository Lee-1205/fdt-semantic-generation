"""Prospective mixed ordinary/signed quantities, source fields separate from NN."""
import collections,hashlib,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from prepare_fuzzy_extraction_v73_cpu import CONTEXT,suffixes
from research.frozen_verbalizer_objective_v72 import complete_expression_spans

TRAIN_NAMES=['Cyrus','Emrys','Fayruz','Cruz','Azriel','Eliza','Zayden','Yasmin']
DEV_NAMES=['Alaric','Blythe','Cyrene','Denzel']
ITEMS=['amber tiles','cotton bags','steel rings','paper envelopes']
COLORS=['orange','white','brown','teal']

def make_rows(names,seed,held=False):
    rng=random.Random(seed);suf=suffixes();features=[];targets=[]
    for mode in ('unsigned','signed'):
        roles=range(4 if held else 16) if mode=='unsigned' else range(2 if held else 8)
        for ni,actor in enumerate(names):
            for role in roles:
                peer=names[(ni+3)%len(names)];item=ITEMS[(ni+role)%4];color=COLORS[(2*ni+role)%4]
                episode=1000+ni*173+role*23;reserve=151+(ni+role)%39 if mode=='unsigned' else 1+(ni+role)%19
                for unit_index in range(1 if mode=='unsigned' else 2):
                    unit=('kg','degrees C')[unit_index];family=(ni+role)%2
                    qbase=2*((ni*16+role)%50) if mode=='unsigned' else (ni*29+role*7+unit_index*13)%100
                    clauses=[];values=[]
                    for member in range(2):
                        q=qbase+member if mode=='unsigned' else qbase;sign=('+','-')[member] if mode=='signed' else ''
                        value=f'{q}' if mode=='unsigned' else f'{sign}{q}.75 {unit}'
                        if mode=='unsigned':
                            if held:
                                clause=(f'Inventory note {episode} states that {actor} delivered {q} {item}; {peer} retained {reserve} {item}, none of which were delivered.' if family==0 else
                                    f'{peer} verified entry {episode} after {actor} set aside {reserve} {item}; {actor} then packed {q} {item} for the {color} room.')
                            else:
                                clause=(f'Record {episode}: {actor} packed {q} {item} before {peer} locked the {color} cabinet; {reserve} {item} stayed inside and were not taken.' if family==0 else
                                    f'After {peer} checked inventory {episode}, {actor} sent {q} {item} to the {color} room, leaving {reserve} {item} with {peer}.')
                            offset=re.search(r'(?<!\d)'+re.escape(f'{q} {item}'),clause).start()
                        else:
                            if held:
                                clause=f"{actor} noted a change of {value}, whereas {peer}'s separate stock weighed +{reserve}.25 kg."
                            elif family==0:
                                clause=f'Under entry {episode}, {actor} measured a shift of {value}; {peer} stored +{reserve}.25 kg of rice and did not move the spare parcels.'
                            else:
                                clause=f'After {peer} confirmed +{reserve}.25 kg of rice in stock, {actor} reported a shift of {value} under entry {episode}; the spare parcels were not moved.'
                            offset=clause.index(value)
                        clauses.append(clause);values.append((q,sign,value,offset))
                    assert values[0][3]==values[1][3]
                    requested=(role*5+ni*3+unit_index)%16 if held else (role+ni*3+unit_index*7)%16
                    context=''.join(rng.choice(CONTEXT) for _ in range((ni*3+role*2+unit_index)%12))
                    context+=rng.choice(suf[(requested-len(context.encode())-values[0][3])%16])
                    pair='retention75:'+hashlib.sha256((context+actor+peer+mode+str(unit_index)+str(seed)).encode()).hexdigest()[:24]
                    for member,(clause,(q,sign,value,offset)) in enumerate(zip(clauses,values)):
                        text=context+clause;identity='retention75:'+hashlib.sha256(text.encode()).hexdigest()[:24]
                        spans=[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b',text)]
                        features.append(dict(id=identity,prompt=text));targets.append(dict(id=identity,pair_id=pair,member=member,mode=mode,family=family,actor=actor,peer=peer,
                            primary_value=q,sign=sign,unit=unit if mode=='signed' else 'items',episode=episode,reserve=reserve,item=item,color=color,
                            clause_start=len(context.encode()),clause=clause,primary_start=len(context.encode())+offset,primary_text=value,primary_position=requested,
                            source_kind='prospective_mixed_retention',critical_spans=complete_expression_spans(text,spans)))
    return features,targets

def main():
    out=ROOT/'research/artifacts/critical_retention_20261112_v75_cpu';assert not out.exists()
    train,targets=make_rows(TRAIN_NAMES,75112);dev,dt=make_rows(DEV_NAMES,75113,True)
    old=ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2';prior=io.load_json(old/'train_features.json');pt=io.load_json(old/'train_targets.json')
    train.extend(prior[512:]);targets.extend(pt[512:]);assert len(train)==640 and len(dev)==64
    held=io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/short_features.json')+io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/long_features.json')+io.load_json(old/'dev_features.json')+dev
    assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in held for r in train)
    forbidden=DEV_NAMES+['Zephyr','Boaz','Ilyaz','Kyrene','Anya','Ezrien','Ariadne','Calista','Dashiell','Esme','Idris','Noemi','Inez','Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette']
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in train for n in forbidden)
    groups=collections.defaultdict(list)
    for i,t in enumerate(targets):groups[t.get('pair_id',t['id'])].append(i)
    ordered=list(groups.values());random.Random(75112).shuffle(ordered);order=[[i] for g in ordered for i in g]
    assert len({r['prompt'] for r in train})==640
    cells=collections.Counter(t['sign']+t['unit'] for t in targets[:512] if t['mode']=='signed');assert cells==dict.fromkeys(['+kg','-kg','+degrees C','-degrees C'],64)
    manifest=dict(schema='critical_retention_v75_CPU_data',TRAIN_rows=640,new_authored_rows=512,unsigned_rows=256,signed_rows=256,minimal_pairs=256,
        preserved_OWT_TRAIN128=True,DEV_rows=64,DEV_unsigned32_signed32=True,TRAIN_names=TRAIN_NAMES,DEV_names=DEV_NAMES,
        primary_signed_unit_counts=dict(cells),target_UTF8_bytes=sum(len(r['prompt'].encode()) for r in train),critical_bytes=sum(s['end']-s['start'] for t in targets for s in t['critical_spans']),
        max_input_bytes=max(len(r['prompt'].encode()) for r in train),finite_epochs=1,features=['id','prompt'],gold_and_pair_loss_score_only=True,
        source_encoder='immutable V74/V73 upstream; no source/gold bypass to readout',old_held_actual_prompts_names_NOT_TRAIN=True,E2_registry_access=0,downloads=0,actual_GPU_optimizer_trial=False,
        limitations='Prospective authored mixed numerical contexts after observed V74 interference; correlated pairs/templates; same existing128 OWT TRAIN; newDEV untrained authored names/surfaces, not a blind natural corpus or semantic certificate.')
    out.mkdir()
    for name,value in [('train_features',train),('train_targets',targets),('dev_features',dev),('dev_targets',dt),('orders',order),('manifest',manifest)]:io.atomic_json(out/(name+'.json'),value)
    io.atomic_json(out/'builder_snapshot_sha.json',dict(file=str(Path(__file__)),SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(manifest))
if __name__=='__main__':main()
