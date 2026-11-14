"""Immutable prospective TRAIN coverage for two independent facts and EOS.

ASCII byte spans/facts/quartets are offline supervision only. No E2 inputs,
new corpus download, model weights, or actual execution plan are accessed.
"""
import collections,hashlib,json,itertools,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from prepare_fuzzy_extraction_v73_cpu import CONTEXT,suffixes
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
OUT=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1'
NAMES=['Cyrus','Emrys','Fayruz','Cruz','Azriel','Eliza','Zayden','Yasmin']
FORBIDDEN=['Alaric','Blythe','Cyrene','Denzel','Zephyr','Boaz','Ilyaz','Kyrene','Anya','Ezrien','Ariadne','Calista','Dashiell','Esme','Idris','Noemi','Inez','Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette']
UNIT_WORDS={'kg':('mass change','separate stock weighed'),'degrees C':('temperature change','separate chamber measured'),'km':('displacement change','separate route spanned')}

def authored():
    rng=random.Random(76113);suffix=suffixes();rows=[];targets=[];group=0
    cells=list(itertools.product(('kg','degrees C','km'),('middle','end'),('+','-'),range(16)))
    cells += [('items',layout,'',role) for layout,role in itertools.product(('middle','end'),range(16))]
    for unit,layout,sign,role in cells:
        actor=NAMES[group%8];peer=NAMES[(group+3)%8];episode=4100+group;group_id=f'joint76:{group:04d}'
        base_a=10+group%20;base_b=50+group%20
        if unit=='items':base_a+=20;base_b+=20
        quartet=[]
        for i,j in itertools.product(range(2),repeat=2):
            a=base_a+i;b=base_b+j
            if unit=='items':
                A=f'{a} steel rings';B=f'{b} steel rings'
                left=f'{actor} shipped {A}';right=f'{peer} kept {B}'
            else:
                A=f'{sign}{a}.75 {unit}';B=f'+{b}.25 {unit}';kind,predicate=UNIT_WORDS[unit]
                left=f'{actor} registered a {kind} of {A}';right=f"{peer}'s {predicate} {B}"
            clause=(f'For entry {episode}, {left}; {right}.' if layout=='end' else
                f'For entry {episode}, {right}; {left}. The two records describe separate quantities.')
            quartet.append((i,j,a,b,A,B,clause))
        # All four numeric combinations have equal byte lengths. Natural prefix
        # variation covers EOS alignment without a GT length feature.
        stem=''.join(rng.choice(CONTEXT) for _ in range(group%4))
        needed=(role-len(stem.encode())-len(quartet[0][-1].encode()))%16
        prefix=stem+rng.choice(suffix[needed])
        for i,j,a,b,A,B,clause in quartet:
            text=prefix+clause;raw=text.encode();assert len(raw)%16==role and len(raw)<16384
            ident='joint76:'+hashlib.sha256(raw).hexdigest()[:24]
            names=[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'\b(?:'+'|'.join(NAMES)+r')\b',text)]
            spans=complete_expression_spans(text,names)
            def expression_span(value):
                numeric=value.removesuffix(' steel rings')
                match=list(re.finditer(r'(?<![A-Za-z0-9_])'+re.escape(numeric)+r'(?![A-Za-z0-9_])',text))
                assert len(match)==1
                return dict(start=match[0].start(),end=match[0].end(),text=numeric)
            rows.append(dict(id=ident,prompt=text))
            targets.append(dict(id=ident,source_kind='prospective_joint_quantity',quartet_id=group_id,member=[i,j],
                mode='unsigned' if unit=='items' else 'signed',unit=unit,layout=layout,sign=sign,EOS_position=role,
                actor=actor,peer=peer,episode=episode,A=a,B=b,primary_text=A,secondary_text=B,
                A_span=expression_span(A),B_span=expression_span(B),clause_start=len(prefix.encode()),clause=clause,
                critical_spans=spans))
        group+=1
    assert group==224 and len(rows)==896
    return rows,targets

def main():
    assert not OUT.exists(),'never overwrite immutable artifact'
    rows,targets=authored();old=ROOT/'research/artifacts/critical_retention_20261112_v75_cpu'
    oldrows=io.load_json(old/'train_features.json');oldt=io.load_json(old/'train_targets.json')
    rows.extend(oldrows[512:]);targets.extend(oldt[512:]);assert len(rows)==1024
    assert len({r['prompt'] for r in rows})==1024 and all(set(r)=={'id','prompt'} for r in rows)
    held_files=[ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/short_features.json',
        ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/long_features.json',
        ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2/dev_features.json',old/'dev_features.json']
    held=[h for f in held_files for h in io.load_json(f)]
    assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in held for r in rows)
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in rows for n in FORBIDDEN)
    groups=[list(range(i,i+4)) for i in range(0,896,4)]+[[i] for i in range(896,1024)]
    random.Random(76113).shuffle(groups);order=[[i] for group in groups for i in group]
    coverage=collections.Counter((t['unit'],t['layout'],t['sign'],t['EOS_position']) for t in targets[:896])
    assert len(coverage)==224 and set(coverage.values())=={4}
    OUT.mkdir()
    for name,v in [('train_features',rows),('train_targets',targets),('orders',order)]:io.atomic_json(OUT/(name+'.json'),v)
    manifest=dict(schema='joint_quantity_v76_CPU_data',TRAIN_rows=1024,authored_rows=896,correlated_quartets=224,
        signed_rows=768,unsigned_rows=128,original_OWT_TRAIN_rows=128,finite_epochs=1,
        source_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),max_input_bytes=max(len(r['prompt'].encode()) for r in rows),
        critical_bytes=sum(s['end']-s['start'] for t in targets for s in t['critical_spans']),
        full_unit_layout_sign_EOS_cross_coverage=True,features=['id','prompt'],facts_spans_quartets_LOSS_SCORE_ONLY=True,
        TRAIN_names=NAMES,forbidden_observed_held_names=FORBIDDEN,new_DEV_rows=0,
        preserved_OWT_manifest_SHA=hashlib.sha256((old/'manifest.json').read_bytes()).hexdigest(),
        held_exclusion_files_SHA={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in held_files},
        actual_model_worker_plan_GPU=False,denoise_calls=0,Exact_copy_bytes=0,E2_final_registry_access=0,downloads=0,
        limitations='Prospective authored repeated grammar after observed failures;224 correlated quartets are not896 independent documents. Same old128 OWT TRAIN. No new evaluation, performance, semantic-space or CUDA recovery result.')
    io.atomic_json(OUT/'manifest.json',manifest)
    io.atomic_json(OUT/'builder_SHA.json',dict(file=str(Path(__file__)),SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(manifest))

if __name__=='__main__':main()
