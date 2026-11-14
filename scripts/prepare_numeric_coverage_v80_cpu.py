"""Prospective 896 correlated numeric TRAIN views + same128 OWT, no NN.

Repair measured A/B sign/width confounds, without copying observed DEV events.
Marginal/selected interaction coverage is explicit; not full factorial.
"""
import collections,hashlib,itertools,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from prepare_joint_quantity_v76_cpu import NAMES,FORBIDDEN
from prepare_fuzzy_extraction_v73_cpu import CONTEXT,suffixes
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
OUT=ROOT/'research/artifacts/numeric_coverage_20261113_v80_cpu_v1'
SIGN_PAIRS=list(itertools.product(('+','-',''),repeat=2))
UNITS=('kg','degrees C','km','items')
ITEMS=('linen rolls','bronze seals','paper labels','wooden spools')


def authored():
    rng=random.Random(80113);pads=suffixes();rows=[];targets=[];quartets=[]
    for ui,unit in enumerate(UNITS):
        for j in range(56):
            group=ui*56+j;digit_width=(j//9)%3+1;lo,hi=((0,8),(10,98),(100,998))[digit_width-1]
            na=lo+(group*7+3)%(hi-lo+1);nb=lo+(group*11+5)%(hi-lo+1)
            sa,sb=SIGN_PAIRS[j%9] if unit!='items' else ('','')
            fraction=('.25','.50','.875')[(j//3+ui)%3] if unit!='items' else ''
            actor,peer=NAMES[(j+ui)%8],NAMES[(j+ui+3)%8];eid=61000+group
            family=(j//16+ui)%3;layout='BA' if j%2 else 'AB';role=(5*j+3*ui)%16
            item=ITEMS[(j//2)%len(ITEMS)]
            def expression(n,sign):return sign+str(n)+fraction+(' '+unit if unit!='items' else '')
            before=''.join(rng.choice(CONTEXT) for _ in range(group%4))
            local=[]
            for a,b in itertools.product((0,1),repeat=2):
                A=expression(na+a,sa);B=expression(nb+b,sb)
                if unit=='items':
                    first=f'{actor} counted {A} {item}';second=f'{peer} counted {B} {item}'
                else:
                    first=f'{actor} recorded a reading of {A}';second=f'{peer} recorded a separate reading of {B}'
                if layout=='BA':ordered=[(second,'B',B),(first,'A',A)]
                else:ordered=[(first,'A',A),(second,'B',B)]
                if family==0:head=f'Register {eid}: ';separator='; ';tail='.'
                elif family==1:head=f'At check {eid}, ';separator=', while ';tail='. Both readings were retained.'
                else:head=f'Archive {eid} lists two entries: ';separator='; independently, ';tail='.'
                text=head;spans={}
                for k,(clause,key,value) in enumerate(ordered):
                    if k:text+=separator
                    offset=clause.index(value);start=len(text.encode())+offset
                    spans[key]=dict(start=start,end=start+len(value.encode()),text=value)
                    text+=clause
                text+=tail;local.append((a,b,text,spans,A,B))
            # Width and formatting stay constant within each four-way contrast.
            assert len({len(v[2].encode()) for v in local})==1
            pad=(role-len(before.encode())-len(local[0][2].encode()))%16;prefix=before+rng.choice(pads[pad]);ids=[]
            for a,b,clause,spans,A,B in local:
                text=prefix+clause;raw=text.encode();ident='coverage80:'+hashlib.sha256(raw).hexdigest()[:24];ids.append(len(rows))
                for s in spans.values():s['start']+=len(prefix.encode());s['end']+=len(prefix.encode());assert raw[s['start']:s['end']].decode()==s['text']
                name_spans=[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'\b(?:'+'|'.join(NAMES)+r')\b',text)]
                rows.append(dict(id=ident,prompt=text))
                targets.append(dict(id=ident,source_kind='prospective_numeric_coverage',quartet_id=f'coverage80:{group:04d}',member=[a,b],unit=unit,
                    mode='unsigned' if not(sa or sb) else 'signed',A_sign=sa or 'implicit_positive',B_sign=sb or 'implicit_positive',digit_width=digit_width,
                    A=na+a,B=nb+b,A_span=spans['A'],B_span=spans['B'],primary_text=A,secondary_text=B,actor=actor,peer=peer,episode=eid,
                    family=family,layout=layout,EOS_position=role,critical_spans=complete_expression_spans(text,name_spans)))
                assert len(raw)%16==role
            quartets.append(ids)
    return rows,targets,quartets


def main():
    assert not OUT.exists(),'immutable new artifact only'
    rows,targets,groups=authored();old=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1'
    oldrows=io.load_json(old/'train_features.json');oldt=io.load_json(old/'train_targets.json')
    assert len(rows)==896 and len(groups)==224 and len(oldrows)==1024
    rows.extend(oldrows[896:]);targets.extend(oldt[896:])
    assert all(set(r)=={'id','prompt'} for r in rows) and len({r['prompt'] for r in rows})==1024
    held_paths=[ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/short_features.json',ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/long_features.json',
        ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2/dev_features.json',ROOT/'research/artifacts/critical_retention_20261112_v75_cpu/dev_features.json',
        ROOT/'research/artifacts/joint_quantity_eval_20261113_v77_cpu_v1/features.json']
    # Exclusion-only access to already-observed development sources. These
    # strings never choose numbers, names, grammar, ordering, or TRAIN labels.
    held=[r for f in held_paths for r in io.load_json(f)]
    assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in held for r in rows)
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in rows[:896] for n in FORBIDDEN)
    # Same finite 384 groups/1920 head exposures is a prospective schedule,
    # NOT an actual worker/plan and NOT the old U/S/O/U/S order. Each authored
    # quartet occurs once; 160 distinct quartets receive one extra exposure.
    rng=random.Random(80114);extra=list(range(224));rng.shuffle(extra);gids=list(range(224))+extra[:160];rng.shuffle(gids)
    owt=list(range(896,1024))*3;rng.shuffle(owt)
    orders=[groups[g]+[o] for g,o in zip(gids,owt)]
    counts=collections.Counter(i for g in orders for i in g)
    assert len(orders)==384 and set(counts)==set(range(1024)) and all(counts[i] in (1,2) for i in range(896)) and all(counts[i]==3 for i in range(896,1024))
    cells=collections.Counter((t['unit'],t['A_sign'],t['B_sign'],t['digit_width']) for t in targets[:896]);roles={u:sorted({t['EOS_position'] for t in targets[:896] if t['unit']==u}) for u in UNITS}
    assert len(cells)==84 and all(v==list(range(16)) for v in roles.values())
    OUT.mkdir()
    for name,value in [('train_features',rows),('train_targets',targets),('orders',orders)]:io.atomic_json(OUT/(name+'.json'),value)
    manifest=dict(schema='V80_NUMERIC_COVERAGE_CPU_ONLY',TRAIN_rows=1024,authored_views=896,correlated_quartets=224,preserved_OWT_TRAIN_rows=128,new_dev_rows=0,
        numeric_units3_each_all9_A_B_sign_pairs_x3_widths=True,item_unsigned3widths=True,unit_sign_width_cells84=[dict(unit=k[0],A_sign=k[1],B_sign=k[2],width=k[3],rows=v) for k,v in cells.items()],
        EOS_roles=roles,full_sign_width_EOS_family_layout_factorial=False,features=['id','prompt'],spans_facts_LOSS_SCORE_ONLY=True,
        prospective_group_count384=True,head_rows=1920,head_group='authored quartet4 + OWT1',authored_quartets_once224_plus160_repeat=True,actual_model_forwards=0,actual_GPU=False,
        source_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),authored_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows[:896]),max_input_bytes=max(len(r['prompt'].encode()) for r in rows),
        head_UTF8_bytes=sum(len(rows[i]['prompt'].encode()) for g in orders for i in g),original_OWT_unchanged=True,
        held_exclusion_only_SHA={str(f):support_sha(f) for f in held_paths},E2_final_registry_access=0,downloads=0,denoise_calls=0,Exact_copy_bytes=0,
        limitations='TRAIN content and grouping changed jointly versus V79, no isolated optimizer comparison. Development is already observed; no new blind result. Source names unchanged TRAIN-only; not a general semantic certificate.',
        builder_SHA=support_sha(Path(__file__)),files_SHA={n+'.json':support_sha(OUT/(n+'.json')) for n in ('train_features','train_targets','orders')})
    io.atomic_json(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ('unit_sign_width_cells84','held_exclusion_only_SHA','files_SHA')}))


def support_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':main()
