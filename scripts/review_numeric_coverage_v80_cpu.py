"""Independent complete prospective TRAIN row/span/contrast/cursor review."""
import collections,decimal,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    root=ROOT/'research/artifacts/numeric_coverage_20261113_v80_cpu_v1';manifest=io.load_json(root/'manifest.json')
    rows=io.load_json(root/'train_features.json');targets=io.load_json(root/'train_targets.json');orders=io.load_json(root/'orders.json')
    old=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1';oldrows=io.load_json(old/'train_features.json');oldt=io.load_json(old/'train_targets.json')
    assert rows[896:]==oldrows[896:] and targets[896:]==oldt[896:] and len(rows)==len(targets)==1024
    assert not ({r['prompt'] for r in rows[:896]} & {r['prompt'] for r in oldrows})
    for name,sha in manifest['files_SHA'].items():assert support.sha(root/name)==sha
    for name,sha in manifest['held_exclusion_only_SHA'].items():assert support.sha(name)==sha
    groups=collections.defaultdict(list);cells=collections.Counter();roles=collections.defaultdict(set);single=collections.defaultdict(set)
    for i,(row,t) in enumerate(zip(rows[:896],targets[:896])):
        raw=row['prompt'].encode();assert set(row)=={'id','prompt'} and t['id']==row['id']=='coverage80:'+hashlib.sha256(raw).hexdigest()[:24]
        assert len(raw)%16==t['EOS_position'] and len(raw)<16384 and t['A']>=0 and t['B']>=0
        for key in ('A','B'):
            span=t[key+'_span'];text=raw[span['start']:span['end']].decode();assert text==span['text']
            numeric=text if t['unit']=='items' else text[:-(len(t['unit'])+1)]
            value=decimal.Decimal(numeric);sign=t[key+'_sign'];integer_text=numeric.lstrip('+-').split('.')[0]
            assert int(integer_text)==t[key] and len(integer_text)==t['digit_width']
            assert (numeric.startswith('+') if sign=='+' else numeric.startswith('-') if sign=='-' else numeric[0].isdigit())
            assert (value<=0 if sign=='-' else value>=0)
            assert any(s['start']<=span['start'] and s['end']>=span['end'] for s in t['critical_spans'])
            if t['digit_width']==1:single[t['unit']].add(t[key])
        assert t['mode']==('unsigned' if t['A_sign']==t['B_sign']=='implicit_positive' else 'signed')
        groups[t['quartet_id']].append(i);cells[t['unit'],t['A_sign'],t['B_sign'],t['digit_width']]+=1;roles[t['unit']].add(t['EOS_position'])
    assert len(groups)==224 and len(cells)==84
    for unit in ('kg','degrees C','km','items'):
        assert roles[unit]==set(range(16)) and single[unit]==set(range(10))
        signs=(('+','-','implicit_positive') if unit!='items' else ('implicit_positive',))
        assert all(cells[unit,sa,sb,w]>0 for sa in signs for sb in signs for w in (1,2,3))
    for ids in groups.values():
        assert len(ids)==4 and {tuple(targets[i]['member']) for i in ids}=={(0,0),(0,1),(1,0),(1,1)}
        canonical=set();first=targets[ids[0]]
        for i in ids:
            t=targets[i];raw=rows[i]['prompt'].encode();a,b=t['member'];assert t['A']==first['A']+a and t['B']==first['B']+b
            # Outside the two numeric spans, every byte within a contrast
            # quartet is identical. No contextual or identity answer leakage.
            for key in sorted(('A','B'),key=lambda k:t[k+'_span']['start'],reverse=True):
                s=t[key+'_span'];raw=raw[:s['start']]+('<'+key+'>').encode()+raw[s['end']:]
            canonical.add(raw)
        assert len(canonical)==1
    counts=collections.Counter(i for g in orders for i in g)
    assert len(orders)==384 and sum(counts.values())==1920 and set(counts)==set(range(1024))
    assert all(len(g)==5 and len({targets[i]['quartet_id'] for i in g[:4]})==1 and g[4]>=896 for g in orders)
    assert all(counts[i] in (1,2) for i in range(896)) and all(counts[i]==3 for i in range(896,1024))
    # Serialize every possible next-group cursor, including the terminal cursor.
    for cursor in range(385):
        payload=json.loads(json.dumps(dict(order=orders,cursor=cursor,attempts=cursor)))
        assert payload['order'][payload['cursor']:]==orders[cursor:]
    proof=dict(status='ROOT_V80_PROSPECTIVE_DATA_CPU_PASS',rows_independently_checked=1024,authored_rows_checked=896,preserved_OWT_rows_exact=128,
        independent_Decimal_span_value_sign_width_checks=True,all224quartet_outside_numeric_bytes_identical=True,all84unit_sign_width_cells_nonempty=True,
        all4units_have_digits0through9_and_all16_EOS_roles=True,all385finite_nextgroup_cursors_exact=True,finite1920_head_rows_checked=True,
        full_cross_factorial=False,source_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),head_UTF8_bytes=sum(len(rows[i]['prompt'].encode()) for g in orders for i in g),
        files_SHA={str(root/n):support.sha(root/n) for n in ('train_features.json','train_targets.json','orders.json','manifest.json')},
        reviewer_SHA=support.sha(Path(__file__)),actual_model_forwards=0,actual_optimizer_updates=0,new_GPU=False,E2_final_registry_access=0,
        limitation='CPU data validity only; mode proportions/order change versus V79. No candidate FDT worker or CUDA restore/performance evidence.')
    io.atomic_json(root/'root_independent_CPU_review.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
