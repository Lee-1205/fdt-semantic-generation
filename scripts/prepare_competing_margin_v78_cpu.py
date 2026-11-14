"""Reuse immutable TRAIN/eval/profile; group the exact V77 exposure sequence."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    d=ROOT/'research/artifacts/competing_margin_20261113_v78_cpu_v1';assert d.exists() and not (d/'orders.json').exists()
    base=ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1';source=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1'
    old=io.load_json(base/'orders.json');rows=io.load_json(source/'train_features.json');targets=io.load_json(source/'train_targets.json')
    groups=[[v[0] for v in old[a:a+5]] for a in range(0,len(old),5)]
    def mode(i):return targets[i]['mode'] if targets[i]['source_kind']=='prospective_joint_quantity' else 'OWT'
    assert len(rows)==1024 and len(groups)==384 and len(old)==1920 and all(len(o)==1 for o in old)
    assert all(list(map(mode,g))==['unsigned','signed','OWT','unsigned','signed'] for g in groups)
    flat=[i for g in groups for i in g];counts=collections.Counter(flat)
    assert flat==[o[0] for o in old] and all(counts[i]=={'signed':1,'unsigned':6,'OWT':3}[mode(i)] for i in range(1024))
    assert all(set(row)=={'id','prompt'} for row in rows)
    manifest=dict(schema='V78_CPU_grouped_exposure_contract',new_unique_TRAIN_rows=0,unique_rows=1024,teacher_rows=1024,head_rows=1920,optimizer_groups=384,
        group_size=5,group_modes=['unsigned','signed','OWT','unsigned','signed'],row_loss_weights=[.2]*5,
        update_rule='SAME parameter state for all five micro rows, sum gradients then unscale/clip/Adam ONCE',
        group_rows_are_correlated_and_repeated=True,teacher_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),
        head_UTF8_bytes=sum(len(rows[i]['prompt'].encode()) for i in flat),head_critical_bytes=sum(sum(s['end']-s['start'] for s in targets[i]['critical_spans']) for i in flat),
        epochs='teacher unique once; head exact finite repeated schedule, not one unique epoch',source_feature_file=str(source/'train_features.json'),
        source_target_file=str(source/'train_targets.json'),frozen_symbol_profile=str(base/'symbol_weight_source.json'),
        eval_data=str(ROOT/'research/artifacts/joint_quantity_eval_20261113_v77_cpu_v1'),
        source_SHA={str(f):support.sha(f) for f in (base/'orders.json',base/'symbol_weight_source.json',source/'train_features.json',source/'train_targets.json')},
        new_actual_worker_plan_seal_GPU=False,denoise0=True,Exact_content0=True,copy0=True)
    assert manifest['head_UTF8_bytes']==2720189 and manifest['teacher_UTF8_bytes']==1010943 and manifest['head_critical_bytes']==86108
    io.atomic_json(d/'orders.json',groups);io.atomic_json(d/'manifest.json',manifest)
    io.atomic_json(d/'builder_SHA.json',dict(file=str(Path(__file__)),SHA=support.sha(Path(__file__))))
    print(json.dumps(manifest))


if __name__=='__main__':main()
