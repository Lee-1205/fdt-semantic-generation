"""Torch-free preparation/independent OS launch of ONE source roundtrip screen."""
import argparse,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'scripts')]
import learning_io_v2 as io
from owned_cpu_preflight_v50 import preflight

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=('prepare','launch'));args=q.parse_args()
    a=P/'research/artifacts/roundtrip_identity_screen_20261114_v108_cpu_v1';worker=P/'scripts/screen_roundtrip_identity_v108_cpu.py'
    if args.mode=='prepare':
        a.mkdir(exist_ok=False)
        old=io.load_json(P/'runs/grid_tail_thought_20261114_v105/plan.json');pins=dict(old['files'])
        for f in (worker,Path(__file__).resolve()):pins[str(f)]=sha(f)
        p={k:old[k] for k in ('root','selection','codec_SHA','cache','data')}
        p.update(schema='cached_roundtrip_all480_identity_screen_v108',authorized=False,CPU_fixture_only=True,
            output=str(a/'readonly_output'),files=pins,max_calls=480,max_output_bytes=96*1024**2,work_seconds=75,
            train_allowed_only_strict_roundtrip=True,no_DEV_in_training=True,no_optimizer=True)
        io.atomic_json(a/'plan.json',p);print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',plan_SHA=sha(a/'plan.json'))))
    else:
        p=io.load_json(a/'plan.json');review=io.load_json(a/'root_independent_prelaunch_review.json')
        assert review['plan_SHA']==sha(a/'plan.json') and review['CPU_code_and_finite_cost_checked']
        for f,expected in p['files'].items():assert sha(f)==expected
        print(json.dumps(preflight(a/'plan.json',Path(p['output']),worker,120)))
