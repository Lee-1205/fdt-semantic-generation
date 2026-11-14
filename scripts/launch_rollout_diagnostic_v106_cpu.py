"""Torch-free plan/hash setup and owned OS CPU diagnostic launcher."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import learning_io_v2 as io
from owned_cpu_preflight_v50 import preflight

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

a=ROOT/'research/artifacts/thought_rollout_20261114_v106_cpu_v1';a.mkdir(exist_ok=False)
run=ROOT/'runs/grid_tail_thought_20261114_v105';old=io.load_json(run/'plan.json');result=io.load_json(run/'result.json')
worker=ROOT/'scripts/diagnose_rollout_v106_cpu.py';pins=dict(old['files'])
for f in (worker,Path(__file__).resolve(),run/'result.json',run/'ROOT_terminal_native.json',Path(result['checkpoints'][-1]['path'])):pins[str(f)]=sha(f)
p={k:old[k] for k in ('root','selection','codec_SHA','cache','data','curriculum','warm','tail_weight')}
p.update(schema='readonly_TRAIN_rollout_steps_v106',max_calls=96,authorized=False,output=str(a/'readonly_output'),files=pins,final_thought=result['checkpoints'][-1])
io.atomic_json(a/'plan.json',p);print(json.dumps(preflight(a/'plan.json',Path(p['output']),worker,120)))
