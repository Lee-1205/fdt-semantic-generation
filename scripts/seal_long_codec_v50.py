"""Seal a fresh clock only after actual CPU worker/OS exit evidence is complete."""
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_long_codec_plan_v50 import ART,OUT,sha
from run_long_codec_20261111_v50 import validate

def main():
    if OUT.exists():raise FileExistsError('old output must not reopen')
    preflight=ART/'cpu_preflight_owned/final_status.json'
    report=json.loads(preflight.read_text())
    if (report['status']!='CPU_PREFLIGHT_PASS' or not report['exit']['all_owned_processes_gone']
        or report['owner_imported_Torch'] or not report['report']['fullsize_engine_restore']):
        raise ValueError('actual CPU+OS evidence')
    p,rows,dev,orders=validate(ART/'plan_unsealed.json',OUT)
    manifest=json.loads((ART/'manifest.json').read_text())
    if sum(row['actual_bytes'] for row in rows)*4!=p['data_exposure']['target_UTF8_bytes']:
        raise ValueError('actual UTF8 exposure cap')
    if 3*len(orders)+4*len(dev)+2*len(p['raw_indices'])*p['raw_max_symbols']!=p['cost']['total_upper']:
        raise ValueError('actual model call calculation')
    p['files'][str(preflight)]=sha(preflight)
    p['files'][str(Path(__file__).resolve())]=sha(__file__)
    start=time.time();p.update(authorized=True,start=start,deadline=start+2100,cutoff=start+1920)
    OUT.mkdir();(OUT/'plan.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    validate(OUT/'plan.json',OUT)
    snap=OUT/'code_snapshot';snap.mkdir()
    for directory,pattern in [('scripts','*v50.py'),('research','trainable_long_codec_v50.py')]:
        for path in (ROOT/directory).glob(pattern):(snap/path.name).write_bytes(path.read_bytes())
    result=dict(plan_SHA=sha(OUT/'plan.json'),actual_CPU_fullstate_restore=True,cost_upper=p['cost']['total_upper'],
                rows=len(rows),epochs=4,target_UTF8_bytes=p['data_exposure']['target_UTF8_bytes'],start=start)
    (OUT/'root_prelaunch_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
