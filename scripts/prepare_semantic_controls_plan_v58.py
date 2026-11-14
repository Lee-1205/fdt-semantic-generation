"""Actual readonly semantic contract; --seal only after owned CPU validation."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_semantic_controls_20261111_v58 import validate,SOURCE_SHA,HEAD_SHA

def main(seal):
    artifact=ROOT/'research/artifacts/semantic_controls_20261111_v58_cpu';unsealed=artifact/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text(encoding='utf-8-sig'));output=Path(p['output']);validate(unsealed,output)
        if output.exists():raise FileExistsError('no old budget reopen')
        proof=artifact/'cpu_preflight_owned/final_status.json';r=json.loads(proof.read_text(encoding='utf-8-sig'))
        if (r['status']!='CPU_PREFLIGHT_PASS' or not r['exit']['all_owned_processes_gone'] or r['owner_imported_Torch']
            or not r['report']['fullsize_engine_restore'] or r['report']['closed_historical_state_executed']):raise ValueError('actual readonly CPU proof')
        p['files'][str(proof)]=support.sha(proof);now=time.time();p.update(authorized=True,start=now,cutoff=now+135,deadline=now+180)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(path,output)
        snapshot=output/'code_snapshot';snapshot.mkdir()
        for source in (ROOT/'scripts/run_semantic_controls_20261111_v58.py',ROOT/'scripts/supervise_semantic_controls_v58.py',Path(__file__)):
            (snapshot/source.name).write_bytes(source.read_bytes())
        report=dict(plan_sha=support.sha(path),new_readonly_budget=True,cost=228,optimizer0=True,source_head_frozen=True)
        (output/'root_prelaunch_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report));return
    if unsealed.exists():raise FileExistsError('immutable prospective plan')
    oldrun=ROOT/'runs/broad_ordered_20261111_v57';old=json.loads((oldrun/'plan.json').read_text(encoding='utf-8-sig'))
    source=Path(old['source']);head=oldrun/'model/checkpoints/state_8192.pt';output=ROOT/'runs/semantic_controls_20261111_v58'
    pins={k:v for k,v in old['files'].items() if Path(k).suffix=='.py'}
    files=[ROOT/'scripts/run_semantic_controls_20261111_v58.py',ROOT/'scripts/supervise_semantic_controls_v58.py',
        ROOT/'scripts/prepare_semantic_controls_v58.py',Path(__file__),oldrun/'root_saved_review.json',source,head,oldrun/'plan.json']
    files.extend(artifact/(name+'.json') for name in ('features','targets','groups','manifest','root_CPU_contract'))
    for file in files:pins[str(file)]=support.sha(file)
    for name,digest in pins.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_semantic_controls_v58',authorized=False,root=str(ROOT),output=str(output),data=str(artifact),
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=HEAD_SHA,head_plan=str(oldrun/'plan.json'),
        files=pins,capacity=16384,attempts=0,epochs=0,amp_dtype='bfloat16',wall_seconds=180,
        start=None,cutoff=None,deadline=None,max_calls=240,max_output_bytes=64*1024**2,cost=228,
        purpose='readonly meaning-vs-surface organization,48correlated groups/192rows/47duplicates; not a blind benchmark',
        feature_contract='id+prompt only; target meanings used in CPU postscore only, no gold role/span/regex features',
        intervention='12 equal-length same-byte-bag role reversals; whole global2 vs local Commit swap, not an aligned semantic-variable causal proof',
        limits='cosine ranking is a limited measurement; semantic information may be nonlinear/distributed and Exact/local surface can vary across paraphrases',
        optimizer_updates=0,checkpoints=0,denoise_calls=0,source_copy_bytes=0,
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    unsealed.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(unsealed,output)
    print(json.dumps(dict(actual_schema_PASS=True,calls228=True,training=False)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
