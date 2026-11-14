"""Fresh V42 two-epoch seal after independent CPU/source/native-owner checks."""
import json,shutil,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import supervise_three_memory_codec_20261030_v42 as owner

def main():
    a=ROOT/'research/artifacts/three_memory_codec_20261030_v42_worker_cpu'
    p=owner.validate(a/'plan_unsealed.json',ROOT/'runs/three_memory_codec_20261030_v42')
    out=Path(p['output']);pre=json.loads((a/'actual_worker_preflight.json').read_text(encoding='utf-8-sig'))
    real=json.loads((a/'real_windows_owner_review.json').read_text(encoding='utf-8'))
    suite=ET.parse(a/'cpu_delta.xml').getroot().find('testsuite')
    pretests=ET.parse(a/'preflight_cpu.xml').getroot().find('testsuite')
    if int(pretests.attrib['tests'])!=6 or int(pretests.attrib['failures']) or int(pretests.attrib['errors']):raise ValueError('owned CPU preflight contracts')
    if int(suite.attrib['tests'])!=12 or int(suite.attrib['failures']) or int(suite.attrib['errors']):raise ValueError('CPU delta')
    if pre['status']!='INDEPENDENT_VALIDATE_PASS' or not real['exit']['all_owned_processes_gone'] or real['owner_sha256']!=owner.native.sha256(Path(owner.__file__)):raise ValueError('actual preflight/native owner')
    if out.exists() or p['authorized'] or any(p[k] is not None for k in ('allocation_start','cutoff','deadline')):raise FileExistsError('ONE fresh allocation only')
    worker=ROOT/'scripts/run_three_memory_codec_20261030_v42.py'
    if owner.duplicates(worker) or owner.native.process_inventory(ROOT):raise RuntimeError('native duplicate worker')
    resources=owner.resources(p)
    snapshot=a/'code_snapshot';snapshot.mkdir(exist_ok=False)
    for name in p['files']:
        source=Path(name)
        if source.suffix=='.py':
            dest=snapshot/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    for file in (Path(__file__).resolve(),ROOT/'scripts/review_three_memory_owner_v42_windows_cpu.py',ROOT/'scripts/finalize_v42_cpu_review_20261030.py'):
        p['files'][str(file)]=owner.native.sha256(file);shutil.copyfile(file,snapshot/file.name)
    # Freeze CPU evidence before allocation clock exists. Manifest excludes itself.
    files={str(f.relative_to(a)).replace('\\','/'):owner.native.sha256(f) for f in a.rglob('*') if f.is_file()}
    manifest={'schema':'v42_CPU_contract_evidence','actual_model_GPU_calls':0,'CPU_tests':18,
        'actual_worker_preflight':'PASS','real_Windows_owner':'PASS','file_SHA256':files}
    manifest_path=a/'manifest.json'
    with manifest_path.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    p['files'][str(manifest_path)]=owner.native.sha256(manifest_path)
    out.mkdir(exist_ok=False);dest=out/'code_snapshot';dest.mkdir()
    shutil.copytree(snapshot,dest/'prelaunch')
    ready={'status':'ROOT_ACTUAL_CPU_OS_RESOURCE_REVIEW_COMPLETE','CPU_tests':18,'actual_worker_preflight':pre,
        'real_Windows_owner':real,'resources':resources,'CPU_manifest_sha256':owner.native.sha256(manifest_path),
        'source_ALL91Adam_scaler_ALLstoredRNG_import_verified':True,'GPU_execution_evidence':False,'old_allocation_reopened':False}
    (out/'readiness.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8')
    start=time.time();p.update(authorized=True,allocation_start=start,cutoff=start+1920,deadline=start+2100)
    path=out/'plan.json'
    with path.open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
    owner.validate(path,out)
    print(json.dumps({'status':'V42_ONE_NEW_SEAL_NOT_EXECUTION','plan_sha256':owner.native.sha256(path),
        'start':start,'cutoff':p['cutoff'],'deadline':p['deadline'],'resources':resources,'CPU_manifest_sha256':owner.native.sha256(manifest_path)}))
if __name__=='__main__':main()
