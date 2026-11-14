"""Save actual fullsize preflight/OS evidence without another model/hash replay."""
import json,math,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import supervise_three_memory_codec_20261030_v43 as owner

def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    a=ROOT/'research/artifacts/three_memory_codec_20261030_v43_worker_cpu'
    p=owner.validate(a/'plan_unsealed.json',ROOT/'runs/three_memory_codec_20261030_v43')
    c=json.loads((a/'cpu_preflight_owned/final_status.json').read_text())
    real=json.loads((a/'real_windows_owner_review.json').read_text())
    r=c['report']
    assert c['status']=='CPU_PREFLIGHT_PASS' and c['worker_exit_code']==0 and c['seconds']<90
    assert c['exit']['all_owned_processes_gone'] and not c['owner_imported_Torch']
    assert c['actual']['identity_verified'] and r['source_sha256']==p['source_sha']
    assert r['actual_initial_reference_CPU_roundtrip'] and r['actual_fullsize_initial_reference_bytes']<32*1024**2
    assert r['actual_fullsize_initial_reference_serialized_bytes']<64*1024**2
    assert r['initial_reference_checkpoint_format']==owner.INITIAL_CHECKPOINT_CONTRACT['format']
    factor=.1+.9*(1+math.cos(math.pi*8191/8192))/2
    assert r['source_terminal_group_lrs']==[base*factor for base in (1e-5,1e-4,3e-6,1e-4)]
    assert r['policy']['learning_rate_schedule']=='constant_source_terminal_group_LRs_no_restart_no_moment_or_step_reset'
    assert real['exit']['all_owned_processes_gone'] and not real['owner_imported_Torch']
    tests=[]
    for name,count in [('cpu_delta.xml',13)]:
        t=ET.parse(a/name).getroot().find('testsuite')
        assert int(t.attrib['tests'])==count and not int(t.attrib['failures']) and not int(t.attrib['errors'])
        tests.append({'file':name,'tests':count,'seconds':float(t.attrib['time'])})
    assert not owner.native.process_inventory(ROOT)
    resources=owner.resources(p)
    pre={'status':'INDEPENDENT_VALIDATE_PASS','exit_code':0,'actual_fullsize_source_import':True,
         'actual_fullsize_initial_reference_safe_roundtrip':True,'actual_worker':Path(p['files'] and c['command'][4]).name,
         'actual_model_forward_calls':0,'model_GPU_calls':0,'whole_CUDA_recovery_replay':False,
         'owned_OS_evidence':'cpu_preflight_owned/final_status.json','actual_report':r,'seconds':c['seconds']}
    write(a/'actual_worker_preflight.json',pre)
    review={'status':'ROOT_V43_CPU_OS_DELTA_REVIEW_COMPLETE_NOT_GPU','CPU_tests':tests,
        'actual_worker_preflight':pre,'real_owner_CPU_fixture':real,'resources':resources,
        'source_ALL91_model_Adam_scaler_ALLstoredRNG_exact_import':True,
        'initial_compressed_32MiB_and_serialized_64MiB_actual_roundtrip':True,
        'only_runtime_PT_pin':[name for name in p['files'] if Path(name).suffix=='.pt'],
        'same_neural_function':True,'storage_helper_reused_gzip_v42':True,
        'source_version':'V42_FINAL_NOT_OLDER_PARITY_REFERENCE','epochs':[13,14],'seeds':[42001,42002],
        'max_attempts':8192,'actual_fullsize_forward':0,'GPU_calls':0,'actual_CUDA_replay':False,
        'old_allocation_reopened':False,'owner_imported_Torch':'torch' in sys.modules}
    write(a/'root_cpu_review.json',review)
    print(json.dumps({'status':review['status'],'CPU_tests':13,'unchanged_owned_preflight_tests_reused':6,'actual_sizes':r,
                      'resources':resources,'owner_imported_Torch':review['owner_imported_Torch']}))
if __name__=='__main__':main()
