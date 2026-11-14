import json,sys
from pathlib import Path
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import owned_cpu_preflight_v42 as owner


def invoke(tmp_path,*,crash=False,missing=False,unconfirmed=False,timeout=False,verify_error=False):
    worker=ROOT/'scripts/run_three_memory_codec_20261030_v42.py'
    plan=tmp_path/'plan.json';output=tmp_path/'never_model_output'
    plan.write_text(json.dumps({'root':str(ROOT),'authorized':False,
        'files':{str(worker):owner.native.sha256(worker)},'CPU_fixture_only':not missing}),encoding='utf-8')
    class Child:
        pid=987654321
        returncode=None if timeout else 0
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
    child=Child();original=owner.io.atomic_json
    def publish(path,value):
        if crash and Path(path).name=='state.json' and value.get('launches')==1:
            raise OSError('CPU_POST_POPEN_STATE_ERROR')
        return original(path,value)
    def launch(*args,**kwargs):
        kwargs['stdout'].write(b'CPU_PREFLIGHT_PASS\n');kwargs['stdout'].flush()
        return child
    def cleanup(*args,**kwargs):
        child.returncode=2 if timeout else 0
        return {'status':'UNCONFIRMED_EXIT' if unconfirmed or verify_error else 'STOPPED',
                'exit_verification':{'all_owned_processes_gone':not (unconfirmed or verify_error)}}
    clock=iter([0.,100.,100.,100.]) if timeout else None
    with patch.object(owner.subprocess,'Popen',side_effect=launch),\
         patch.object(owner.io,'atomic_json',side_effect=publish),\
         patch.object(owner.native,'verify_exit',side_effect=RuntimeError('VERIFY_ERROR') if verify_error else None,
                      return_value={'all_owned_processes_gone':not unconfirmed}) as verify,\
         patch.object(owner.native,'cleanup_owned_after_exception',side_effect=cleanup) as clean,\
         patch.object(owner.time,'monotonic',side_effect=lambda:next(clock)) if timeout else patch.object(owner.time,'sleep'):
        if crash or missing or unconfirmed or timeout or verify_error:
            with pytest.raises(RuntimeError):owner.preflight(plan,output,worker,120)
        else:owner.preflight(plan,output,worker,120)
    state=json.loads((tmp_path/'cpu_preflight_owned/final_status.json').read_text())
    assert state['launches']==1 and not output.exists()
    return state,clean.call_count,verify.call_count


def test_natural_fixture_exit_verified_without_cleanup(tmp_path):
    state,clean,verify=invoke(tmp_path)
    assert state['status']=='CPU_PREFLIGHT_PASS' and clean==0 and verify==1


def test_postPopen_publication_error_cannot_skip_owned_cleanup(tmp_path):
    state,clean,verify=invoke(tmp_path,crash=True)
    assert state['status']=='STOPPED' and clean==1 and verify==0
    assert 'CPU_POST_POPEN_STATE_ERROR' in state['reason']


def test_actual_preflight_requires_fullsize_gzip_size_report(tmp_path):
    state,clean,_=invoke(tmp_path,missing=True)
    assert state['status']=='STOPPED' and clean==1 and 'report required' in state['reason']


def test_lingering_natural_child_blocks_model_launch(tmp_path):
    state,clean,_=invoke(tmp_path,unconfirmed=True)
    assert state['status']=='UNCONFIRMED_EXIT' and clean==1


def test_hard_worker_clock_has_separate_cleanup_reserve(tmp_path):
    state,clean,_=invoke(tmp_path,timeout=True)
    assert state['status']=='STOPPED' and clean==1
    assert state['worker_seconds_cap']==90 and state['total_seconds_cap']==120


def test_exit_verification_exception_still_attempts_cleanup_and_blocks(tmp_path):
    state,clean,_=invoke(tmp_path,verify_error=True)
    assert state['status']=='UNCONFIRMED_EXIT' and clean==1
