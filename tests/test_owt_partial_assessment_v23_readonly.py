import json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import text_generation_support_v1 as support
import run_owt_partial_assessment_v23_readonly as worker
import supervise_owt_partial_assessment_v23_readonly as owner
from test_owt_fuzzy_reconstruction_v12 import fixture,Budget
from test_decoder_alignment_runtime_v8 import Child

def test_tiny_readonly_real_paths_preserve_model_RNG_no_training_or_optimizer():
    torch.set_num_threads(2);e=fixture();model=e.model.requires_grad_(False).eval();budget=Budget()
    before=support.state_sha(model.state_dict());rng=torch.get_rng_state().clone()
    with patch.object(model,'reconstruction_loss',side_effect=AssertionError('training forbidden')):
        metrics,tensors,raw,train=worker.assess(model,e.features[:8],e.targets[:8],e.features[:4],e.features[:4],budget,'cpu')
    assert support.state_sha(model.state_dict())==before and torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters()) and set(raw)=={'normal','swap','zero'}
    assert len(train)==4 and len(raw['normal'])==4 and tensors['normal'].shape[:2]==(8,129)
    assert budget.calls['encode']==3 and budget.calls['denoise']==0 and budget.calls['decoder']<=3+4*129
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_readonly_actual_plan_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((worker.OUTPUT/'plan.json').read_text());start=time.time()
    p.update(authorized=True,output=str(tmp_path),allocation_start=start,deadline=start+120,cutoff=start+90)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p));owner.validate(path,tmp_path)
    worker.validate(path,tmp_path,ROOT/'scripts/run_owt_partial_assessment_v23_readonly.py');patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_PARTIAL_V23_READONLY_ASSESSMENT_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_owt_partial_assessment_v23_readonly.py')
        out=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2)
        assert out['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
