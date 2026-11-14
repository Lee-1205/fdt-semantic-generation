"""CPU analytic proposal test and actual immutable-anchor identity; no NN."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.feasible_readout_delta_v89 import LinearConstraint,correct_actual_delta
from research.immutable_reference_anchor_v89 import build,verify,digest
from research.actual_update_retention_v80 import TrainDecision

def main():
    torch.set_num_threads(4);a=ROOT/'research/artifacts/feasible_readout_delta_20261113_v89_cpu_v1';a.mkdir(exist_ok=False)
    w=torch.tensor([[.1,-.1],[0.,0.]],dtype=torch.float64);b=torch.zeros(2,dtype=torch.float64);proposal=torch.tensor([[-.1,.1],[0.,0.]],dtype=torch.float64);q=b.clone()
    c=LinearConstraint(0,0,1,torch.tensor([1.,0.],dtype=torch.float64),.05,'CPU_TRAIN:analytic',0,'a'*64)
    def gap(x,y,h):return float(torch.dot(x[0]-x[1],torch.tensor(h,dtype=x.dtype))+y[0]-y[1])
    def full(x,y):return TrainDecision(gap(x,y,(1.,0.))>=.05 and gap(x,y,(0.,1.))>0,dict(A=gap(x,y,(1.,0.)),B=gap(x,y,(0.,1.))))
    rng=torch.get_rng_state().clone();out=correct_actual_delta(w,b,proposal,q,[c],full,classes=2)
    assert out['receipt']['global_reference_pass'] and gap(out['weight'],out['bias'],(1.,0.))>=.05 and gap(out['weight'],out['bias'],(0.,1.))>0 and torch.equal(rng,torch.get_rng_state())
    p=io.load_json(ROOT/'runs/critical_repair_20261113_v88/plan.json');result=io.load_json(ROOT/'runs/critical_repair_20261113_v88/root_result.json')
    assert result['all_frozen_except_last2_bitwise_source'] and result['all_owned_gone_before_analysis']
    sourcepath=Path(p['source']);currentpath=ROOT/'runs/critical_repair_20261113_v88/model/runtime.pt'
    assert support.sha(sourcepath)==p['source_SHA'] and support.sha(currentpath)==result['source_and_final_checkpoints']['final']['SHA']
    origin=torch.load(sourcepath,map_location='cpu',weights_only=True);current=torch.load(currentpath,map_location='cpu',weights_only=True)
    assert origin['parent_reference']==current['parent_reference'] and origin['model_keys']==current['model_keys'] and set(origin['active_names'])==set(current['active_names']) and len(current['active_names'])==7
    last={'readout.readout.net.2.weight','readout.readout.net.2.bias'}
    for key in set(current['active_names'])-last:assert torch.equal(current['active_weights'][key],origin['active_weights'][key])
    assert current['frozen_model_SHA']==origin['frozen_model_SHA'] and 'model' not in current['runtime'] and 'model' not in origin['runtime']
    reference=current['runtime']['bank_ref'];features=torch.load(reference['features'],map_location='cpu',weights_only=True);targets=torch.load(reference['targets'],map_location='cpu',weights_only=True)
    assert support.sha(reference['features'])==reference['features_SHA'] and support.sha(reference['targets'])==reference['targets_SHA']
    assert targets['bank_runtime']['config']['parent_SHA']==p['source_SHA'] and targets['bank_runtime']['calibration']['weak_floor_mask'].sum()==152
    # Operator snapshot comes from the already sealed V88 source closure.
    ops={f:s for f,s in p['files'].items() if Path(f).parent in (ROOT/'research',ROOT/'src/fdt_local')}
    for name in ('scripts/critical_bank_engine_v86.py','scripts/frozen_verbalizer_engine_v72.py'):ops[str(ROOT/name)]=p['files'][str(ROOT/name)]
    for f,s in ops.items():assert support.sha(f)==s,f
    operator_SHA=digest(dict(files=ops,encoder_autocast='BF16',readout='full_FP32',matmul='highest',TF32=False))
    def signature(envelope):
        state=envelope['runtime'];prefix={k:v for k,v in envelope['active_weights'].items() if k not in last}
        return dict(embedding_SHA=envelope['frozen_model_SHA'],prefix_SHA=support.state_sha(prefix),core_config_SHA=digest(state['core']),input_capacity=state['input_capacity'],precision_policy=p['precision_policy'],operator_SHA=operator_SHA)
    original_sig=signature(origin);warm_sig=signature(current);assert original_sig==warm_sig and features['metadata']['embedding_SHA']==warm_sig['embedding_SHA'] and features['metadata']['prefix_SHA']==warm_sig['prefix_SHA']
    anchor=build(p['source_SHA'],original_sig,reference);assert verify(anchor,warm_sig,reference)
    try:verify(anchor,warm_sig,dict(reference,targets_SHA='0'*64));raise AssertionError('anchor reset accepted')
    except ValueError:pass
    assert not torch.cuda.is_initialized()
    io.atomic_json(a/'original_V80_FP32_reference_anchor.json',anchor);io.atomic_json(a/'operator_files.json',ops)
    review=dict(status='ROOT_V89_CPU_FEASIBLE_DELTA_AND_ANCHOR_CONTRACT_PASS_NOT_TRAINED',CPU_unit_tests=8,
        analytic=dict(scalar_requires_A_alpha_max=.25,scalar_requires_B_alpha_min=.5,no_shared_scalar=True,corrected_A_gap=gap(out['weight'],out['bias'],(1.,0.)),corrected_B_gap=gap(out['weight'],out['bias'],(0.,1.)),receipt=out['receipt']),
        actual_reference_anchor=dict(origin_V80_SHA=p['source_SHA'],warm_V88_SHA=support.sha(currentpath),same_frozen_serialization_parent=True,same_five_prefix_tensors=True,same_frozen_embedding=True,same_core_capacity_and_precision=True,
            original_bank_files_preserved=True,original_weak152_floors_reused_not_recalibrated=True,anchor_SHA=anchor['anchor_SHA'],operators_pinned=len(ops)),
        all_old_files_unchanged=True,actual_learned_model_forwards=0,actual_learned_parameter_updates=0,actual_GPU_calls=0,CUDA_initialized=False,global_Torch_RNG_unchanged=True,
        scope='Synthetic analytic example plus actual checkpoint/reference identity. Does NOT show actual rejected V88 proposals repaired: those weights/logits/IDs were not saved. Not optimal projection of whole convex intersection, not new worker/optimizer/fullstate recovery or general semantic success.',
        next='Integrate bounded actual-Adam proposal correction in one transaction with original anchor, exact rollback/AMPskip/RNG/exposure and complete finite TRAIN reference; save blocker IDs from existing checks, never rerun V88 just for logs. Fixed2sweeps/16activeconstraints/radius<=1/padding1e-6 only prospective CPU policy.')
    io.atomic_json(a/'root_cpu_review.json',review)
    files=[Path(__file__).resolve(),ROOT/'research/feasible_readout_delta_v89.py',ROOT/'research/immutable_reference_anchor_v89.py',ROOT/'tests/test_feasible_readout_delta_v89.py',ROOT/'tests/test_immutable_reference_anchor_v89.py',a/'root_cpu_review.json',a/'original_V80_FP32_reference_anchor.json',a/'operator_files.json']
    io.atomic_json(a/'evidence_SHA.json',{str(f):support.sha(f) for f in files});print(json.dumps(review))

if __name__=='__main__':main()
