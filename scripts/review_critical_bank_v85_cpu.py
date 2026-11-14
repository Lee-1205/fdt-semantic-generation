"""Independent CPU component review, not an actual FDT driver/preflight."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.critical_reference_bank_v85 import CriticalReferenceBank
from research.readout_precision_policy_v81 import POLICY

def main():
    torch.set_num_threads(4);a=ROOT/'research/artifacts/critical_reference_bank_20261113_v85_cpu_v1';a.mkdir(exist_ok=False)
    raw=[b'abcdefghijklmno.',b'pqrstuvwxyzabcd.',b'efghijklmnopqrs.'];assert all(len(v)==16 for v in raw)
    sources=[dict(id='CPU_SYNTHETIC:'+str(i),SHA=hashlib.sha256(v).hexdigest()) for i,v in enumerate(raw)]
    identity=dict(embedding_SHA='a'*64,prefix_SHA='b'*64,parent_SHA='c'*64);config=dict(sources=sources,broad_indices=[0],**identity)
    def make():return CriticalReferenceBank(**config)
    def packets(i):
        y=torch.full((2,16),257,dtype=torch.long);y[0]=torch.tensor(list(raw[i]));y[1,0]=256;valid=y!=257;critical=torch.zeros_like(valid);critical[0,3]=True
        m=torch.full((2,16),2.,dtype=torch.float32)
        if i:m[0,3]=.6
        feature=dict(hidden=(torch.arange(1024,dtype=torch.float32).reshape(2,512)+i)/1024,source_id=sources[i]['id'],source_SHA=sources[i]['SHA'],precision_policy=POLICY,**identity)
        target=dict(source_id=sources[i]['id'],source_SHA=sources[i]['SHA'],source_bytes=16,labels=y,critical_mask=critical,teacher_correct=valid,teacher_margin=m,precision_policy=POLICY)
        return feature,target
    rng=torch.get_rng_state().clone();b=make();b.ingest(0,*packets(0));partial=a/'CPU_synthetic_partial.pt'
    with partial.open('xb') as f:torch.save(b.export(),f)
    before=support.state_sha(b.export());restored=make();restored.restore(torch.load(partial,map_location='cpu',weights_only=True));assert support.state_sha(restored.export())==before
    for i in (1,2):b.ingest(i,*packets(i));restored.ingest(i,*packets(i))
    assert support.state_sha(b.export())==support.state_sha(restored.export())
    features,targets=b.components();assert set(features)=={'schema','hidden','source_ranges','metadata'} and 'hidden' not in targets and 'labels' not in features
    labels=targets['labels'];z=torch.full((*labels.shape,258),-20.,dtype=torch.float32);z[...,0]=0;z.scatter_(-1,labels[...,None],targets['teacher_margin'][...,None])
    b.calibrate(z);assert b.decision(z).accept
    ordinary=z.clone();ordinary[0,4,labels[0,4]]=.01;assert b.decision(ordinary).accept
    weak=z.clone();weak[2,3,labels[2,3]]=.01;receipt=b.decision(weak);assert not receipt.accept and receipt.measurements['weak_critical_margin_violations']==1 and receipt.measurements['new_critical_flips']==0
    complete=a/'CPU_synthetic_calibrated.pt'
    with complete.open('xb') as f:torch.save(b.export(),f)
    c=make();c.restore(torch.load(complete,map_location='cpu',weights_only=True));assert support.state_sha(b.export())==support.state_sha(c.export()) and c.decision(z).accept
    for name,packet in [('CPU_synthetic_features.pt',features),('CPU_synthetic_loss_only_targets.pt',targets)]:
        with (a/name).open('xb') as f:torch.save(packet,f)
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    old=ROOT/'research/artifacts/critical_frontier_20261113_v84_cpu_v1';selection=io.load_json(old/'historical_BF16_TRAIN_selection.json');extra=sum(len(v['patch_indices']) for v in selection if not v['in_old100'])
    assert extra==179 and 7487+extra==7666
    report=dict(status='ROOT_V85_REFERENCE_BANK_CPU_COMPONENT_INDEPENDENT_REVIEW_PASS',component_ONLY=True,CPU_unit_contract_tests=9,tests='6 bank integration +3 frontier selector',
        synthetic_sources=3,synthetic_hidden_dimension=512,actual_model_forwards=0,actual_GPU_calls=0,encoder_calls=0,optimizer_steps=0,denoise_calls=0,
        CUDA_initialized=False,global_Torch_RNG_unchanged=True,exact_partial_and_calibrated_bank_restore=True,finite_cursor_no_replay=True,separate_feature_loss_packets=True,
        ordinary_margin_drawdown_allowed_while_top1_retained=True,weak_critical_margin_violation_rejected_without_top1_flip=True,
        fresh_full_FP32_policy_and_model_epoch_required=True,strict_bank_patch_cap=8192,historical_merged_patch_estimate=7666,historical_hidden_FP32_bytes=15699968,hard_bank_hidden_FP32_bytes=16777216,
        preparation_not_GPU_performance=True,full_optimizer_scaler_ALLRNG_driver_restore_NOT_done=True,actual_new_driver_preflight_NOT_done=True,
        caveat='Synthetic logits exercise bank semantics only, not actual learned Linear or language ability. FP32 provenance must be guaranteed by the pinned actual future teacher execution; dtype/policy metadata alone cannot distinguish a falsely labelled BF16-to-FP32 cast. Historical7666 estimates must be recomputed for the actual parent/FP32 pass.',
        next='Integrate bank capture/calibration/decision with a new finite lastLinear driver; explicit device transfer, source/call/phase state and enabled-scaler/Adam/ALLRNG restore, bounded postPopen OS ownership, resource/output caps and actual preflight before one cause-directed training trial. No training plan/seal/launch exists yet.')
    io.atomic_json(a/'root_cpu_review.json',report)
    files=list(a.glob('CPU_synthetic*.pt'))+[a/'root_cpu_review.json',Path(__file__).resolve(),ROOT/'research/critical_reference_bank_v85.py',ROOT/'tests/test_critical_reference_bank_v85.py',old/'root_cpu_review.json',old/'historical_BF16_TRAIN_selection.json']
    io.atomic_json(a/'evidence_SHA.json',{str(f):support.sha(f) for f in files});print(json.dumps(report))

if __name__=='__main__':main()
