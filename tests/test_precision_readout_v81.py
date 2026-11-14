import copy
import pytest
import torch
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.precision_readout_v81 import from_commit,MODES


def fixture():
    torch.manual_seed(81);h=CommitPatchReadout(dimension=32,hidden=48).eval().requires_grad_(False)
    z=torch.randn(1,6,32);c=DifferentiableCodecCommit.from_semantic(z,(('CPU','v1','fixture'),),'CPUepoch')
    return h,c,z


def test_full_fp32_matches_original_and_all_modes_preserve_parameters():
    h,c,z=fixture();before=copy.deepcopy(h.state_dict())
    assert torch.equal(from_commit(h,c,'full_fp32')[0],h.from_commit(c))
    with torch.autocast('cpu',dtype=torch.bfloat16):old=h.from_commit(c)
    assert torch.equal(from_commit(h,c,'bf16')[0],old)
    for mode in MODES:
        logits,trace=from_commit(h,c,mode);assert logits.shape==(5,16,258) and torch.isfinite(logits).all()
        assert logits.dtype==(torch.bfloat16 if mode in ('bf16','prefix_fp32_last_bf16') else torch.float32)
    assert all(torch.equal(v,before[k]) for k,v in h.state_dict().items())


def test_last_projection_float32_is_real_compute_and_allows_input_gradient():
    h,c,z=fixture();z.requires_grad_(True);c=DifferentiableCodecCommit.from_semantic(z,(('CPU','v1','fixture'),),'CPUepoch')
    logits,trace=from_commit(h,c,'last_fp32');assert trace['hidden_dtype']=='torch.bfloat16' and trace['logit_dtype']=='torch.float32'
    logits.square().mean().backward();assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert all(p.grad is None for p in h.parameters())


def test_uniform_logits_offset_exposes_BF16_discrete_decision_risk():
    v=torch.tensor([-41.61,-41.60],dtype=torch.float32);assert int(v.argmax())==1
    assert int(v.bfloat16().float().argmax())==0
    assert int((v-v.max()).bfloat16().float().argmax())==1
    # Re-centering AFTER an irreversible output cast cannot recover the lost gap.
    cast=v.bfloat16().float();assert int((cast-cast.max()).argmax())==0


def test_invalid_mode_and_non_Commit_rejected():
    h,c,z=fixture()
    with pytest.raises(ValueError):from_commit(h,c,'answer_dependent_mode')
    with pytest.raises(TypeError):from_commit(h,z,'full_fp32')
