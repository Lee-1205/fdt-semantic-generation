"""ROOT independent algebra + pinned CPU recovery + ORIGINAL actual bank.

No learned-model forward/update or additional actual-model trial. The actual
V88 bank is read as immutable data; analytic correction uses only two rows.
"""
import json
import math
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'src')]
import torch
from research.feasible_readout_delta_v89 import LinearConstraint, correct_actual_delta
from research.actual_update_retention_v80 import TrainDecision
from research.immutable_reference_anchor_v89 import verify
from research import critical_bank_storage_v87 as storage
import text_generation_support_v1 as support
import learning_io_v2 as io


def main():
    rng = torch.get_rng_state().clone()
    artifact = ROOT / 'research/artifacts/feasible_critical_engine_20261113_v90_cpu_v1'
    previous = ROOT / 'research/artifacts/feasible_readout_delta_20261113_v89_cpu_v1'
    for evidence in (artifact / 'evidence_SHA.json', previous / 'evidence_SHA.json'):
        for file, expected in io.load_json(evidence).items():
            assert support.sha(file) == expected, file
    proof = io.load_json(artifact / 'root_engine_cpu_review.json')
    assert proof['status'] == 'ROOT_V90_ACTUAL_ADAM_CORRECTION_CPU_ENGINE_CONTRACT_PASS_NOT_RESEARCH_TRAINING'
    assert proof['actual_hidden512_final_Linear_parameters'] == 2117664
    assert proof['populated_Adam_scaler_ALLRNG_next_group_exact'] and proof['distinct_warm_head_static_teacher']
    assert proof['static_teacher_reused_not_fresh_exposure'] and proof['original_anchor_floors_not_recalibrated']
    assert all(v['exact'] and v['consumed_attempt'] == 1 for v in proof['STOP_exact_no_replay'].values())
    partial = proof['STOP_exact_no_replay']['after_encoder_before_readout']
    assert partial['encoded_micros'] == [0] and partial['completed_micros'] == 0
    assert partial['calls'] == dict(encode=1, readout=0, reference_linear=1, denoise=0)
    anchor = io.load_json(previous / 'original_V80_FP32_reference_anchor.json')
    assert verify(anchor, anchor['representation'], anchor['bank_reference'])
    bank = storage.load(anchor['bank_reference'], ROOT, support.sha)
    assert support.state_sha(bank) == anchor['bank_reference']['bank_runtime_SHA']
    c = bank['calibration']
    assert int(c['weak_floor_mask'].sum()) == 152 and int(c['top1_mask'].sum()) == 119769
    assert torch.equal(c['required_margin'], c['initial_margin'].clamp(0., 1.) * .5)
    assert bank['config']['parent_SHA'] == anchor['origin_source_SHA']
    operators = io.load_json(previous / 'operator_files.json')
    for path, expected in operators.items(): assert support.sha(path) == expected, path
    # Independently compute first AdamW proposal incl decay and epsilon using
    # clipped g, rather than equating the raw gradient with the actual delta.
    warm = torch.tensor([[.1, -.1], [0., 0.]], dtype=torch.float64)
    bias = torch.zeros(2, dtype=torch.float64)
    gradient = torch.tensor([[1., -1.], [0., 0.]], dtype=torch.float64)
    gradient *= 1. / (math.sqrt(2.) + 1e-6)
    proposal = warm * (1. - .2 * .01) - .2 * gradient / (gradient.abs() + 1e-8)
    constraint = LinearConstraint(0, 0, 1, torch.tensor([1., 0.], dtype=torch.float64),
        .05, 'CPU_TRAIN:analytic', 0, 'a' * 64)
    def gap(w, b, h): return float(torch.dot(w[0]-w[1], torch.tensor(h, dtype=w.dtype)) + b[0]-b[1])
    def full(w, b): return TrainDecision(gap(w,b,(1.,0.)) >= .05 and gap(w,b,(0.,1.)) > 0,
        dict(A=gap(w,b,(1.,0.)), B=gap(w,b,(0.,1.))))
    result = correct_actual_delta(warm, bias, proposal, bias, [constraint], full, classes=2)
    deficit = .050001 - gap(proposal, bias, (1.,0.))
    tau = deficit / 4.
    expected = proposal.clone(); expected[0,0] += tau; expected[1,0] -= tau
    expected_b = torch.tensor([tau, -tau], dtype=torch.float64)
    assert torch.allclose(result['weight'], expected, atol=1e-14, rtol=0.)
    assert torch.allclose(result['bias'], expected_b, atol=1e-14, rtol=0.)
    assert result['receipt']['global_reference_pass'] and torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    report = dict(status='ROOT_V90_INDEPENDENT_ALGEBRA_STATIC_ANCHOR_AND_CPU_ENGINE_REVIEW_PASS',
        focused_CPU_tests=11, synthetic_engine_seconds=proof['fixture_seconds'],
        original_actual_bank=dict(anchor_SHA=anchor['anchor_SHA'], patches=bank['patch_count'],
            protected_top1=int(c['top1_mask'].sum()), original_weak_floors=152,
            targets_unchanged=True, floors_not_ratcheted=True, unchanged_representation_operator_files=len(operators)),
        analytic=dict(raw_Adam_proposal_A=gap(proposal,bias,(1.,0.)),
            corrected_A=gap(result['weight'],result['bias'],(1.,0.)), corrected_B=gap(result['weight'],result['bias'],(0.,1.)),
            formula_independently_reconstructed=True),
        actual_learned_model_calls=0, actual_learned_parameter_updates=0, GPU_calls=0, CUDA_initialized=False,
        whole_runnable_worker_plan_owner_ready=False, actual_fullsize_preflight_done=False,
        new_hypothesis='Direction correction of real Adam delta, original immutable reference, reused original teacher; no scalar-only restart/floor relaxation.',
        next_scope_PROSPECTIVE_NOT_PLAN=dict(attempts=96, head_rows_max=480, reused_teacher_rows=1024,
            fresh_teacher_encode_calls=0, head_encode_max=480, head_readout_max=480,
            reference_linear_max=194, final_cached_readout=194, new_baseline_model_calls=0,
            expected_max_calls=1348, suggested_cap=1400, owner_seconds=600, model_cutoff_seconds=480,
            reserve_seconds=120, output_MiB_including_STOP_temp=144, start_GiB=108.140625,
            no_accepted_window=32),
        limits='CPU synthetic updates and saved actual-bank identity do not show improved unseen/long English reconstructions, actual92rejections repaired, CUDA recovery or complete semantic codec. Actual worker/cost/schema/preflight/OSowner/independent validation and fresh C/GPU resources remain required.',
        Cfree=shutil.disk_usage(ROOT).free)
    io.atomic_json(artifact / 'root_independent_review.json', report)
    files=[Path(__file__).resolve(), artifact/'root_independent_review.json', artifact/'root_engine_cpu_review.json',
        artifact/'evidence_SHA.json', previous/'original_V80_FP32_reference_anchor.json',
        previous/'root_cpu_review.json', previous/'operator_files.json']
    io.atomic_json(artifact / 'root_independent_evidence_SHA.json', {str(f):support.sha(f) for f in files})
    print(json.dumps(report))


if __name__ == '__main__': main()
