"""Prospective TRAIN-only reference selection and fixed-cost candidate policy.

No worker, executable plan, seal, permission question, or model forward.
"""
import collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    data=ROOT/'research/artifacts/numeric_coverage_20261113_v80_cpu_v1';out=ROOT/'research/artifacts/update_retention_20261113_v80_cpu_v1'
    assert io.load_json(data/'root_independent_CPU_review.json')['status']=='ROOT_V80_PROSPECTIVE_DATA_CPU_PASS'
    rows=io.load_json(data/'train_features.json');targets=io.load_json(data/'train_targets.json');cells=collections.defaultdict(list)
    key=lambda i:hashlib.sha256(rows[i]['prompt'].encode()).hexdigest()
    for i,t in enumerate(targets[:896]):cells[t['unit'],t['A_sign'],t['B_sign'],t['digit_width']].append(i)
    anchor=[min(ids,key=key) for _,ids in sorted(cells.items())]
    ordered=sorted(range(896,1024),key=lambda i:(len(rows[i]['prompt'].encode()),key(i)))
    for block in range(4):anchor.extend(sorted(ordered[32*block:32*(block+1)],key=key)[:4])
    assert len(anchor)==len(set(anchor))==100 and sum(i<896 for i in anchor)==84
    slots=sum((len(rows[i]['prompt'].encode())+15)//16+1 for i in anchor)
    selection=dict(schema='V80_TRAIN_REFERENCE_SELECTION_ONLY',indices=anchor,authored_cells=84,OWT_length_stratified_rows=16,
        selector='lowest source hash per TRAIN unit/A-sign/B-sign/digit-width cell; four lowest-hash OWT rows per length quartile',
        features='future frozen readout prefix activations derived solely from TRAIN Commit; no label/span/rawbyte as NN input',
        targets='TRAIN labels, critical flags and initial batched-reference correct masks in separate loss-only packet',
        max_hidden_rows=slots,hidden512_BF16_value_bytes=slots*512*2,
        teacher_or_label_used_for_conditioning=False,held_out_rows_selected=0,inference_Exact_memory_entries=0,
        limitation='100 TRAIN reference sources only; not all TRAIN or unseen meaning guarantee. Batched-reference initial predictions must be computed and checked at actual cache construction, not assumed from old rowwise logits.')
    io.atomic_json(out/'anchor_selection.json',selection)
    old=io.load_json(ROOT/'runs/last_linear_20261113_v79/plan.json');raw=io.load_json(out/'root_lossless_raw_review.json')
    estimate=dict(old['resource_estimate']['components']);estimate['raw_and_parity']-=raw['potential_new_output_saving_bytes'];estimate['TRAIN_anchor_feature_and_target_reserved']=24*1024**2
    assert selection['hidden512_BF16_value_bytes']<24*1024**2 and sum(estimate.values())<1024**3
    policy=dict(schema='V80_PROSPECTIVE_CPU_POLICY_NOT_EXECUTION_PLAN',authorized=False,permission_wait=False,
        purpose='Repair measured numeric coverage confounds and constrain actual Adam parameter proposals using fixed TRAIN-only reference sources',
        phase='English language-meaning-language repair; no diffusion',source_selection='V75',source_runtime_SHA='f01e851284df0bd971498ee709b81158abb4ce157bf7e19578c146614411c124',
        active_names=['readout.readout.net.2.weight','readout.readout.net.2.bias'],frozen='all input/memory/binding and D terminal/norm/firstLinear',
        storage_deltas7_optimizer2=True,head_attempt_cap384=True,source_head_rows1920=True,new_data_and_groups_jointly_changed=True,
        actual_update_candidate=dict(alphas=[1.,.5,.25,.125],full_proposal_moments_retained_for_damped_acceptance=True,
            all_reject='weights+Adam restored; finite-attempt scaler advanced, source/cursor/RNG/calls consumed',AMP_skip='no reference calls; no weight or Adam mutation',
            exception='weights+Adam+scaler restored; gradients cleared; caller saves consumed/inflight state and stops; never replay',
            reference_proposal='No new initial-reference-correct critical/noncritical top1 flips on the selected100 TRAIN sources; initial batched reference computed once, never gold as feature',
            unverified='Actual full FDT engine/scaler/cursor/feature-cache/phase/CPU preflight and CUDA integration not implemented'),
        prospective_calls=dict(previous_same_phase_call_total=6752,initial_batched_reference=1,max_reference_candidate_calls=384*4,total_max=8289),
        proposed_owner_wall900_cut780_reserve120=True,proposed_call_cap8500=True,proposed_hard_output_bytes=1024**3,proposed_start_free_bytes=109*1024**3,
        protected_C_bytes=100*1024**3,expected_owned_bytes=sum(estimate.values()),estimate_components=estimate,
        storage_policy='New raw files only: lossless BF16 carrier for exact FP32 values; restore original dtype/bits before NLL/FSM; preserve full FP32 long/parity. Actual output including temporary+STOP hard limit must be implemented; no old file rewritten.',
        data_manifest_SHA=support.sha(data/'manifest.json'),anchor_selection_SHA=support.sha(out/'anchor_selection.json'),
        actual_worker_exists=False,actual_owner_exists=False,actual_executable_plan_or_seal=False,actual_GPU_launches=0,
        limitations='Prospective cost and storage estimate, not executable caps/approval or ability result. No raw-gradient or global semantic safety guarantee. Future thought-only full-codec freeze remains unchanged.')
    io.atomic_json(out/'prospective_policy.json',policy)
    print(json.dumps(dict(reference_sources=100,authored_cells=84,OWT_rows=16,hidden_rows=slots,hidden_bytes=selection['hidden512_BF16_value_bytes'],expected_owned_bytes=sum(estimate.values()),prospective_calls=8289,worker_plan_seal_GPU=False)))


if __name__=='__main__':main()
