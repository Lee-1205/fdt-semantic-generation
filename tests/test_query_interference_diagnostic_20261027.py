from pathlib import Path
import json
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_query_interference_diagnostic_20261027 as diagnostic
import supervise_query_interference_diagnostic_20261027 as supervisor

DATA = ROOT / "research/artifacts/role8_query_mix_20261027_v1/dev.json"


def _encoded(batch=2, q=16):
    return {
        "init": torch.arange(batch*q*3).reshape(batch,q,3).float(),
        "query_roles": torch.arange(batch*q*5).reshape(batch,q,5).float(),
        "query_mask": torch.ones(batch,q),
        "facts": torch.arange(batch*4*3).reshape(batch,4,3).float(),
        "fact_mask": torch.ones(batch,4),
        "memory": torch.arange(batch*2*3).reshape(batch,2,3).float(),
        "ground": torch.arange(batch*4*4).reshape(batch,4,4).float(),
        "grades": torch.arange(batch*4*4).reshape(batch,4,4).float(),
    }


def test_reversed_queries_align_all_query_tensors_and_restore_outputs():
    enc = _encoded()
    reversed_enc = diagnostic.reverse_encoded_queries(enc)
    for key in ("init", "query_roles", "query_mask"):
        assert torch.equal(reversed_enc[key], enc[key].flip(1))
        assert torch.equal(diagnostic.undo_reverse_queries(reversed_enc[key]), enc[key])
    assert torch.equal(reversed_enc["facts"], enc["facts"])
    logits = torch.arange(2*16*4).reshape(2,16,4)
    assert torch.equal(diagnostic.undo_reverse_queries(diagnostic.reverse_queries(logits)), logits)


def test_singleton_views_slice_exact_query_slot_and_preserve_static_graph_context():
    enc = _encoded(batch=2)
    noise = torch.arange(2*16*3).reshape(2,16,3).float()
    singles, one_noise = diagnostic.singleton_views(enc, noise)
    assert singles["init"].shape == (32,1,3)
    assert singles["query_roles"].shape == (32,1,5)
    assert singles["query_mask"].shape == (32,1)
    assert one_noise.shape == (32,1,3)
    for b in range(2):
        for q in range(16):
            idx=b*16+q
            assert torch.equal(singles["init"][idx,0], enc["init"][b,q])
            assert torch.equal(singles["query_roles"][idx,0], enc["query_roles"][b,q])
            assert torch.equal(one_noise[idx,0], noise[b,q])
            assert torch.equal(singles["facts"][idx], enc["facts"][b])
            assert torch.equal(singles["memory"][idx], enc["memory"][b])
            assert torch.equal(singles["ground"][idx], enc["ground"][b])
            assert torch.equal(singles["grades"][idx], enc["grades"][b])


def test_selected_rows_are_systematic_across_source_groups_and_cpu_inputs_are_five_features():
    rows=json.loads(DATA.read_text(encoding="utf-8"))
    selected=diagnostic.validate_rows(rows)
    assert selected["indices"] == [0,8,16,24,32,40,48,56]
    chosen=[rows[i] for i in selected["indices"]]
    assert [r["original_sampling_target_class"] for r in chosen] == [0,0,1,1,2,2,3,3]
    assert all("target_class" not in r and len(r["queries"]) == 16 for r in chosen)
    assert selected["projection_sha256"]


def test_seed17_noise_reconstructs_original_four_row_panel_slices_exactly():
    indices=[0,8,16,24,32,40,48,56]
    noise=diagnostic.paired_noise_for_indices(indices,z_dim=7,device="cpu",seed=17)
    assert noise.shape == (8,16,7)
    for pos,index in enumerate(indices):
        start=(index//4)*4
        g=torch.Generator(device="cpu").manual_seed(17+start)
        expected=torch.randn((4,16,7),generator=g)[index-start:index-start+1]
        assert torch.equal(noise[pos:pos+1],expected)


def test_forward_budgets_distinguish_logical_exposures_from_actual_batched_calls():
    assert diagnostic.expected_forward_counts(8) == {"logical_forwards":104,"actual_batched_calls":26}
    budget=diagnostic.ForwardBudget(200,10**12)
    for _ in range(2):
        budget.charge(4)  # encode per four-graph batch
    for _ in range(24):
        budget.charge(4)  # 3 modes * 4 denoise calls per four-graph batch
    assert (budget.logical_forwards,budget.actual_batched_calls)==(104,26)
    with pytest.raises(RuntimeError,match="cap exceeded"):
        for _ in range(97): budget.charge(1)


def test_output_contract_allows_owned_supervisor_logs_but_rejects_reused_results():
    representative={"plan.json","seal.json","readiness.json","supervisor.stdout.log",
        "supervisor.stderr.log","worker.stdout.log","worker.stderr.log","code_snapshot",
        "supervisor","control"}
    diagnostic.validate_output_names(representative)
    with pytest.raises(FileExistsError,match="output conflict"):
        diagnostic.validate_output_names(representative|{"diagnostic_result.json"})
    with pytest.raises(FileExistsError,match="output conflict"):
        diagnostic.validate_output_names(representative|{"clock.json"})


def test_draft_parent_and_no_training_no_holdout_contract():
    plan=json.loads((ROOT/'research/artifacts/query_interference_diagnostic_20261027_v1/cpu_preflight_draft.json')
                    .read_text(encoding='utf-8'))
    assert plan['authorized'] is False
    assert plan['parent_payload_sha256']==diagnostic.EXPECTED_PARENT
    assert plan['selected_dev_indices']==[0,8,16,24,32,40,48,56]
    assert (plan['wall_seconds'],plan['max_logical_forwards'],plan['expected_logical_forwards'],plan['expected_actual_batched_calls'])==(120,200,104,26)
    assert plan['optimizer_allowed'] is False and plan['checkpoint_write_allowed'] is False
    assert plan['e2_final_split_accessed'] is False


def test_duplicate_detection_identifies_invoked_worker_not_supervisor_worker_argument():
    worker = str(ROOT / 'scripts/run_query_interference_diagnostic_20261027.py')
    owner = str(ROOT / 'scripts/supervise_query_interference_diagnostic_20261027.py')
    paths = [worker.lower().replace('\\', '/')]
    assert supervisor.runs_active_worker(f'"python.exe" -B -X utf8 "{worker}" --plan plan.json', paths)
    assert not supervisor.runs_active_worker(f'"python.exe" -B -X utf8 "{owner}" --worker "{worker}" --plan plan.json', paths)
    assert not supervisor.runs_active_worker('python.exe -m pytest', paths)
