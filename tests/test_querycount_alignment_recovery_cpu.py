import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
import run_querycount_alignment_20261027_recovery_v2 as worker
import querycount_alignment_v1 as data


def _views():
    return {q: json.loads((ROOT / "research/artifacts/querycount_alignment_20261027_v1" / f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in (1, 4, 8, 16)}


def test_actual_partial9_fullstate_and_next_attempt_batch_cpu():
    plan = {"recovery_marker": str(worker.RECOVERY_MARKER), "recovery_payload_sha256": worker.RECOVERY_PAYLOAD_SHA256}
    result = worker.validate_recovery_payload(plan, _views())
    assert result["payload_sha256"] == "5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675"
    assert result["algorithm_state_sha256"] == "d346ee6410ca7627fd63e4dbb82d04057c715cd7765a0eb33e28c9917910034a"
    assert (result["updates"], result["attempts"], result["forward_attempts"], result["sampler_cursor"]) == (9, 9, 816, 72)
    assert result["next_attempt10_batch"] == [(139,8),(779,4),(1010,8),(957,4),(747,1),(213,1),(407,16),(481,16)]
    assert result["baseline_present"] and result["optimizer_populated"] and result["all_rng_fields_present"]


def test_partial9_recovery_contract_uses_global_counters_and_remaining_budget():
    plan = {"phase":2,"target_successful_updates":128,"max_attempts":119,"successful_at_start":9,"attempts_at_start":9,"seed":25,
        "parent_updates":256,"parent_attempts":256,"parent_sha256":worker.PARENT_SHA256,
        "parent_query_mix_updates":256,"ancestor_active_balanced_updates":128,"ancestor_role8_updates":128,
        "ancestor_e2_updates":128,"ancestor_e1_updates":1440,"phase_steps":[480,320,256],
        "trainable_prefixes":worker.TRAINABLE_PREFIXES,"parent_optimizer_imported":False,"parent_rng_imported":False,"e2_final_split_evaluated":False}
    worker.validate_phase2_contract(plan)
    plan["attempts_at_start"] = 0
    with pytest.raises(ValueError): worker.validate_phase2_contract(plan)


def test_recursive_state_equality_rejects_rng_or_optimizer_difference():
    import torch
    left={"optim":{"x":torch.tensor([1.0])},"rng":[1,2]}
    right={"optim":{"x":torch.tensor([1.0])},"rng":[1,2]}
    assert worker.exact_tree_equal(left,right)
    right["rng"][1]=3
    assert not worker.exact_tree_equal(left,right)

def test_initialize_recovery_calls_inherited_restore_preserves_state_and_saves_before_step(tmp_path, monkeypatch):
    import copy, types, random
    import numpy as np
    from fdt_local.runner import Engine
    from fdt_local.storage import CheckpointStore
    plan=json.loads((ROOT/"runs/querycount_alignment_20261027_recovery_v2/plan.json").read_text(encoding="utf-8"))
    _, source=CheckpointStore(worker.RECOVERY_MARKER.parent).verify(worker.RECOVERY_MARKER)
    artifact=Path(plan["artifact_dir"])
    rows={"train_views":{q:json.loads((artifact/f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in (1,4,8,16)},
          "dev_views":{q:json.loads((artifact/f"dev_q{q:02d}.json").read_text(encoding="utf-8")) for q in (1,4,8,16)},
          "uniform_dev":json.loads(Path(plan["uniform_dev_data_path"]).read_text(encoding="utf-8")),
          "old_role_dev":json.loads(Path(plan["old_role_dev_path"]).read_text(encoding="utf-8")),
          "transfer_views":{q:json.loads(Path(plan["transfer_data_paths"][str(q)]).read_text(encoding="utf-8")) for q in (1,4,8,16)}}
    rows["train"]=[r for q in (1,4,8,16) for r in rows["train_views"][q]]
    rows["dev"]=[r for q in (1,4,8,16) for r in rows["dev_views"][q]]
    class Model:
        def __init__(self): self.state={}; self.params={n:torch.nn.Parameter(torch.zeros(1)) for n in source["trainable_names"]}
        def load_state_dict(self,state,strict=True): self.state=state; return types.SimpleNamespace(missing_keys=[],unexpected_keys=[])
        def state_dict(self): return self.state
        def named_parameters(self): return iter(self.params.items())
        def parameters(self): return iter(self.params.values())
    class Scaler:
        def __init__(self): self.s={}
        def load_state_dict(self,s): self.s=copy.deepcopy(s)
        def state_dict(self): return copy.deepcopy(self.s)
    class Harness(worker.QueryBoundEngine):
        def _optimizer(self):
            group=source["optimizer"]["param_groups"][0]
            params=list(self.model.parameters())
            self.optimizer=torch.optim.AdamW([{"params":params,"lr":group["lr"],"base_lr":group["base_lr"]}],weight_decay=.01,foreach=False)
            self.scaler=Scaler()
    h=Harness.__new__(Harness); h.model=Model(); h.contract=source["contract"]
    h.sampler=torch.Generator().manual_seed(3); h.noise=types.SimpleNamespace(get_state=lambda:source["noise_rng"],set_state=lambda value:None)
    h.train_data=rows["train"]; h.querycount_train_views=rows["train_views"]
    h.cuda_rng=copy.deepcopy(source["cuda_rng"])
    h.payload=lambda: {
      "model":h.model.state_dict(),"optimizer":h.optimizer.state_dict(),"scaler":h.scaler.state_dict(),
      "trainable_names":[n for n,p in h.model.named_parameters() if p.requires_grad],
      **{k:getattr(h,k) for k in ("phase","phase_step","updates","attempts","exposure","query_exposure","ground_exposure","gates","history")},
      "sampler_rng":h.sampler.get_state(),"noise_rng":h.noise.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":h.cuda_rng,
      "python_rng":random.getstate(),"numpy_rng":worker_numpy_state(),"scheduler":source["scheduler"],
      "query_bound_runtime":h.query_bound_runtime,"querycount_sampler_state":h.qc_sampler.state_dict()}
    def worker_numpy_state():
        n=np.random.get_state(); return {"name":n[0],"keys":n[1].tolist(),"position":n[2],"gaussian":n[3],"cached":n[4]}
    original_set=worker.torch.cuda.set_rng_state_all
    monkeypatch.setattr(worker.torch.cuda,"set_rng_state_all",lambda states:setattr(h,"cuda_rng",copy.deepcopy(states)))
    orig_ones=torch.ones
    monkeypatch.setattr(worker.torch,"ones",lambda *a,**kw:orig_ones(*a,**{**kw,"device":"cpu"}) if kw.get("device")=="cuda" else orig_ones(*a,**kw))
    expected=source["contract"]["query_bound_execution"]["query_class_weights"]
    monkeypatch.setattr(worker,"query_class_weights",lambda *a,**k:torch.tensor(expected))
    exp=worker.Experiment.__new__(worker.Experiment)
    exp.plan=plan; exp.run_dir=tmp_path/"model"; exp.out=tmp_path; exp.engine=h; exp.train_rows=rows["train"]
    exp.plan_path=ROOT/"runs/querycount_alignment_20261027_recovery_v2/plan.json"
    exp.budget=types.SimpleNamespace(attempted=0)
    saved=[]
    exp.progress=lambda *a,**k:None
    exp.save=lambda tag=None:saved.append((tag,worker.algorithm_state_digest(h.payload())))
    exp._initialize_recovery(rows)
    assert saved and saved[0][0]=="imported_recovery_state"
    assert saved[0][1]==worker.RECOVERY_STATE_SHA256
    assert (h.updates,h.attempts,exp.budget.attempted,h.qc_sampler.state_dict()["cursor"])==(9,9,816,72)
    assert exp.baseline==source["query_bound_runtime"]["baseline"]
    assert h.optimizer.state and h.scaler.state_dict()==source["scaler"]
    assert h.optimizer.step.__name__=="step"


import torch

