from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import contextlib
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
import run_querycount_alignment_20261027 as worker
import querycount_alignment_v1 as data
import supervise_querycount_alignment_20261027 as supervisor


def test_phase2_contract_is_128_attempts_seed25_and_weights_only():
    plan = {"phase":2,"target_successful_updates":128,"max_attempts":128,"seed":25,
        "parent_updates":256,"parent_attempts":256,"parent_sha256":worker.PARENT_SHA256,
        "parent_query_mix_updates":256,"ancestor_active_balanced_updates":128,
        "ancestor_role8_updates":128,"ancestor_e2_updates":128,"ancestor_e1_updates":1440,
        "phase_steps":[480,320,256],"trainable_prefixes":worker.TRAINABLE_PREFIXES,
        "parent_optimizer_imported":False,"parent_rng_imported":False,"e2_final_split_evaluated":False}
    worker.validate_phase2_contract(plan)
    plan["seed"] = 24
    try: worker.validate_phase2_contract(plan)
    except ValueError: pass
    else: raise AssertionError("wrong stage seed accepted")


def test_nested_batch_indices_reconstruct_query_count_and_graph_exposure():
    train_views = {q: json.loads((ROOT/"research/artifacts/querycount_alignment_20261027_v1"/f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in (1,4,8,16)}
    rows = [r for q in (1,4,8,16) for r in train_views[q]]
    runtime = {"attempted_id_exposure":{k:[0]*8 for k in worker.EXPOSURE_KEYS},
        "successful_id_exposure":{k:[0]*8 for k in worker.EXPOSURE_KEYS},
        "attempted_query_class_exposure":[0]*4,"successful_query_class_exposure":[0]*4,
        "unique_graph_exposure":[0]*1024,"attempted_query_count_exposure":{str(q):0 for q in (1,4,8,16)},
        "successful_query_count_exposure":{str(q):0 for q in (1,4,8,16)}}
    batch=[[0,1,1026,1027],[2052,2053,3078,3079]]
    summary=worker.update_training_exposure(runtime,rows,batch,True)
    assert summary["query_count_exposure"]=={"1":2,"4":2,"8":2,"16":2}
    assert sum(v>0 for v in runtime["unique_graph_exposure"])==8
    assert runtime["attempted_query_count_exposure"]==runtime["successful_query_count_exposure"]


def test_one_finite_sampler_epoch_has_balanced_counts_and_unique_graphs():
    train_views = {q: json.loads((ROOT/"research/artifacts/querycount_alignment_20261027_v1"/f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in (1,4,8,16)}
    a=data.BalancedQueryCountBatchSampler(train_views,seed=25)
    b=data.BalancedQueryCountBatchSampler(train_views,seed=25)
    seen=set(); batches=0
    while True:
        left,right=a.next_batch(),b.next_batch()
        assert left==right
        if left is None: break
        batches+=1
        ids=[r["source_graph_index"] for r in left]
        assert len(set(ids))==8 and not (seen & set(ids))
        seen.update(ids)
        assert Counter(r["query_count"] for r in left)==Counter({1:2,4:2,8:2,16:2})
    assert (batches,len(seen))==(128,1024)


def test_sampler_checkpoint_restores_exact_next_batch():
    views={q:json.loads((ROOT/"research/artifacts/querycount_alignment_20261027_v1"/f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in (1,4,8,16)}
    sampler=data.BalancedQueryCountBatchSampler(views,seed=25)
    first=sampler.next_batch(); state=json.loads(json.dumps(sampler.state_dict()))
    restored=data.BalancedQueryCountBatchSampler.from_state(views,state)
    assert restored.next_batch()==sampler.next_batch()
    assert len(first)==8


def test_query_weights_use_every_declared_prefix_target():
    views={q:json.loads((ROOT/"research/artifacts/querycount_alignment_20261027_v1"/f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in (1,4,8,16)}
    rows=[r for q in (1,4,8,16) for r in views[q]]
    expected=worker.query_class_weights(rows)
    assert len(rows)==4096 and expected.shape==(4,) and abs(float(expected.mean())-1)<1e-6


def test_fresh_scaler_cpu_defaults_and_query_count_limits():
    scaler=worker.torch.amp.GradScaler("cpu",enabled=True,init_scale=1024.0)
    worker.validate_fresh_runtime_state({},set(),set(),[3e-5],scaler.state_dict())
    assert worker.TARGET_UPDATES==128 and worker.MAX_ATTEMPTS==128
    assert worker.DIAGNOSTIC_STEPS==(32,64,96,128)


def _metrics(predictions, labels):
    return worker.compute_metrics(torch.tensor(predictions), torch.tensor(labels),
                                  torch.ones(len(labels), dtype=torch.bool), torch.ones(len(labels), dtype=torch.long))


def _panel(metric, ground=.99, init=.99):
    return {"ground_accuracy":ground,"results":{"ddim_4":metric,"initializer":{"accuracy":init}}}


def test_real_compute_metrics_confusion_reconstructs_errors_and_zero_rule():
    exact=_metrics([0,1,2,3],[0,1,2,3])
    one_error=_metrics([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,0],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])
    assert worker.metric_errors(exact)==0
    assert worker.metric_errors(one_error)==1
    base={}
    for key in ("uniform_dev64","old_role_dev128","old_normal16","old_large16",
                "primary_querymix_dev64","paired_new_ids16"):
        base[key]=_panel(exact)
    for q in (1,4,8,16):
        base[f"querycount_dev_q{q:02d}"]=_panel(exact)
        base[f"transfer_q{q:02d}"]=_panel(exact)
    assert worker.baseline_alignment_action(base,{})[0]=="PURPOSE_ALREADY_MET_NO_OPTIMIZER"
    base["transfer_q04"]=_panel(one_error)
    assert worker.baseline_alignment_action(base,{})[0]=="TRAIN_BOUNDED"


def test_count_balanced_q1_q4_goal_uses_rates_and_discloses_zero_ceiling():
    exact=_metrics([0,1,2,3],[0,1,2,3])
    two=_metrics([0,0,2,3],[0,1,2,3])
    one=_metrics([0,1,2,0],[0,1,2,3])
    base={f"querycount_dev_q{q:02d}":_panel(two if q==1 else one if q==4 else exact) for q in (1,4,8,16)}
    current={f"querycount_dev_q{q:02d}":_panel(exact) for q in (1,4,8,16)}
    goals=worker.development_goal(base,current,current)
    assert goals["count_balanced_q1_q4_error_rate_halved"] is True
    zero=worker.development_goal(current,current,current)
    assert zero["zero_error_baseline_ceiling"] is True
    assert zero["count_balanced_q1_q4_error_rate_halved"] is None


def test_new_relative_panels_keep_ground_and_initializer_safety_floors():
    exact=_metrics([0,1,2,3],[0,1,2,3])
    baseline={f"querycount_dev_q{q:02d}":_panel(exact) for q in (1,4,8,16)}
    baseline.update({f"transfer_q{q:02d}":_panel(exact) for q in (1,4,8,16)})
    baseline.update({key:_panel(exact) for key in ("uniform_dev64","old_role_dev128","old_normal16","old_large16","primary_querymix_dev64","paired_new_ids16")})
    current=json.loads(json.dumps(baseline)); current["transfer_q04"]= _panel(exact,ground=.94,init=.94)
    streak, failures=worker.retention_streak(0,baseline,current,{})
    assert streak==1 and {f["metric"] for f in failures} >= {"ground_accuracy","initializer_accuracy"}


def test_legacy_retention_uses_own_bound_baseline_not_original_full_parent():
    base_metric=_metrics([0,1,2,3],[0,1,2,3])
    one_labels=[0,1,2,3]*25
    one_predictions=one_labels.copy(); one_predictions[-1]=2
    one_error=_metrics(one_predictions,one_labels)
    baseline={key:_panel(base_metric,.99,.99) for key in ("uniform_dev64","old_role_dev128","old_normal16","old_large16","primary_querymix_dev64","paired_new_ids16")}
    current=json.loads(json.dumps(baseline))
    current["old_normal16"]=_panel(one_error,.99,.99)
    parent={key:_panel(base_metric,.99,.99) for key in baseline}
    failures=worker.retention_failures(baseline,current,parent)
    assert not failures


def test_cpu_fake_engine_step_consumes_one_balanced_batch_and_serializes_cursor(monkeypatch):
    class Model:
        def __init__(self): self.weight=torch.nn.Parameter(torch.tensor(1.0))
        def train(self): pass
        def parameters(self): return [self.weight]
        def named_parameters(self): return [("velocity.weight",self.weight)]
        def encode(self,features): return {"stub":self.weight}
        def target(self,features,labels): return labels.float()
    class Optimizer:
        def __init__(self,p): self.p=p; self.param_groups=[{"base_lr":3e-5,"lr":3e-5}]
        def zero_grad(self,set_to_none=True): self.p.grad=None
        def step(self): pass
    class Scaler:
        def scale(self,loss): return loss
        def unscale_(self,opt): pass
        def step(self,opt): opt.step()
        def update(self): pass
        def get_scale(self): return 1024.
    rows=[{"query_count":q,"source_graph_index":i,"facts":[],"queries":[],"labels":[]} for i,q in enumerate((1,1,4,4,8,8,16,16))]
    class Sampler:
        def __init__(self): self.cursor=0
        def next_batch(self): self.cursor+=8; return rows
        def state_dict(self): return {"cursor":self.cursor,"rng_state":[1,2,3]}
    engine=object.__new__(worker.QueryBoundEngine)
    engine.out=ROOT; engine.estimate_checkpoint=lambda:0; engine.attempts=engine.updates=engine.phase_step=engine.exposure=0
    engine.model=Model(); engine.optimizer=Optimizer(engine.model.weight); engine.scaler=Scaler(); engine.qc_sampler=Sampler()
    engine.row_index={(r["query_count"],r["source_graph_index"]):i for i,r in enumerate(rows)}
    engine.train_data=rows; engine.cfg=None; engine.plan=SimpleNamespace(accumulation=2,phase_steps=(480,320,256))
    engine.phase=2; engine.noise=None; engine.weights=None; engine.query_exposure=[0]*4; engine.history=[]
    monkeypatch.setattr(worker,"ensure_space",lambda *a,**k:None)
    monkeypatch.setattr(worker,"collate",lambda batch,cfg,device: ({"query_mask":torch.ones(len(batch),16)},
        {"labels":torch.zeros(len(batch),16,dtype=torch.long)},1))
    monkeypatch.setattr(worker,"diffusion_loss",lambda model,*args: (model.weight.square(),{"loss":model.weight.detach()}))
    monkeypatch.setattr(worker.torch,"autocast",lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(worker.torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(worker.torch.cuda,"max_memory_allocated",lambda:0)
    monkeypatch.setattr(worker.torch.cuda,"max_memory_reserved",lambda:0)
    entry=engine.step()
    assert entry["batch_query_counts"]==[1,1,4,4,8,8,16,16]
    assert len(set(entry["batch_source_graph_indices"]))==8
    assert entry["sampler_state"]["cursor"]==8 and json.dumps(entry["sampler_state"])


def test_restore_rehydrates_balanced_sampler_cursor_and_runtime(monkeypatch):
    class TinySampler:
        def __init__(self,state): self.state=state
        @classmethod
        def from_state(cls,views,state): return cls(state)
        def state_dict(self): return self.state
    monkeypatch.setattr(worker.querycount_data.BalancedQueryCountBatchSampler,"from_state",TinySampler.from_state)
    def fake_restore(self,payload): self.restored=True
    monkeypatch.setattr(worker.Engine,"restore",fake_restore)
    rows=[{"query_count":q,"source_graph_index":g} for q in (1,4,8,16) for g in (0,1)]
    engine=object.__new__(worker.QueryBoundEngine); engine.train_data=rows; engine.querycount_train_views={}
    runtime={"attempted_query_count_exposure":{"1":2}}
    engine.restore({"querycount_sampler_state":{"cursor":8},"query_bound_runtime":runtime})
    assert engine.restored and engine.qc_sampler.state["cursor"]==8
    assert engine.query_bound_runtime==runtime and engine.row_index[(16,1)]==7


def test_cpu_preflight_authorization_marker_accepts_draft_and_sealed_boolean():
    worker.validate_authorization({"authorized":False})
    worker.validate_authorization({"authorized":True})
    with pytest.raises(ValueError): worker.validate_authorization({"authorized":False},actual_run=True)
    worker.validate_authorization({"authorized":True},actual_run=True)


def test_supervisor_terminal_classifies_zero_optimizer_purpose_complete():
    assert supervisor.classify_worker_terminal(0,"PURPOSE_ALREADY_MET_NO_OPTIMIZER",None,None,None)=="PURPOSE_ALREADY_MET_NO_OPTIMIZER"
    assert supervisor.classify_worker_terminal(0,"COMPLETE",None,None,None)=="COMPLETE"
    assert supervisor.classify_worker_terminal(0,"PURPOSE_ALREADY_MET_NO_OPTIMIZER",None,{"ProcessId":5},None)=="ERROR"


def test_supervisor_deadline_begins_at_owner_invocation_before_preflight():
    mono,unix=100.0,1000.0
    deadline_mono,deadline_unix=supervisor.immutable_owner_deadline(mono,unix)
    simulated_preflight_end=mono+180
    assert deadline_mono==1000.0 and deadline_unix==1900.0
    assert simulated_preflight_end < deadline_mono
    assert supervisor.immutable_owner_deadline(mono,unix)==(deadline_mono,deadline_unix)



