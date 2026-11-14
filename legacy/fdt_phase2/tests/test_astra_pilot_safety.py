import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_retention_requires_both_metrics_against_both_parents():
    from run_astra_bounded_pilot import retention_fail
    def r(a,b):return {'fixed52':{'row_mean_nll':a,'token_weighted_nll':b}}
    assert not retention_fail(r(1.009,2.019),[r(1,2),r(1.02,2.02)])
    assert retention_fail(r(1.011,2),[r(1,2),r(1.02,2.02)])
    assert retention_fail(r(1,2.021),[r(1.02,2.02),r(1,2)])


def test_attempt_budget_charges_interrupted_attempts_before_any_update(tmp_path):
    from run_astra_bounded_pilot import reserve_attempt,PilotStop
    state={'step':0,'cursor':0}
    reserve_attempt(tmp_path,[{'supervised':249999,'active':900000}],state)
    with pytest.raises(PilotStop,match='RESERVED_SUPERVISED_BUDGET'):
        reserve_attempt(tmp_path,[{'supervised':2,'active':2}],state)
    # The failed reservation did not consume budget; the original reservation
    # remains charged even though this fixture performed no optimizer update.
    assert len(list((tmp_path/'attempts').glob('*.json')))==1
    result=reserve_attempt(tmp_path,[{'supervised':1,'active':1}],state)
    assert result['supervised']==250000


def test_pending_retention_failure_cannot_be_skipped_after_restore(tmp_path,monkeypatch):
    import run_astra_bounded_pilot as pilot
    state={'pending_milestone':50000,'source_regressions':{}}
    def save_model(path,*args):
        path.write_bytes(b'model')
        pilot.save(path.with_suffix('.sha256.json'),{'sha256':pilot.digest(path)})
    monkeypatch.setattr(pilot,'save_model',save_model)
    monkeypatch.setattr(pilot,'retention',lambda *a:{'fixed52':{'row_mean_nll':2.,'token_weighted_nll':2.},'sources':{}})
    baseline={'fixed52':{'row_mean_nll':1.,'token_weighted_nll':1.}}
    for _ in range(2):
        with pytest.raises(pilot.PilotStop,match='FIXED52_RETENTION'):
            pilot.pending_gate(None,None,state,tmp_path,{},None,{},1e15,[baseline])
        assert state['pending_milestone']==50000


def test_primary_holdout_has_paired_changed_and_missing_evidence():
    from astra_pilot_holdout import make_holdout
    cases=make_holdout();assert len(cases)==192
    positives=[r for r in cases if r['family']=='fact' and r['variant']!='omitted']
    assert len(positives)==96
    for r in positives:assert r['expected'] in r['evidence_span'] and r['evidence_span'] in r['prompt']
    omitted=[r for r in cases if r.get('variant')=='omitted']
    assert len(omitted)==48 and all(r['expected'] is None and r['evidence_span'] is None for r in omitted)


def test_pilot_stored_int32_labels_become_loss_compatible_long():
    import torch
    from run_astra_bounded_pilot import move_batch
    group={'input_ids':torch.tensor([[1,4,2]],dtype=torch.int32),
           'labels':torch.tensor([[-100,4,2]],dtype=torch.int32),
           'attention_mask':torch.tensor([[1,1,1]],dtype=torch.int32)}
    result=move_batch(group,torch.device('cpu'))
    assert all(v.dtype==torch.int64 for v in result.values())
    logits=torch.randn(1,3,8,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,8),result['labels'][:,1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
