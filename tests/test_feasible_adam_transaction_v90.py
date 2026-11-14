"""Real CPU AdamW/enabled-scaler transactions on explicit affine fixtures."""
import copy
import pytest
import torch
from research.actual_update_retention_v80 import TrainDecision, _equal
from research.feasible_adam_transaction_v90 import feasible_adam_proposal, ReferenceObservation
from research.feasible_readout_delta_v89 import LinearConstraint
from research.last_linear_objective_v79 import LAST_LINEAR_NAMES

A = 'a' * 64


def fixture(bias=(0., 0.)):
    w = torch.nn.Parameter(torch.tensor([[.1, -.1], [0., 0.]], dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float64))
    opt = torch.optim.AdamW([w, b], lr=.2, weight_decay=.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=1024.)
    return {LAST_LINEAR_NAMES[0]: w, LAST_LINEAR_NAMES[1]: b}, opt, scaler


def gradients(named, opt, scaler, overflow=False):
    opt.zero_grad(set_to_none=True)
    w, b = named.values()
    g = torch.tensor([[1., -1.], [0., 0.]], dtype=w.dtype)
    loss = (w * g).sum() + b.sum() * 0
    scaler.scale(loss).backward()
    if overflow:
        w.grad.fill_(float('inf'))
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(list(named.values()), 1.)


def reference(named, guard_bias=False, overflow_count=False):
    w, b = named.values()
    points = [(torch.tensor([1., 0.], dtype=w.dtype), 0, 1, .05, 'CPU_TRAIN:A')]
    if guard_bias:
        points.append((torch.zeros(2, dtype=w.dtype), 1, 0, 0., 'CPU_TRAIN:other'))
    def inspect():
        constraints, diag = [], []
        for h, g, j, floor, name in points:
            gap = float(torch.dot(w[g] - w[j], h) + b[g] - b[j])
            if gap < floor:
                constraints.append(LinearConstraint(0, g, j, h, floor, name, 0, A))
                diag.append(dict(source_id=name, gap=gap, floor=floor, gold=g, rival=j))
        if overflow_count:
            return ReferenceObservation(TrainDecision(False, dict(violations=17)), (), 17, tuple([dict(source_id='CPU_TRAIN:A')] * 16), A)
        return ReferenceObservation(TrainDecision(not constraints, dict(violations=len(constraints))),
                                    tuple(constraints), len(constraints), tuple(diag), A)
    return inspect


def test_actual_Adam_direction_repaired_and_full_proposal_moments_retained():
    named, opt, scaler = fixture()
    raw, rawopt, rawscaler = fixture()
    gradients(named, opt, scaler); gradients(raw, rawopt, rawscaler)
    rawscaler.step(rawopt); rawscaler.update(); rawopt.zero_grad(set_to_none=True)
    out = feasible_adam_proposal(opt, named, scaler, reference(named), anchor_SHA=A, classes=2)
    assert out['accepted'] and out['acceptance'] == 'bounded_corrected_actual_proposal'
    assert len(out['reference_checks']) == 2 and not out['reference_checks'][0]['accept']
    assert out['reference_checks'][1]['accept'] and out['correction']['corrections'] == 1
    w, b = named.values()
    assert float((w[0, 0] - w[1, 0] + b[0] - b[1]).detach()) >= .05
    assert float((w[0, 1] - w[1, 1] + b[0] - b[1]).detach()) > 0
    assert not _equal(list(named.values()), list(raw.values()))
    assert _equal(opt.state_dict(), rawopt.state_dict()) and _equal(scaler.state_dict(), rawscaler.state_dict())
    assert all(p.grad is None for p in named.values())
    # Public optimizer/scaler state recovers the exact NEXT proposed transaction,
    # including populated moments and enabled-scaler growth tracker.
    snap = copy.deepcopy((list(named.values()), opt.state_dict(), scaler.state_dict()))
    recovered, ropt, rscale = fixture()
    with torch.no_grad():
        for dst, src in zip(recovered.values(), snap[0]): dst.copy_(src)
    ropt.load_state_dict(snap[1]); rscale.load_state_dict(snap[2])
    for n, o, s in ((named, opt, scaler), (recovered, ropt, rscale)):
        gradients(n, o, s)
        feasible_adam_proposal(o, n, s, reference(n), anchor_SHA=A, classes=2)
    assert _equal(list(named.values()), list(recovered.values()))
    assert _equal(opt.state_dict(), ropt.state_dict()) and _equal(scaler.state_dict(), rscale.state_dict())


def test_active_correction_harms_other_context_and_full_check_rolls_back():
    named, opt, scaler = fixture((0., .02))
    gradients(named, opt, scaler)
    before = copy.deepcopy((list(named.values()), opt.state_dict(), scaler.state_dict()))
    out = feasible_adam_proposal(opt, named, scaler, reference(named, guard_bias=True), anchor_SHA=A, classes=2)
    assert not out['accepted'] and out['rejection'] == 'full_reference_rejected_active_only_repair'
    assert len(out['reference_checks']) == 2 and out['reference_checks'][1]['diagnostics'][0]['source_id'] == 'CPU_TRAIN:other'
    assert _equal(list(named.values()), before[0]) and _equal(opt.state_dict(), before[1])
    assert scaler.state_dict()['_growth_tracker'] == before[2]['_growth_tracker'] + 1


def test_17_blockers_are_not_truncated_to_16_for_correction():
    named, opt, scaler = fixture()
    gradients(named, opt, scaler)
    before = copy.deepcopy((list(named.values()), opt.state_dict()))
    out = feasible_adam_proposal(opt, named, scaler, reference(named, overflow_count=True), anchor_SHA=A, classes=2)
    assert not out['accepted'] and out['correction'] is None and len(out['reference_checks']) == 1
    assert out['reference_checks'][0]['blocker_count'] == 17 and len(out['reference_checks'][0]['diagnostics']) == 16
    assert _equal(list(named.values()), before[0]) and _equal(opt.state_dict(), before[1])


@pytest.mark.parametrize('target', ['weights', 'Adam', 'scaler'])
def test_reference_mutation_rolls_back_whole_transaction(target):
    named, opt, scaler = fixture()
    gradients(named, opt, scaler)
    before = copy.deepcopy((list(named.values()), opt.state_dict(), scaler.state_dict()))
    base = reference(named)
    def bad():
        seen = base()
        if target == 'weights':
            with torch.no_grad(): next(iter(named.values())).add_(1)
        elif target == 'Adam':
            next(iter(opt.state.values()))['exp_avg'].add_(1)
        else:
            value = copy.deepcopy(scaler.state_dict()); value['_growth_tracker'] += 1; scaler.load_state_dict(value)
        return seen
    with pytest.raises(ValueError):
        feasible_adam_proposal(opt, named, scaler, bad, anchor_SHA=A, classes=2)
    assert _equal(list(named.values()), before[0]) and _equal(opt.state_dict(), before[1]) and _equal(scaler.state_dict(), before[2])
    assert all(p.grad is None for p in named.values())


def test_STOP_after_corrected_full_check_restores_weights_Adam_and_scaler():
    named, opt, scaler = fixture()
    gradients(named, opt, scaler)
    before = copy.deepcopy((list(named.values()), opt.state_dict(), scaler.state_dict()))
    stages = []
    def stop(value):
        stages.append(value['stage'])
        if value['stage'] == 'corrected_proposal_full_reference_observed':
            raise InterruptedError('CPU cooperative STOP after actual corrected projection')
    with pytest.raises(InterruptedError):
        feasible_adam_proposal(opt, named, scaler, reference(named), anchor_SHA=A, classes=2, event=stop)
    assert 'bounded_direction_correction_requested' in stages
    assert _equal(list(named.values()), before[0]) and _equal(opt.state_dict(), before[1]) and _equal(scaler.state_dict(), before[2])


def test_AMPskip_consumes_scaler_but_does_not_call_reference_or_change_Adam():
    named, opt, scaler = fixture()
    gradients(named, opt, scaler, overflow=True)
    before = copy.deepcopy((list(named.values()), opt.state_dict()))
    def forbidden(): raise AssertionError('no reference call after AMP skip')
    out = feasible_adam_proposal(opt, named, scaler, forbidden, anchor_SHA=A, classes=2)
    assert out['AMP_skip'] and not out['accepted'] and not out['reference_checks']
    assert scaler.get_scale() == 512. and _equal(list(named.values()), before[0]) and _equal(opt.state_dict(), before[1])
