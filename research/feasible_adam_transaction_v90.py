"""One actual AdamW proposal, at most one bounded directional correction.

Only the last Linear is mutable. Accepted corrected weights retain the actual
proposal's Adam moments; this is projected Adam, not unmodified AdamW. Full
rejection restores weights/Adam while the finite-attempt scaler advances.
Exceptions restore weights/Adam/scaler and clear gradients. Consumed rows,
RNG and budget are owned by the caller and NEVER replayed by this primitive.
"""
import copy
import math
from dataclasses import dataclass
import torch
from research.actual_update_retention_v80 import TrainDecision, _equal
from research.last_linear_objective_v79 import LAST_LINEAR_NAMES
from research.feasible_readout_delta_v89 import correct_actual_delta


@dataclass(frozen=True)
class ReferenceObservation:
    decision: TrainDecision
    constraints: tuple
    blocker_count: int
    diagnostics: tuple
    anchor_SHA: str


def feasible_adam_proposal(optimizer, named_parameters, scaler, reference, *,
                           anchor_SHA, guard=lambda: None, event=lambda x: None,
                           classes=258):
    """Caller has accumulated five micros, unscaled and clipped ONCE already."""
    if (list(named_parameters) != list(LAST_LINEAR_NAMES)
            or not isinstance(optimizer, torch.optim.AdamW)
            or not scaler.is_enabled() or not isinstance(anchor_SHA, str)
            or len(anchor_SHA) != 64):
        raise ValueError('exact lastLinear2 AdamW, enabled scaler and immutable anchor')
    params = list(named_parameters.values())
    owned = [p for group in optimizer.param_groups for p in group['params']]
    if (len(owned) != 2 or {id(p) for p in owned} != {id(p) for p in params}
            or any(not p.requires_grad for p in params)
            or params[0].ndim != 2 or params[1].ndim != 1
            or len(params[0]) != len(params[1]) or len(params[1]) % classes
            or len({p.untyped_storage().data_ptr() for p in params}) != 2):
        raise ValueError('independent final Linear parameters and ownership')
    before = [p.detach().clone() for p in params]
    old_adam = copy.deepcopy(optimizer.state_dict())
    old_scaler = copy.deepcopy(scaler.state_dict())
    old_scale = scaler.get_scale()
    receipt = dict(accepted=False, AMP_skip=False, Adam_proposals=1,
                   Adam_steps_committed=0, method='actual_Adam_delta_bounded_correction_v90',
                   anchor_SHA=anchor_SHA, full_reference_cap=2, reference_checks=[],
                   correction=None, Adam_moments='retain full actual proposal ONLY on accepted original/corrected weights')

    def install(values):
        with torch.no_grad():
            for p, value in zip(params, values):
                p.copy_(value)

    def rollback(include_scaler):
        install(before)
        optimizer.load_state_dict(old_adam)
        if include_scaler:
            # Public AMP update clears the per-optimizer unscale/step stage.
            scaler.update(new_scale=float(old_scale))
            scaler.load_state_dict(old_scaler)
        optimizer.zero_grad(set_to_none=True)

    try:
        event(dict(stage='actual_Adam_requested'))
        guard()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            if (not all(torch.equal(p, v) for p, v in zip(params, before))
                    or not _equal(optimizer.state_dict(), old_adam)):
                raise ValueError('AMP skip mutated weights/Adam')
            receipt['AMP_skip'] = True
            optimizer.zero_grad(set_to_none=True)
            return receipt
        proposal = [p.detach().clone() for p in params]
        proposal_adam = copy.deepcopy(optimizer.state_dict())
        proposal_scaler = copy.deepcopy(scaler.state_dict())
        if any(not torch.isfinite(p).all() for p in proposal):
            raise FloatingPointError('nonfinite actual Adam proposal')

        def observe(stage):
            event(dict(stage=stage))
            guard()
            expected = [p.detach().clone() for p in params]
            with torch.no_grad():
                seen = reference()
            if (not all(torch.equal(p, v) for p, v in zip(params, expected))
                    or not _equal(optimizer.state_dict(), proposal_adam)
                    or not _equal(scaler.state_dict(), proposal_scaler)):
                raise ValueError('reference mutated parameter/Adam/scaler transaction')
            if (not isinstance(seen, ReferenceObservation)
                    or not isinstance(seen.decision, TrainDecision)
                    or type(seen.decision.accept) is not bool
                    or seen.anchor_SHA != anchor_SHA
                    or type(seen.blocker_count) is not int or seen.blocker_count < 0
                    or not isinstance(seen.constraints, tuple)
                    or not isinstance(seen.diagnostics, tuple)
                    or len(seen.diagnostics) != min(16, seen.blocker_count)
                    or len(seen.constraints) != (seen.blocker_count if seen.blocker_count <= 16 else 0)
                    or seen.decision.accept != (seen.blocker_count == 0)
                    or any(c.anchor_SHA != anchor_SHA for c in seen.constraints)
                    or not seen.decision.measurements):
                raise ValueError('explicit complete finite TRAIN observation')
            measurements = {k: float(v) for k, v in seen.decision.measurements.items()}
            if any(not isinstance(k, str) or not math.isfinite(v) for k, v in measurements.items()):
                raise FloatingPointError('finite TRAIN reference measurements')
            receipt['reference_checks'].append(dict(stage=stage, accept=seen.decision.accept,
                measurements=measurements, blocker_count=seen.blocker_count,
                diagnostics=copy.deepcopy(seen.diagnostics),
                diagnostic_scope='first16 in fixed bank order; active correction forbidden if total>16'))
            event(dict(stage=stage + '_observed', observation=copy.deepcopy(receipt['reference_checks'][-1])))
            guard()
            return seen

        first = observe('actual_proposal_full_reference')
        if first.decision.accept:
            receipt.update(accepted=True, Adam_steps_committed=1, acceptance='original_actual_proposal')
        elif first.blocker_count > 16:
            receipt['rejection'] = 'active_constraint_cap_exceeded_no_truncation_no_retry'
        else:
            def full_corrected(weight, bias):
                install((weight, bias))
                return observe('corrected_proposal_full_reference').decision
            event(dict(stage='bounded_direction_correction_requested', blocker_count=first.blocker_count))
            guard()
            corrected = correct_actual_delta(*before, *proposal, first.constraints,
                                             full_corrected, classes=classes)
            receipt['correction'] = corrected['receipt']
            event(dict(stage='bounded_direction_correction_finished', correction=copy.deepcopy(corrected['receipt'])))
            guard()
            if corrected['weight'] is not None:
                install((corrected['weight'], corrected['bias']))
                receipt.update(accepted=True, Adam_steps_committed=1, acceptance='bounded_corrected_actual_proposal')
            else:
                receipt['rejection'] = corrected['receipt']['status']
        if not receipt['accepted']:
            rollback(False)
        else:
            if not _equal(optimizer.state_dict(), proposal_adam):
                raise ValueError('accepted proposal moments changed')
            optimizer.zero_grad(set_to_none=True)
        if len(receipt['reference_checks']) > 2:
            raise ValueError('fixed full-reference call cap')
        return receipt
    except BaseException:
        rollback(True)
        raise
