import copy
import pytest
import torch
from research.diagnostic_reference_v90 import DiagnosticReference
from research.readout_precision_policy_v81 import POLICY


def fixture(patches=2):
    y = torch.full((patches, 16), 257, dtype=torch.long)
    y[:, 3] = 65
    mask = y != 257
    features = dict(hidden=torch.zeros(patches, 512), source_ranges=[dict(
        start=0, end=patches, source_id='CPU_TRAIN:a', source_patch_indices=list(range(9, 9 + patches)))])
    calibration = dict(labels=y, top1_mask=mask, weak_floor_mask=mask, critical_mask=mask,
        required_margin=torch.full_like(y, .05, dtype=torch.float32), precision_policy=POLICY)
    z = torch.zeros(patches, 16, 258)
    z[:, 3, 65] = .01
    z[:, 3, 66] = .02
    return features, calibration, z


def test_diagnostics_reuse_full_projection_and_preserve_offsets_and_original_floor():
    f, c, z = fixture()
    inspector = DiagnosticReference(f, c, 'cpu', 'a' * 64)
    seen = inspector.inspect(z)
    assert seen.decision == inspector.decision(z) and seen.blocker_count == 2
    assert seen.constraints[0].byte_offset == 9 * 16 + 3
    assert seen.diagnostics[1]['byte_offset'] == 10 * 16 + 3
    assert seen.diagnostics[0]['gold'] == 65 and seen.diagnostics[0]['rival'] == 66
    assert seen.diagnostics[0]['top1_flip'] and seen.diagnostics[0]['weak_floor_violation']
    before = copy.deepcopy(inspector.c)
    z[:, 3, 65] = .08
    assert inspector.inspect(z).decision.accept
    assert torch.equal(inspector.c['required_margin'], before['required_margin'])
    # Mutable constructor inputs cannot silently change the guarded values.
    c['required_margin'].zero_(); f['hidden'].fill_(999.)
    assert torch.equal(inspector.c['required_margin'], before['required_margin']) and not inspector.hidden.any()


def test_full_count_overflow_does_not_supply_incomplete_repair_constraints():
    f, c, z = fixture(17)
    seen = DiagnosticReference(f, c, 'cpu', 'a' * 64).inspect(z)
    assert seen.blocker_count == 17 and len(seen.diagnostics) == 16 and seen.constraints == ()


def test_nonfinite_reference_cannot_appear_as_passing_comparison():
    f, c, z = fixture()
    z[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError): DiagnosticReference(f, c, 'cpu', 'a' * 64).inspect(z)
