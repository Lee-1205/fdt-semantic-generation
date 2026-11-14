"""Full finite TRAIN guard and bounded blocker IDs from the SAME projection.

No extra forward for diagnostics. Targets/floors/features are detached training
data, not inference memory. A new strongest rival after correction is detected
by the mandatory second FULL check; no automatic constraint/call-cap growth.
"""
import copy
import torch
from research.critical_reference_device_v86 import DeviceReference
from research.feasible_readout_delta_v89 import LinearConstraint
from research.feasible_adam_transaction_v90 import ReferenceObservation
from research.actual_update_retention_v80 import TrainDecision


class DiagnosticReference(DeviceReference):
    def __init__(self, features, calibration, device, anchor_SHA):
        super().__init__(features, calibration, device)
        if not isinstance(anchor_SHA, str) or len(anchor_SHA) != 64:
            raise ValueError('immutable anchor identity')
        self.anchor_SHA = anchor_SHA
        self.hidden = self.hidden.detach().clone()
        self.c = {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                  for k, v in self.c.items()}
        self.patch_sources = []
        cursor = 0
        for r in features['source_ranges']:
            if (r['start'] != cursor or r['end'] - r['start'] != len(r['source_patch_indices'])
                    or not r['source_id']):
                raise ValueError('complete source-position mapping')
            self.patch_sources.extend((r['source_id'], p) for p in r['source_patch_indices'])
            cursor = r['end']
        if cursor != len(self.hidden):
            raise ValueError('full reference source mapping')

    @torch.no_grad()
    def inspect(self, z):
        if z.dtype != torch.float32 or not torch.isfinite(z).all():
            raise FloatingPointError('genuine finite full-FP32 reference logits')
        c = self.c
        y = c['labels']
        other = z.clone()
        other.scatter_(-1, y[..., None], -torch.inf)
        rival_score, rival = other.max(-1)
        gap = z.gather(-1, y[..., None]).squeeze(-1) - rival_score
        flipped = c['top1_mask'] & (z.argmax(-1) != y)
        eroded = c['weak_floor_mask'] & (gap < c['required_margin'])
        critical = c['critical_mask']
        decision = TrainDecision(not bool((flipped | eroded).any()), dict(
            new_critical_flips=int((flipped & critical).sum()),
            new_noncritical_EOS_flips=int((flipped & ~critical).sum()),
            weak_critical_margin_violations=int(eroded.sum()),
            protected_top1=int(c['top1_mask'].sum()),
            protected_weak_critical=int(c['weak_floor_mask'].sum()), bank_patches=len(z)))
        blockers = (flipped | eroded).nonzero()
        count = len(blockers)
        constraints = []
        diagnostics = []
        for patch, role in blockers[:16].cpu().tolist():
            source_id, source_patch = self.patch_sources[patch]
            gold = int(y[patch, role]); competitor = int(rival[patch, role])
            floor = float(c['required_margin'][patch, role]) if bool(c['weak_floor_mask'][patch, role]) else 0.
            observed = float(gap[patch, role])
            diagnostics.append(dict(bank_patch=patch, source_id=source_id,
                byte_offset=source_patch * 16 + role, role=role, gold=gold,
                rival=competitor, gap=observed, floor=floor,
                weak_floor_violation=bool(eroded[patch, role]), top1_flip=bool(flipped[patch, role]),
                deficit=max(0., floor - observed), anchor_SHA=self.anchor_SHA))
            if count <= 16:
                constraints.append(LinearConstraint(role, gold, competitor, self.hidden[patch],
                    floor, source_id, source_patch * 16 + role, self.anchor_SHA))
        return ReferenceObservation(decision, tuple(constraints), count,
                                    tuple(diagnostics), self.anchor_SHA)
