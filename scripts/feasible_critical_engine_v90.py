"""Finite live-encoder engine with original teacher/reference, warm head.

No runnable worker, plan or automatic continuation. Existing verified TRAIN
teacher targets are reused explicitly, NOT new encoder/teacher exposure. All
five gradient micros still execute the live frozen encoder. Original floors
are never recalibrated from the warm model. Only the final Linear may change.
"""
import copy
import hashlib
import random
import numpy as np
import torch
from compact_critical_engine_v87 import CompactCriticalEngine, ROOT
from research.critical_reference_bank_v85 import CriticalReferenceBank
from research import critical_bank_storage_v87 as storage
from research.immutable_reference_anchor_v89 import verify, digest
from research.diagnostic_reference_v90 import DiagnosticReference
from research.feasible_adam_transaction_v90 import feasible_adam_proposal
from research.competing_margin_objective_v78 import decoder_losses, GROUP_ROW_WEIGHTS
from research.readout_precision_policy_v81 import POLICY
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
import text_generation_support_v1 as support

SCHEMA = 'fdt_original_anchor_feasible_actual_Adam_engine_v90'
OBJECTIVE = 'V78_loss+original_TRAIN_anchor+one_actual_Adam_delta_bounded_direction_correction'


class FeasibleCriticalEngine(CompactCriticalEngine):
    def __init__(self, *args, static_reference, representation, **kwargs):
        super().__init__(*args, **kwargs)
        if set(static_reference) != {'anchor', 'bank_ref', 'teacher_refs', 'teacher_sha'}:
            raise ValueError('explicit immutable original TRAIN artifacts')
        s = copy.deepcopy(static_reference)
        verify(s['anchor'], representation, s['bank_ref'])
        actual = dict(representation, embedding_SHA=self.embedding_sha,
            prefix_SHA=self.frozen_prefix_SHA, core_config_SHA=digest(self.model.encoder.c.__dict__),
            input_capacity=self.model.encoder.input_capacity, precision_policy=POLICY)
        if actual != representation:
            raise ValueError('actual frozen tensors/core/capacity differ from original anchor')
        bank_state = storage.load(s['bank_ref'], ROOT, support.sha)
        if support.state_sha(bank_state) != s['bank_ref']['bank_runtime_SHA']:
            raise ValueError('original calibrated bank digest')
        config = bank_state['config']
        if (config['embedding_SHA'] != self.embedding_sha or config['prefix_SHA'] != self.frozen_prefix_SHA
                or config['parent_SHA'] != s['anchor']['origin_source_SHA']
                or config['precision_policy'] != POLICY
                or config['sources'] != [dict(id=r['id'], SHA=hashlib.sha256(r['prompt'].encode()).hexdigest()) for r in self.rows]
                or config['broad_indices'] != self.budget.anchor_indices
                or len(s['teacher_refs']) != len(self.rows) or len(s['teacher_sha']) != 64):
            raise ValueError('original TRAIN teacher/reference/representation allowlist')
        self.bank = CriticalReferenceBank(**{k: v for k, v in config.items() if k != 'precision_policy'})
        self.bank.restore(bank_state)
        self.static_reference = s
        self.static_reference_SHA = support.state_sha(s)
        self.representation = copy.deepcopy(representation)
        self.warm_readout_SHA = self.teacher_sha
        self.teacher_sha = s['teacher_sha']  # Distinct STATIC teacher, never relabel warm logits.
        self.cache_refs = copy.deepcopy(s['teacher_refs'])
        self.cache_cursor = len(self.rows)
        self.bank_ref = copy.deepcopy(s['bank_ref'])
        self.bank_ready = True
        self._device_reference = None
        self.warm_reference_verified = False
        self.phase = 'original_anchor_warm_verification'
        self.objective_report.update(objective=OBJECTIVE, static_reference_SHA=self.static_reference_SHA,
            warm_readout_SHA=self.warm_readout_SHA, teacher_origin_SHA=s['anchor']['origin_source_SHA'],
            new_teacher_rows=0, reused_static_teacher_rows=len(self.rows), reference_calls_per_group_cap=2,
            actual_delta_correction=dict(passes=2, active_constraints=16, relative_radius=1., padding=1e-6),
            accepted_moments='full actual Adam proposal retained; corrected weights define projected Adam',
            reference_recalibration=False)
        for i in range(len(self.rows)):
            self.packet(i)

    def cache_next(self):
        raise RuntimeError('static original teacher reused; no new teacher phase/replay')

    def initialize_reference(self):
        raise RuntimeError('original reference must not be recalibrated from warm head')

    def commit_reference_files(self):
        raise RuntimeError('original immutable reference files retained; no rewrite')

    @torch.no_grad()
    def reference_observation(self):
        if self._device_reference is None:
            features, _ = self.bank.components()
            self._device_reference = DiagnosticReference(features, self.bank.calibration,
                self.device, self.static_reference['anchor']['anchor_SHA'])
        self.budget.charge('reference_linear')
        self.reference_patch_work += len(self._device_reference.hidden)
        if self.reference_patch_work > 8192 * (2 * len(self.orders) + 2):
            raise ValueError('fixed two-check-per-group full-reference work cap')
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=False):
            z = self.model.readout.readout.net[2](self._device_reference.hidden).reshape(-1, 16, 258)
        return self._device_reference.inspect(z)

    @torch.no_grad()
    def validate_warm_reference(self):
        if self.inflight is not None or self.attempts or self.warm_reference_verified:
            raise ValueError('one warm check against ORIGINAL bank, not new calibration')
        self.inflight = dict(stage='original_anchor_warm_check_requested')
        seen = self.reference_observation()
        self.inflight['observation'] = dict(accept=seen.decision.accept,
            measurements=seen.decision.measurements, blocker_count=seen.blocker_count,
            diagnostics=seen.diagnostics)
        if not seen.decision.accept:
            raise ValueError('warm source violates original reference; no floor reset')
        self.warm_reference_verified = True
        self.inflight = None
        self.phase = 'head_training'
        return copy.deepcopy(seen.decision.measurements)

    def reference_decision(self):
        return self.reference_observation().decision

    def step(self, force_overflow=False):
        if self.inflight is not None:
            raise RuntimeError('interrupted consumed group, automatic replay forbidden')
        if not self.warm_reference_verified or self.attempts >= len(self.orders):
            raise StopIteration('original warm verification then finite groups only')
        self.budget.check()
        ids = self.orders[self.attempts]
        packets = [self.packet(i) for i in ids]
        self.attempts += 1
        self.cursor += 1
        for t in packets:
            self.exposure['head_rows'] += 1
            self.exposure['head_UTF8_bytes'] += t['source_bytes']
            self.exposure['head_symbols'] += t['source_bytes'] + 1
            self.exposure['head_critical_positions'] += int(t['critical_mask'].sum())
        self.inflight = dict(stage='head_group_requested', consumed_attempt=self.attempts,
                             rows=ids, completed_micros=[], encoded_micros=[])
        self.optimizer.zero_grad(set_to_none=True)
        micros = []
        try:
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for j, (index, t, weight) in enumerate(zip(ids, packets, GROUP_ROW_WEIGHTS)):
                self.inflight.update(stage='head_micro_source_requested', micro_index=j)
                c = self.extract_live(self.rows[index])
                self.exposure['head_encoder_rows'] += 1
                self.exposure['head_encoder_UTF8_bytes'] += t['source_bytes']
                self.inflight['encoded_micros'].append(j)
                self.budget.charge('readout')
                self.inflight['stage'] = 'head_micro_backward'
                z = self.model.verbalize(c)
                if z.dtype != torch.float32:
                    raise ValueError('actual full-FP32 backward')
                loss, parts = decoder_losses(z, t['labels'].to(self.device), self.symbol_weights,
                    t['critical_mask'].to(self.device), t['teacher_gold_probability'].to(self.device),
                    t['teacher_correct'].to(self.device), t['teacher_margin'].to(self.device))
                if not torch.isfinite(loss) or float(loss.detach()) > 100:
                    raise FloatingPointError('bounded numeric micro loss')
                self.scaler.scale(weight * loss).backward()
                record = dict(row=index, bytes=t['source_bytes'], weight=weight, loss=float(loss.detach()))
                record.update({k: float(v.detach()) if torch.is_tensor(v) else v for k, v in parts.items()})
                micros.append(record)
                self.inflight['completed_micros'] = copy.deepcopy(micros)
                del z, c, loss
            named = dict(self.model.named_parameters())
            if force_overflow:
                named[self.names[0]].grad.fill_(float('inf'))
            self.inflight['stage'] = 'head_unscale_clip_once'
            self.scaler.unscale_(self.optimizer)
            if (any(named[n].grad is None for n in self.names)
                    or any(v.grad is not None for n, v in named.items() if n not in self.names)):
                raise ValueError('only lastLinear2 owns gradients')
            norm = torch.nn.utils.clip_grad_norm_([named[n] for n in self.names], 1.)
            self.inflight['stage'] = 'head_actual_delta_reference'
            def transaction_event(value):
                self.inflight['transaction'] = copy.deepcopy(value)
            receipt = feasible_adam_proposal(self.optimizer, {n: named[n] for n in self.names},
                self.scaler, self.reference_observation,
                anchor_SHA=self.static_reference['anchor']['anchor_SHA'],
                guard=self.budget.check, event=transaction_event)
            skip = receipt['AMP_skip']
            self.skips += int(skip)
            self.updates += int(receipt['accepted'])
            self.rejections += int(not skip and not receipt['accepted'])
            self.history.append(dict(attempt=self.attempts, updates=self.updates, rows=ids,
                micros=micros, AMP_skip=skip, rejected=not skip and not receipt['accepted'],
                update_receipt=receipt, scale=self.scaler.get_scale(),
                grad_norm=float(norm) if torch.isfinite(norm) else None,
                loss=sum(v['weight'] * v['loss'] for v in micros), clip_and_Adam_calls=1,
                GPU_peak_bytes=torch.cuda.max_memory_allocated() if self.device == 'cuda' else 0,
                Exact_entries=0))
            self.optimizer.zero_grad(set_to_none=True)
            self.inflight = None
            if (self.skips >= 8 or sum(h['AMP_skip'] for h in self.history[-16:]) >= 3
                    or (not skip and not torch.isfinite(norm))
                    or (self.device == 'cuda' and torch.cuda.max_memory_allocated() > 12 * 1024**3)):
                raise FloatingPointError('AMP/numeric/GPU stop')
            return self.history[-1]
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def export(self):
        s = super().export()
        s.update(schema=SCHEMA, objective=OBJECTIVE, static_reference_SHA=self.static_reference_SHA,
            representation=copy.deepcopy(self.representation), warm_readout_SHA=self.warm_readout_SHA,
            warm_reference_verified=self.warm_reference_verified)
        return s

    def restore(self, s):
        pending = s['inflight'] is not None and s['inflight']['stage'].startswith('head_')
        if (s['schema'] != SCHEMA or s['objective'] != OBJECTIVE
                or s['objective_report'] != self.objective_report or s['precision_policy'] != POLICY
                or s['static_reference_SHA'] != self.static_reference_SHA
                or s['representation'] != self.representation or s['warm_readout_SHA'] != self.warm_readout_SHA
                or s['plan_sha256'] != self.plan_sha or s['exact_policy'] != EXACT_POLICY
                or s['parent_reference'] != self.budget.parent_reference or s['orders'] != self.orders
                or s['optimizer_names'] != self.optimizer_names or s['storage_delta_names'] != self.storage_delta_names
                or not torch.equal(s['symbol_weights'], self.symbol_weights.cpu())
                or s['embedding_sha'] != self.embedding_sha or s['teacher_sha'] != self.teacher_sha
                or s['frozen_prefix_SHA'] != self.frozen_prefix_SHA or s['frozen_head_prefix_SHA'] != self.frozen_prefix_SHA
                or s['frozen_except_last2_SHA'] != self.frozen_except_last2_SHA
                or s['input_capacity'] != self.model.encoder.input_capacity or s['gradients'] or s['new_dev_refs']
                or s['bank_ref'] != self.bank_ref or s['bank_state'] is not None
                or s['cache_refs'] != self.cache_refs or s['cache_cursor'] != len(self.rows)):
            raise ValueError('complete warm/static-original-anchor engine identity')
        if (s['cursor'] != s['attempts'] or not 0 <= s['attempts'] <= len(self.orders)
                or len(s['history']) != s['attempts'] - int(pending)
                or [h['rows'] for h in s['history']] != self.orders[:len(s['history'])]
                or s['updates'] + s['skips'] + s['rejections'] != len(s['history'])
                or not s['bank_ready'] or (s['attempts'] and not s['warm_reference_verified'])):
            raise ValueError('finite consumed source/history ledger')
        if (set(s['calls']) != set(self.budget.calls)
                or any(type(v) is not int or v < 0 for v in s['calls'].values())
                or s['calls']['denoise'] != 0
                or s['reference_patch_work'] != self.bank.patch_count * s['calls']['reference_linear']
                or s['reference_patch_work'] > 8192 * (2 * len(self.orders) + 2)
                or any(h['update_receipt']['anchor_SHA'] != self.static_reference['anchor']['anchor_SHA']
                       or len(h['update_receipt']['reference_checks']) > 2 for h in s['history'])):
            raise ValueError('consumed calls/work and immutable-anchor receipts')
        if (s['exposure']['teacher_rows'] or s['exposure']['encoder_rows']
                or s['exposure']['encoder_UTF8_bytes']
                or s['exposure']['head_rows'] != 5 * s['attempts']
                or s['exposure']['head_encoder_rows'] != sum(len(h['micros']) for h in s['history'])
                    + (len(s['inflight']['encoded_micros']) if pending else 0)):
            raise ValueError('static teacher is not fresh exposure; consumed live micros ledger')
        self.model.load_state_dict(s['model'], strict=True)
        self.optimizer.load_state_dict(s['optimizer'])
        self.scaler.load_state_dict(s['scaler'])
        for key in ('attempts', 'updates', 'cursor', 'skips', 'phase', 'history', 'exposure',
                    'evaluations', 'inflight', 'teacher_GPU_peak_bytes', 'rejections',
                    'reference_patch_work', 'reference_final', 'warm_reference_verified'):
            setattr(self, key, copy.deepcopy(s[key]))
        self._device_reference = None
        self.budget.calls = copy.deepcopy(s['calls'])
        self.budget.block_work = s['long_block_work']
        self.inactive_cuda_rng = copy.deepcopy(s['cuda_rng'])
        self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.readout.readout.state_dict()) != self.warm_readout_SHA:
            raise ValueError('initial warm head changed; static teacher is a separate identity')
        for i in range(len(self.cache_refs)):
            self.packet(i)
        torch.set_rng_state(s['torch_rng'])
        if self.device == 'cuda':
            torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng'])
        n = s['numpy_rng']
        np.random.set_state((n['name'], np.array(n['keys'], dtype=np.uint32), n['position'], n['gaussian'], n['cached']))
