"""Finite codec driver candidate: exact old81 Adam + fresh new10-head ownership.

No executable GPU worker/plan is supplied here. Four-group fullstate recovery is
separate from (and does not retroactively upgrade) the source-inventory review.
"""
import copy
import math
import random
from contextlib import nullcontext

import numpy as np
import torch

from research.text_diffusion_v1 import pack_prompts, pack_answers
from research.three_memory_codec_v33 import ThreeMemoryCodec
from research.three_memory_contract_v32 import PHASE, assert_codec_training_only
from research.owt_extraction_tail_v29 import OBJECTIVE as SOURCE_OBJECTIVE, TRAINABILITY as SOURCE_TRAINABILITY
from owt_fuzzy_engine_v17_base import AEEngine, AMP_STOP_POLICY, amp_stop_reason
from owt_resource_budget_v29 import RESOURCE_POLICY
import text_generation_support_v1 as support

SCHEMA = 'fdt_three_memory_language_codec_v33_fullstate'
OWNERSHIP = {'old_populated_groups': 3, 'new_groups': 1, 'new_heads_base_lr': 1e-4,
    'new_heads_weight_decay': .01, 'old_Adam_moments_preserved': True,
    'warmstart_changes_extraction_function': True, 'strict_old_learning_trajectory_resume': False}
CONTRACT = {'phase': PHASE, 'context': 'input_complete_per_token_long_recent_FP64',
    'persistent': 'learned_prior_only_no_certified_writer',
    'exact': 'fresh_input_UTF8_codepoint_index_top16_plus_NULL',
    'realizer': 'differentiable_Commit_only', 'copy_bytes': 0, 'denoise_calls': 0,
    'clean_CE_weight': .5, 'corrupted_prefix_CE_weight': .5, 'past_byte_mask_probability': .5,
    'loss_reduction': 'elementwise_none_then_sum'}


def old_optimizer_view(optimizer):
    groups = optimizer['param_groups'][:3]
    ids = {i for group in groups for i in group['params']}
    return {'state': {i: optimizer['state'][i] for i in ids if i in optimizer['state']}, 'param_groups': groups}


class CodecEngine(AEEngine):
    def __init__(self, model, features, targets, orders, budget, plan_sha, device='cuda', *, source_parameter_groups, data_revision):
        if type(model) is not ThreeMemoryCodec or not data_revision:
            raise ValueError('new codec model/data revision required')
        if len(features) != len(targets) or not orders or len(orders) > 8192:
            raise ValueError('finite codec data/order contract')
        for f, y in zip(features, targets):
            if set(f) != {'id', 'prompt'} or not isinstance(f['id'], str) or f['id'] != y['id'] or f['prompt'] != y['target']:
                raise ValueError('input-only features and SAME input target')
        if any(not batch or len(batch) > 8 or any(type(i) is not int or not 0 <= i < len(features) for i in batch) for batch in orders):
            raise ValueError('bounded valid batch indices')
        if len(source_parameter_groups) != 3:
            raise ValueError('three source populated groups')
        super().__init__(model, features, targets, orders, budget, plan_sha, device)
        old = [n for group in source_parameter_groups for n in group]
        named = dict(model.named_parameters())
        if len(old) != len(set(old)) or any(n not in named or not named[n].requires_grad for n in old):
            raise ValueError('old ownership lost')
        added = [n for n, p in named.items() if p.requires_grad and n not in set(old)]
        if not added or any(not n.startswith('memory_codec.') for n in added):
            raise ValueError('only new memory heads may join ownership')
        self.names = copy.deepcopy(source_parameter_groups)+[added]
        self.optimizer = torch.optim.AdamW([
            {'params': [named[n] for n in names], 'lr': lr, 'base_lr': lr}
            for names, lr in zip(self.names, (1e-5, 1e-4, 3e-6, OWNERSHIP['new_heads_base_lr']))
        ], weight_decay=.01)
        self.data_revision = data_revision
        self.migration = {}; self.ancestry = {}; self.worker_stage = 'new_allocation'
        assert_codec_training_only(model.named_parameters(), 0)

    def import_parent(self, state, source_plan, source_sha):
        if state['schema'] != 'fdt_owt_fuzzy_AE_v31_fullstate' or state['plan_sha256'] != source_plan or state['gradients']:
            raise ValueError('actual completed V31 source schema/identity')
        if state['training_objective'] != SOURCE_OBJECTIVE or state['trainability'] != SOURCE_TRAINABILITY:
            raise ValueError('source learning contract')
        if state.get('constant_mask_probability') != .5 or state.get('resource_policy') != RESOURCE_POLICY or state.get('amp_guard_policy') != AMP_STOP_POLICY:
            raise ValueError('source fixed objective/resource/AMP contract')
        if state['optimizer_parameter_names'] != self.names[:3] or self.attempts or self.cursor or self.history:
            raise ValueError('source ownership/new allocation')
        if state['external_closed_assessment_calls'] != self.budget.plan.get('external_closed_assessment_calls'):
            raise ValueError('external source ledger')
        self.model.import_legacy_weights(state['model'])
        old_frozen = support.state_sha({n: p for n, p in self.model.named_parameters() if not p.requires_grad})
        if old_frozen != state['frozen_sha']:
            raise ValueError('original frozen core/language prefix changed')
        opt = copy.deepcopy(state['optimizer'])
        ids = [i for group in opt['param_groups'] for i in group['params']]
        if ids != list(range(len(ids))) or set(opt['state']) != set(ids) or len(ids) != sum(map(len, self.names[:3])):
            raise ValueError('all source Adam states must be populated/contiguous')
        fresh = copy.deepcopy(self.optimizer.state_dict()['param_groups'][3])
        fresh['params'] = list(range(len(ids), len(ids)+len(self.names[3])))
        opt['param_groups'].append(fresh)
        self.optimizer.load_state_dict(opt); self.scaler.load_state_dict(state['scaler'])
        self.frozen_sha = old_frozen; self.budget.calls = dict(state['actual_calls'])
        torch.set_rng_state(state['torch_rng'])
        if str(self.device).startswith('cuda'): torch.cuda.set_rng_state_all(state['cuda_rng'])
        else: self.inactive_cuda_rng = copy.deepcopy(state['cuda_rng'])
        random.setstate(state['python_rng']); n = state['numpy_rng']
        np.random.set_state((n['name'], np.array(n['keys'], dtype=np.uint32), n['position'], n['gaussian'], n['cached']))
        exported = super().export()
        for name in ('scaler', 'frozen_sha', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'):
            if support.state_sha(exported[name]) != support.state_sha(state[name]):
                raise ValueError('source '+name+' changed')
        if support.state_sha(old_optimizer_view(exported['optimizer'])) != support.state_sha(state['optimizer']):
            raise ValueError('old populated Adam states changed')
        if any(i in exported['optimizer']['state'] for i in fresh['params']):
            raise ValueError('new head moments must start empty')
        self.migration = {'source_sha256': source_sha, 'source_plan_sha256': source_plan,
            'source_old_model_sha256': support.state_sha(state['model']),
            'source_old_optimizer_sha256': support.state_sha(state['optimizer']),
            'source_frozen_sha256': state['frozen_sha'], 'source_parameter_groups': copy.deepcopy(self.names[:3]),
            'new_parameter_names': list(self.names[3]), 'new_parameter_initial_sha256': support.state_sha({n: exported['model'][n] for n in self.names[3]}),
            'old_state_count': len(ids), 'new_state_count_at_import': 0,
            'model_function_changed_not_strict_trajectory_resume': True}
        self.ancestry = {k: copy.deepcopy(state[k]) for k in ('plan_sha256', 'attempts', 'updates', 'cursor', 'orders', 'history',
            'exposure', 'actual_calls', 'evaluations', 'ancestry', 'external_closed_assessment_calls')}
        self.ancestry['source_sha256'] = source_sha
        self.verify_frozen(); self.budget.invalidate_resources()
        return copy.deepcopy(self.migration)

    def memory_epoch(self):
        # Rebuild all snapshots after every attempted update. No cached Exact
        # key may survive even an AMP skip or a later model restoration.
        return f'{self.plan_sha}:attempt{self.attempts}:updates{self.updates}'

    def encode_features(self, features, charge=lambda kind: None, disabled_memories=()):
        if any(set(f) != {'id', 'prompt'} for f in features):
            raise ValueError('inference input features only')
        charge('encode')
        x = pack_prompts([f['prompt'] for f in features], 128, self.device)
        return self.model.encode_input(x, source_ids=tuple(f['id'] for f in features),
            source_versions=(self.data_revision,)*len(features), model_epoch=self.memory_epoch(), disabled_memories=disabled_memories)

    def step(self):
        self.budget.check(); indices = self.next_indices(); self.attempts += 1; self.cursor += len(indices)
        self.model.set_training_attempt(self.attempts)
        self.exposure['selected_rows'] += len(indices)
        byte_count = sum(len(self.targets[i]['target'].encode('utf-8')) for i in indices)
        self.exposure['selected_target_UTF8_bytes'] += byte_count
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        factor = .1+.9*(1+math.cos(math.pi*(self.attempts-1)/len(self.orders)))/2
        for group in self.optimizer.param_groups: group['lr'] = group['base_lr']*factor
        x = pack_prompts([self.features[i]['prompt'] for i in indices], 128, self.device)
        y = pack_answers([self.targets[i]['target'] for i in indices], 129, self.device)
        def charge(kind):
            self.budget.charge(kind)
            if kind == 'encode':
                self.exposure['processed_rows'] += len(indices)
                self.exposure['target_UTF8_bytes'] += byte_count
                self.exposure['source_sentence_boundaries'] += len(indices)
        amp = torch.autocast('cuda', dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
        try:
            with amp:
                loss, episode = self.model.reconstruction_loss(x, y, source_ids=tuple(self.features[i]['id'] for i in indices),
                    source_versions=(self.data_revision,)*len(indices), model_epoch=self.memory_epoch(), charge=charge)
            if not torch.isfinite(loss) or float(loss.detach()) > 1000: raise FloatingPointError('codec nonfinite/loss explosion')
            self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None: raise RuntimeError('missing codec gradient '+name)
                if not parameter.requires_grad and parameter.grad is not None: raise RuntimeError('frozen gradient '+name)
            norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.)
            before = self.scaler.get_scale(); self.scaler.step(self.optimizer); self.scaler.update()
            skip = self.scaler.get_scale() < before; self.skips += int(skip); self.updates += int(not skip)
            rec = {'attempt': self.attempts, 'updates': self.updates, 'rows': list(indices), 'loss': float(loss.detach()),
                'AMP_skip': skip, 'grad_norm': float(norm) if torch.isfinite(norm) else None, 'scale': self.scaler.get_scale(),
                'mean_fuzzy_degree': float(episode.diagnostics['memberships'].detach().mean()),
                'outcome': 'AMP_SKIP' if skip else 'UPDATED', 'mask_probability': .5, **self.model.last_objective_stats}
            self.history.append(rec); self.optimizer.zero_grad(set_to_none=True)
            for key in ('masked_prefix_bytes', 'eligible_prefix_bytes', 'decoder_supervised_symbol_occurrences'):
                self.exposure[key] = self.exposure.get(key, 0)+rec[key]
            reason = amp_stop_reason(self.history, self.ancestry.get('history', []), self.skips, skip, bool(torch.isfinite(norm)))
            if reason: raise FloatingPointError('codec persistent AMP/gradient stop: '+reason)
            if len(self.history) > 10 and rec['loss'] > 20*sorted(r['loss'] for r in self.history[-11:-1] if 'loss' in r)[5]:
                raise FloatingPointError('20x codec loss explosion')
            return rec
        except BaseException as exc:
            if len(self.history) < self.attempts:
                self.history.append({'attempt': self.attempts, 'updates': self.updates, 'rows': list(indices),
                    'outcome': 'FAILED_ATTEMPT', 'error': repr(exc)})
            raise

    def export(self):
        state = super().export()
        state.update(schema=SCHEMA, codec_contract=copy.deepcopy(CONTRACT), ownership_policy=copy.deepcopy(OWNERSHIP),
            resource_policy=copy.deepcopy(RESOURCE_POLICY), amp_guard_policy=copy.deepcopy(AMP_STOP_POLICY),
            data_revision=self.data_revision, worker_stage=self.worker_stage, memory_epoch=self.memory_epoch(),
            migration=copy.deepcopy(self.migration), ancestry=copy.deepcopy(self.ancestry),
            external_closed_assessment_calls=copy.deepcopy(self.budget.plan.get('external_closed_assessment_calls')),
            allocation_calls=dict(self.budget.allocation_calls))
        return state

    def restore(self, state):
        if state['schema'] != SCHEMA or state['codec_contract'] != CONTRACT or state['ownership_policy'] != OWNERSHIP:
            raise ValueError('codec fullstate schema/contract')
        if state['resource_policy'] != RESOURCE_POLICY or state['amp_guard_policy'] != AMP_STOP_POLICY or state['data_revision'] != self.data_revision:
            raise ValueError('codec resource/data contract')
        if state['migration']['source_parameter_groups'] != self.names[:3] or state['migration']['new_parameter_names'] != self.names[3]:
            raise ValueError('codec full ownership migration')
        if state['external_closed_assessment_calls'] != self.budget.plan.get('external_closed_assessment_calls'):
            raise ValueError('codec external ledger')
        if state['frozen_sha'] != state['migration']['source_frozen_sha256']:
            raise ValueError('original frozen scope changed')
        if state['memory_epoch'] != f"{self.plan_sha}:attempt{state['attempts']}:updates{state['updates']}":
            raise ValueError('model/index embedding epoch mismatch')
        adapted = dict(state); adapted['schema'] = 'fdt_owt_fuzzy_AE_v12_fullstate'
        super().restore(adapted)
        self.migration = copy.deepcopy(state['migration']); self.ancestry = copy.deepcopy(state['ancestry'])
        self.worker_stage = state['worker_stage']; self.budget.allocation_calls = dict(state['allocation_calls'])
        if state['memory_epoch'] != self.memory_epoch(): raise ValueError('model/index embedding epoch mismatch')
        self.model.set_training_attempt(max(1, self.attempts)); self.budget.invalidate_resources()
        assert_codec_training_only(self.model.named_parameters(), 0)
