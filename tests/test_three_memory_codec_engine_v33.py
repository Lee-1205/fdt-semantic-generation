import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from test_three_memory_codec_v33 import config
from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE, OBJECTIVE, TRAINABILITY
from research.three_memory_codec_v33 import ThreeMemoryCodec
from owt_fuzzy_engine_v17_base import AEEngine, AMP_STOP_POLICY
from owt_resource_budget_v29 import RecoveryBudget, RESOURCE_POLICY
from three_memory_codec_engine_v33 import CodecEngine, old_optimizer_view
import text_generation_support_v1 as support

torch.set_num_threads(2)


def policy(path):
    return {'output': str(path), 'cutoff': time.time()+180,
        'resource_policy': RESOURCE_POLICY,
        'external_closed_assessment_calls': {'total': 1936, 'included_in_source_ledger': False},
        'limits': {'max_calls': 1000, 'max_output_bytes': 1024**3}}


def fixture(path):
    torch.manual_seed(391); random.seed(391); np.random.seed(391)
    features = [{'id': 'fixture'+str(i), 'prompt': f'number {i} not 25'} for i in range(23)]
    targets = [{'id': f['id'], 'target': f['prompt']} for f in features]
    orders = [list(range(8)), list(range(8, 16)), list(range(16, 23))]
    old = AEEngine(ExtractionTailFuzzyAE(config()), features, targets, orders, RecoveryBudget(policy(path)), 'CPU_V31_FIXTURE', 'cpu')
    named = dict(old.model.named_parameters())
    added = [n for n, p in named.items() if p.requires_grad and n.startswith(('core.language.blocks.1.', 'core.language.blocks.2.'))]
    old.names = [[n for n in old.names[0] if n not in added], old.names[1], added]
    assert list(map(len, old.names)) == [19, 44, 18]
    old.optimizer = torch.optim.AdamW([
        {'params': [named[n] for n in group], 'lr': lr, 'base_lr': lr}
        for group, lr in zip(old.names, (1e-5, 1e-4, 3e-6))], weight_decay=.01)
    old.step(); source = copy.deepcopy(old.export())
    source.update(schema='fdt_owt_fuzzy_AE_v31_fullstate', training_objective=OBJECTIVE,
        trainability=TRAINABILITY, constant_mask_probability=.5, resource_policy=RESOURCE_POLICY,
        amp_guard_policy=AMP_STOP_POLICY,
        external_closed_assessment_calls=policy(path)['external_closed_assessment_calls'], ancestry={})
    # The source is an explicitly tiny driver fixture, not an actual V31 trial.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(833); model = ThreeMemoryCodec(config())
    engine = CodecEngine(model, features, targets, orders, RecoveryBudget(policy(path)), 'CPU_CODEC_V33', 'cpu',
        source_parameter_groups=source['optimizer_parameter_names'], data_revision='CPU_INPUT_FIXTURE_ONLY')
    migration = engine.import_parent(source, 'CPU_V31_FIXTURE', 'CPU_SOURCE_FIXTURE')
    return engine, source, migration


def test_four_group_migration_old81_populated_exact_new10_empty_and_all_rng(tmp_path):
    engine, source, migration = fixture(tmp_path)
    state = engine.export()
    assert list(map(len, engine.names)) == [19, 44, 18, 10]
    assert len(engine.optimizer.state) == 81 and migration['new_state_count_at_import'] == 0
    assert support.state_sha(old_optimizer_view(state['optimizer'])) == support.state_sha(source['optimizer'])
    assert all(torch.equal(state['model'][n], value) for n, value in source['model'].items())
    for key in ('scaler', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'frozen_sha'):
        assert support.state_sha(state[key]) == support.state_sha(source[key])
    digest = support.state_sha(source)
    engine.worker_stage = 'codec_training'; engine.step()
    assert len(engine.optimizer.state) == 91 and support.state_sha(source) == digest
    for group in engine.optimizer.param_groups[:3]:
        assert {int(engine.optimizer.state[p]['step']) for p in group['params']} == {2}
    assert {int(engine.optimizer.state[p]['step']) for p in engine.optimizer.param_groups[3]['params']} == {1}
    assert engine.history[-1]['denoise_calls'] == engine.history[-1]['copy_bytes'] == 0
    assert engine.exposure['decoder_supervised_symbol_occurrences'] > 0
    engine.verify_frozen()


def test_full91_state_enabled_scaler_AMPskip_exact_next_last7_and_commit_epoch(tmp_path):
    engine, _, _ = fixture(tmp_path)
    engine.worker_stage = 'codec_training'; engine.evaluations = {'source_reference': 'CPU_SAVED_ONLY'}
    engine.step(); old_model = support.state_sha(engine.model.state_dict())
    hook = engine.model.core.anchors.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    try: assert engine.step()['AMP_skip']
    finally: hook.remove()
    assert support.state_sha(engine.model.state_dict()) == old_model and engine.scaler.is_enabled()
    saved = support.save_full(tmp_path, copy.deepcopy(engine.export()))
    snapshot = torch.load(tmp_path/'checkpoints'/saved['file'], weights_only=True)
    assert len(engine.next_indices()) == 7 and snapshot['cursor'] == 16
    expected = engine.step(); final = copy.deepcopy(engine.export())
    restored, _, _ = fixture(tmp_path); restored.restore(snapshot)
    assert restored.worker_stage == 'codec_training' and restored.evaluations == snapshot['evaluations']
    assert restored.model.hold_attempt == 2 and restored.budget.last_resource_check is None
    assert restored.step() == expected and support.state_sha(restored.export()) == support.state_sha(final)
    assert restored.cursor == 23 and restored.attempts == 3 and restored.updates == 2 and restored.skips == 1
    assert restored.budget.allocation_calls == {'encode': 3, 'denoise': 0, 'decoder': 6}
    with pytest.raises(StopIteration): restored.step()
    with torch.no_grad():
        episode = restored.encode_features(restored.features[:2])
    assert episode.commit.model_epoch == restored.memory_epoch()
    with pytest.raises(ValueError): restored.encode_features([{'id': 'x', 'prompt': 'hi', 'gold': 'leak'}])
    bad = copy.deepcopy(snapshot); bad['memory_epoch'] = 'stale'
    before_bad_restore = support.state_sha(restored.export())
    with pytest.raises(ValueError): restored.restore(bad)
    assert support.state_sha(restored.export()) == before_bad_restore
    assert not torch.cuda.is_initialized()
