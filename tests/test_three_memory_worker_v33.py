import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from test_three_memory_codec_engine_v33 import fixture
import run_three_memory_codec_20261029_v33 as worker
import supervise_three_memory_codec_20261029_v33 as owner

PLAN = ROOT/'research/artifacts/three_memory_codec_20261029_v33_worker_cpu/plan_unsealed_r3.json'


def test_actual_new_plan_schema_exact_finite_order_and_independent_owner_contract():
    p = json.loads(PLAN.read_text(encoding='utf-8'))
    _, f, y, orders = worker.validate(PLAN, Path(p['output']), worker.__file__)
    owner.validate(PLAN, Path(p['output']))
    assert len(f['train']) == len(y['train']) == 64 and len(orders) == 2048
    for start in range(0, 2048, 8):
        assert sorted(i for batch in orders[start:start+8] for i in batch) == list(range(64))
    assert orders != worker.finite_orders(64, range(391, 647))
    assert sum(v for k, v in worker.COST.items() if k != 'total_max') == worker.COST['total_max'] == 11028
    assert p['authorized'] is False and p['deadline'] is None and not Path(p['output']).exists()
    assert p['external_closed_assessment_calls']['calls'] == {'encode': 40, 'denoise': 0, 'decoder': 1896}
    assert p['external_closed_assessment_calls']['total'] == 1936
    assert worker.CONTRACT == owner.CONTRACT and worker.OWNERSHIP == owner.OWNERSHIP and worker.LIMITS == owner.LIMITS
    assert worker.START_STORAGE == owner.START_STORAGE
    assert p['limits']['start_free_bytes'] == (20+6+1+8)*1024**3
    assert p['limits']['start_free_bytes']-p['limits']['max_output_bytes']-1024**3 == (20+8)*1024**3


def test_actual_teacher_trace_reconstructs_full_exact_scores_ND_and_raw_api_is_input_only(tmp_path):
    engine, _, _ = fixture(tmp_path)
    metrics, tensors = worker.teacher_assessment(engine, engine.features[:2], engine.targets[:2], engine.budget, 'cpu',
        modes=('normal',), trace_indices=(0, 1))
    assert metrics['normal']['sentence_EOS_total'] == 2 and len(tensors['memory_trace']) == 2
    for trace in tensors['memory_trace']:
        scores = F.normalize(trace['exact_query'], dim=-1) @ F.normalize(trace['exact_keys'], dim=-1).T
        scores = scores.masked_fill(~trace['exact_valid'][None], -torch.inf)
        ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :16]
        assert torch.equal(ids, trace['candidate_ids'])
        chosen = scores.gather(-1, ids)
        weights = torch.cat((chosen, trace['exact_null_logit'].expand(len(chosen), 1)), -1).softmax(-1)
        assert torch.allclose(weights[:, :-1], trace['candidate_weights'], atol=1e-7)
        assert torch.allclose(weights[:, -1], trace['null_weight'], atol=1e-7)
        n = len(trace['source'].encode('utf-8'))
        mu = trace['token_mu'][:n].double(); v = trace['token_values'][:n].double()
        recent = mu * trace['decay'].double()[None]**torch.arange(n-1, -1, -1)[:, None]
        assert torch.allclose(mu.T @ v, trace['long_N'], atol=1e-12)
        assert torch.allclose(mu.sum(0)[:, None], trace['long_D'], atol=1e-12)
        assert torch.allclose(recent.T @ v, trace['recent_N'], atol=1e-12)
        assert trace['denoise_calls'] == trace['copy_bytes'] == 0
    rows = worker.reconstruct_inputs(engine, engine.features[:2], engine.budget, 'cpu')
    assert len(rows) == 2 and [r['source'] for r in rows] == [r['prompt'] for r in engine.features[:2]]
    assert all(r['copy_bytes'] == r['denoise_calls'] == 0 for r in rows)
    with pytest.raises(ValueError):
        worker.reconstruct_inputs(engine, [{'id': 'bad', 'prompt': 'input', 'gold': 'leak'}], engine.budget, 'cpu')
    assert not torch.cuda.is_initialized()


class Child:
    pid = 999987
    returncode = 0
    def poll(self): return self.returncode
    def wait(self, timeout=None): return self.returncode


@pytest.mark.parametrize('crash,confirmed,stop_publish_error', [
    (False, True, False), (True, True, False), (False, False, False), (True, True, True)])
def test_actual_owner_postPopen_natural_exception_unconfirmed_and_STOP_write_failure(tmp_path, crash, confirmed, stop_publish_error):
    p = json.loads(PLAN.read_text(encoding='utf-8'))
    p.update(authorized=True, output=str(tmp_path), allocation_start=time.time())
    p.update(deadline=p['allocation_start']+900, cutoff=p['allocation_start']+780)
    path = tmp_path/'plan.json'; path.write_text(json.dumps(p), encoding='utf-8')
    owner.validate(path, tmp_path); worker.validate(path, tmp_path, worker.__file__)
    original_json = owner.io.atomic_json; original_text = owner.io.atomic_text
    def atomic_json(destination, value):
        if crash and Path(destination).name == 'state.json' and value.get('launches') == 1 and not value.get('stop_reason'):
            raise RuntimeError('CPU_POST_POPEN_EXCEPTION')
        return original_json(destination, value)
    def atomic_text(destination, value):
        if stop_publish_error and Path(destination).name == 'STOP': raise OSError('CPU_STOP_WRITE_ERROR')
        return original_text(destination, value)
    def launch(*args, **kwargs):
        (tmp_path/'result.json').write_text(json.dumps({'status': 'THREE_MEMORY_CODEC_V33_COMPLETE'}), encoding='utf-8')
        return Child()
    with patch.object(owner, 'preflight'), patch.object(owner, 'resources', return_value={}), \
         patch.object(owner, 'duplicates', return_value=[]), patch.object(owner.native, 'process_inventory', return_value=[]), \
         patch.object(owner.subprocess, 'Popen', side_effect=launch), patch.object(owner.io, 'atomic_json', side_effect=atomic_json), \
         patch.object(owner.io, 'atomic_text', side_effect=atomic_text), \
         patch.object(owner.native, 'verify_exit', return_value={'all_owned_processes_gone': confirmed}) as cleanup:
        code = owner.supervise(path, tmp_path, Path(worker.__file__))
    end = json.loads((tmp_path/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert cleanup.call_count == 1
    expected = 'UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE'
    assert end['status'] == expected and code == (0 if expected == 'COMPLETE' else 2)
    if stop_publish_error: assert 'CPU_STOP_WRITE_ERROR' in end['STOP_publish_error']


def test_duplicate_check_only_first_invoked_script_never_supervisor_worker_argument():
    command = f'"{sys.executable}" "{owner.__file__}" --worker "{worker.__file__}"'
    real = f'"{sys.executable}" "{worker.__file__}" --plan fixture'
    class Result:
        stdout = json.dumps([{'ProcessId': 1, 'CommandLine': command}, {'ProcessId': 2, 'CommandLine': real}])
    with patch.object(owner.native, 'run_native', return_value=Result()):
        found = owner.duplicates(Path(worker.__file__))
    assert [r['ProcessId'] for r in found] == [2]
