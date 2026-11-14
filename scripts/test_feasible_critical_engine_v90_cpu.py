"""Tiny synthetic E + actual512-wide head CPU engine/fullstate contract.

One bounded fixture run, immutable preexisting synthetic teacher/bank reused.
This is NOT an update or forward of the learned 123M FDT, nor CUDA restart.
"""
import copy
import io as memoryio
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'src')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64, FuzzyRoundTripV64
from research.fp32_ordered_readout_v86 import FP32OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from research.immutable_reference_anchor_v89 import build, digest
from research import critical_bank_storage_v87 as bank_storage
from research import linked_lastlinear_runtime_v79 as linked
from research.feasible_adam_transaction_v90 import ReferenceObservation
from research.actual_update_retention_v80 import TrainDecision
from feasible_critical_engine_v90 import FeasibleCriticalEngine
import text_generation_support_v1 as support
import learning_io_v2 as io


def main():
    began = time.monotonic()
    torch.set_num_threads(2); torch.manual_seed(90113); random.seed(90113); np.random.seed(90113)
    directory = ROOT / 'research/artifacts/feasible_critical_engine_20261113_v90_cpu_v1'
    directory.mkdir(exist_ok=False)
    old = ROOT / 'research/artifacts/compact_critical_engine_20261113_v87_cpu_v1/a'
    parent = ROOT / 'research/artifacts/critical_bank_engine_20261113_v86_cpu_v1/CPU_parent.pt'
    source = torch.load(parent, map_location='cpu', weights_only=True)['model']
    initial = copy.deepcopy(source)
    initial['readout.readout.net.0.bias'].add_(.125)
    # A DIFFERENT warm head, common shift of all 258 class biases. Frozen
    # representation identical; teacher logits/margins are NOT regenerated.
    initial['readout.readout.net.2.bias'].add_(1e-5)
    rows = [dict(id='CPU_TRAIN:u', prompt='Cruz stored 27 pins.'),
            dict(id='CPU_TRAIN:s', prompt='Cyrus logged +2.50 kg at 03:17.'),
            dict(id='CPU_TRAIN:o', prompt='A plain sentence without a numeral.')]
    labels = {r['id']: complete_expression_spans(r['prompt'], [dict(start=m.start(), end=m.end(), text=m.group(), kind='name')
               for m in re.finditer(r'Cruz|Cyrus', r['prompt'])]) for r in rows}
    refs = []
    for i, row in enumerate(rows):
        path = old / 'teacher_targets' / f'{i:04d}.pt'
        t = torch.load(path, map_location='cpu', weights_only=True)
        refs.append(dict(targets=str(path), targets_SHA=support.sha(path), cache_key=t['cache_key'], source_id=row['id']))
    teacher_sha = t['teacher_sha']
    fp, tp = old / 'TRAIN_reference/bank.features.pt', old / 'TRAIN_reference/bank.targets.pt'
    feature = torch.load(fp, map_location='cpu', weights_only=True)
    target = torch.load(tp, map_location='cpu', weights_only=True)
    bank = bank_storage.reconstruct(feature, target)
    reference = dict(features=str(fp), features_SHA=support.sha(fp), targets=str(tp), targets_SHA=support.sha(tp), bank_runtime_SHA=support.state_sha(bank))
    config = Config.smoke()
    representation = dict(embedding_SHA=feature['metadata']['embedding_SHA'], prefix_SHA=feature['metadata']['prefix_SHA'],
        core_config_SHA=digest(config.__dict__), input_capacity=256, precision_policy=feature['metadata']['precision_policy'],
        operator_SHA=digest(dict(CPU_fixture_only=True, original_parent_SHA=support.sha(parent))))
    anchor = build('f' * 64, representation, reference)
    static = dict(anchor=anchor, bank_ref=reference, teacher_refs=refs, teacher_sha=teacher_sha)
    orders = [[0, 1, 2, 0, 1]] * 3
    profile = dict(weights=[1.] * 257 + [0.], source_SHA='CPU_synthetic_symbol_profile')
    parentref = dict(path=str(parent.resolve()), SHA=support.sha(parent), model_key='model', model_SHA=support.state_sha(source))
    class Budget:
        data_sha='CPU_v86'; parent_reference={'warm_source': 'CPU_v90', 'static_teacher': 'CPU_v86'}
        parent_SHA='e'*64; anchor_indices=[0, 1, 2]; critical_targets=labels; frozen_symbol_profile=profile
        def __init__(self, label):
            self.output=directory/label; self.output.mkdir()
            self.calls=dict(encode=0, readout=0, reference_linear=0, denoise=0)
            self.block_work=0; self.fail_at=None
        def check(self, write_bytes=0, emergency=False):
            if time.monotonic()-began > 90: raise TimeoutError('fixed CPU90sec fixture')
            if shutil.disk_usage(ROOT).free < 100*1024**3+write_bytes: raise OSError('100GiB plus pending write')
            if sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())+write_bytes > 8*1024**2: raise OSError('fixed CPU8MiB new output')
        def charge(self, kind):
            self.check()
            if sum(self.calls.values()) >= 64: raise RuntimeError('fixed CPU64calls per fixture engine')
            if self.fail_at is not None and sum(self.calls.values()) >= self.fail_at:
                self.fail_at=None; raise InterruptedError('CPU cooperative STOP')
            self.calls[kind] += 1
    def create(label):
        with torch.random.fork_rng(devices=[]):
            encoder=FuzzyCodecV64(config, input_capacity=256)
            encoder.enable_position_adapter(); encoder.enable_long_capacity(); encoder.enable_local_banks()
            model=FuzzyRoundTripV64(encoder, FP32OrderedFuzzyReadout(config.z_dim, config.value_dim, hidden=512))
            model.load_state_dict(initial, strict=True)
        return FeasibleCriticalEngine(model, rows, orders, Budget(label), 'CPU_plan_v90', 'cpu', static_reference=static, representation=representation)
    a=create('main')
    assert a.teacher_sha != a.warm_readout_SHA and not a.optimizer.state and a.scaler.is_enabled()
    a.validate_warm_reference(); complete=copy.deepcopy(a.export())
    assert a.exposure['teacher_rows'] == 0 and a.budget.calls['encode'] == 0
    for method in (a.cache_next, a.initialize_reference, a.commit_reference_files):
        try: method(); raise AssertionError('reference/teacher rewrite allowed')
        except RuntimeError: pass
    a.step(); middle=copy.deepcopy(a.export())
    assert a.updates == 1 and len(a.optimizer.state) == 2
    assert sum(dict(a.model.named_parameters())[n].numel() for n in a.names) == 2117664
    envelope=linked.pack(middle, source, parentref, support.state_sha)
    stream=memoryio.BytesIO(); torch.save(envelope, stream); stream.seek(0)
    saved=linked.unpack(torch.load(stream, map_location='cpu', weights_only=True), ROOT, support.state_sha)
    assert support.state_sha(saved) == support.state_sha(middle)
    r1=(torch.rand(3), random.random(), np.random.rand())
    a.step(True); a.step(); expected=copy.deepcopy(a.export())
    recovered=create('restored'); recovered.restore(saved)
    r2=(torch.rand(3), random.random(), np.random.rand())
    assert torch.equal(r1[0], r2[0]) and r1[1:] == r2[1:]
    recovered.step(True); recovered.step()
    assert support.state_sha(recovered.export()) == support.state_sha(expected)
    assert a.attempts == a.cursor == 3 and a.updates == 2 and a.skips == 1
    assert a.exposure['head_rows'] == a.exposure['head_encoder_rows'] == 15
    assert a.budget.calls['encode'] == a.budget.calls['readout'] == 15
    assert not a.exposure['teacher_rows'] and all(int(v['step']) == 2 for v in a.optimizer.state.values())
    a.verify_frozen()
    rejected=create('reject'); rejected.restore(middle)
    before=copy.deepcopy(rejected.export()); original=rejected.reference_observation
    def force_overflow():
        original()  # The ordinary full reference projection is STILL charged.
        return ReferenceObservation(TrainDecision(False, dict(CPU_forced_blockers=17)), (), 17,
            tuple(dict(source_id='CPU_forced_constraint_overflow') for _ in range(16)), anchor['anchor_SHA'])
    rejected.reference_observation=force_overflow
    rejected.step(); after=copy.deepcopy(rejected.export())
    assert support.state_sha(before['model']) == support.state_sha(after['model'])
    assert support.state_sha(before['optimizer']) == support.state_sha(after['optimizer'])
    assert after['attempts'] == 2 and after['rejections'] == 1
    assert after['scaler']['_growth_tracker'] == before['scaler']['_growth_tracker'] + 1
    reject_restore=create('restore_rejection'); reject_restore.restore(after)
    assert support.state_sha(reject_restore.export()) == support.state_sha(after)
    stop_proofs={}
    for label, offset in (('before_encoder', 0), ('after_encoder_before_readout', 1), ('after_one_micro', 2), ('actual_proposal', 10)):
        e=create('STOP_'+label); e.restore(complete); e.budget.fail_at=sum(e.budget.calls.values())+offset
        try: e.step(); raise AssertionError('expected cooperative STOP')
        except InterruptedError: pass
        stopped=copy.deepcopy(e.export())
        assert support.state_sha(stopped['model']) == support.state_sha(complete['model'])
        assert support.state_sha(stopped['optimizer']) == support.state_sha(complete['optimizer'])
        assert support.state_sha(stopped['scaler']) == support.state_sha(complete['scaler'])
        again=create('restore_STOP_'+label); again.restore(stopped)
        assert support.state_sha(again.export()) == support.state_sha(stopped)
        try: again.step(); raise AssertionError('automatic consumed-group replay')
        except RuntimeError: pass
        stop_proofs[label]=dict(exact=True, consumed_attempt=stopped['attempts'], calls=stopped['calls'],
            encoded_micros=stopped['inflight']['encoded_micros'], completed_micros=len(stopped['inflight']['completed_micros']))
    for field in ('static_reference_SHA', 'warm_readout_SHA', 'representation'):
        changed=copy.deepcopy(middle); changed[field]='changed'
        try: recovered.restore(changed); raise AssertionError('warm/original anchor identity reset')
        except ValueError: pass
    assert not torch.cuda.is_initialized()
    proof=dict(status='ROOT_V90_ACTUAL_ADAM_CORRECTION_CPU_ENGINE_CONTRACT_PASS_NOT_RESEARCH_TRAINING',
        focused_transaction_reference_tests=11, real_CPU_Adam_enabled_scaler=True, actual_hidden512_final_Linear_parameters=2117664,
        static_teacher_reused_not_fresh_exposure=True, original_anchor_floors_not_recalibrated=True,
        distinct_warm_head_static_teacher=True, exactly_seven_stored_overrides_two_optimizer_tensors=True,
        populated_Adam_scaler_ALLRNG_next_group_exact=True, AMPskip_consumes_source_and_scaler=True,
        rejection_rolls_weights_Adam_back_scaler_source_consumed=True, STOP_exact_no_replay=stop_proofs,
        in_memory_fullstate_bytes=len(stream.getvalue()), reconstructed_algorithm_SHA=support.state_sha(saved),
        anchor_SHA=anchor['anchor_SHA'], original_teacher_refs=refs, original_bank_ref=reference,
        actual_learned_FDT_model_calls=0, actual_learned_parameter_updates=0, GPU_calls=0, CUDA_initialized=False,
        limitation='Tiny synthetic encoder and3CPU TRAIN sentences; transaction algebra tests additionally exercise corrected acceptance/rejection and STOP after correction. This engine fixture is not actual learned full-size preflight, CUDA restart or a runnable sealed worker/owner/plan.',
        fixture_seconds=time.monotonic()-began, Cfree=shutil.disk_usage(ROOT).free)
    io.atomic_json(directory/'root_engine_cpu_review.json', proof)
    files=[Path(__file__).resolve(), ROOT/'scripts/feasible_critical_engine_v90.py', ROOT/'research/feasible_adam_transaction_v90.py',
        ROOT/'research/diagnostic_reference_v90.py', ROOT/'research/feasible_readout_delta_v89.py', ROOT/'research/immutable_reference_anchor_v89.py',
        ROOT/'tests/test_feasible_adam_transaction_v90.py', ROOT/'tests/test_diagnostic_reference_v90.py', parent, fp, tp,
        directory/'root_engine_cpu_review.json', *(Path(r['targets']) for r in refs)]
    io.atomic_json(directory/'evidence_SHA.json', {str(f): support.sha(f) for f in files})
    print(json.dumps(proof))


if __name__ == '__main__': main()
