"""Storage-only wrapper: original model/data/train/runtime files remain unchanged."""
import contextlib
import fcntl
import gc
import hashlib
import json
import os
import pathlib
import time
import torch
from checkpoint_store import CheckpointStore, GoogleDriveBackend, PersistenceError, json_file
from train import Trainer, Plan, PHASES, evaluate, make_split, source_digest, write_json
from model import Config
from run_control import Budget, run_with_budget

EXPECTED_SOURCE = 'de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79'
ROOT_FOLDER_ID = '1cH9Ha58Uc3HTHhp9BV9yJpvgE84-ABil'
RUN_NAME = 'FDT100_T4_recovery_20261006_v2'


def wrapper_digest():
    h = hashlib.sha256()
    for name in ('checkpoint_store.py', 'durable_runner.py', 'run_control.py'):
        h.update(name.encode())
        h.update((pathlib.Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def run_folder(service):
    names = service.files().list(
        q=f"'{ROOT_FOLDER_ID}' in parents and trashed=false and name='{RUN_NAME}'",
        fields='files(id,name,mimeType)', pageSize=100).execute(num_retries=3).get('files', [])
    if len(names) > 1:
        raise PersistenceError('동일 이름의 복구 폴더가 복수입니다. 자동 선택하지 않습니다.')
    if names:
        if names[0]['mimeType'] != 'application/vnd.google-apps.folder':
            raise PersistenceError('복구 폴더 이름이 일반 파일과 충돌합니다.')
        return names[0]['id']
    return service.files().create(body={'name': RUN_NAME,
        'mimeType': 'application/vnd.google-apps.folder', 'parents': [ROOT_FOLDER_ID]},
        fields='id').execute(num_retries=3)['id']


class DurableTrainer(Trainer):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        self._unsafe_step = False
        self._persistence_failed = False
        self._saved_signature = None
        self._step_diagnostics = {}
        super().__init__(*args, **kwargs)

    def _restore(self):
        marker = self.store.committed
        if marker is None:
            raise PersistenceError('검증된 원격 완료 marker가 없습니다.')
        payload = torch.load(self.out / marker['file'], map_location='cpu', weights_only=True)
        if (json.loads(json.dumps(payload['contract'])) != marker['contract'] or
            payload['updates'] != marker['optimizer_updates'] or
            payload['attempts'] != marker['optimizer_attempts']):
            raise PersistenceError('체크포인트 내부 계약/업데이트와 marker가 다릅니다.')
        required = ('optimizer', 'scaler', 'phase', 'phase_step', 'sampler_rng',
                    'noise_rng', 'torch_rng', 'cuda_rng')
        if not all(key in payload for key in required):
            raise PersistenceError('정확 재개에 필요한 optimizer/AMP/RNG 상태 누락')
        self._step_diagnostics = payload.get('last_step_diagnostics', {})
        del payload
        gc.collect()
        super()._restore()
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise PersistenceError('복구한 모델 파라미터에 비유한 값이 있습니다.')
        self._saved_signature = self._signature()

    def _signature(self):
        return (self.updates, self.attempts, self.phase, self.phase_step,
                len(self.gates), len(self.history))

    def save(self):
        if self._unsafe_step or self._persistence_failed:
            print('UNCOMMITTED_STATE_NOT_SAVED: 직전 검증본을 보존합니다.', flush=True)
            return
        signature = self._signature()
        if signature == self._saved_signature:
            return
        payload = {
            'contract': self.contract, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'scaler': self.scaler.state_dict(),
            'parent': self.parent, 'sampler_rng': self.sampler.get_state(),
            'noise_rng': self.noise.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
            'last_step_diagnostics': self._step_diagnostics,
        }
        for name in ('phase', 'phase_step', 'updates', 'attempts', 'exposure',
                     'query_exposure', 'ground_exposure', 'gates', 'history',
                     'gradient_audit', 'elapsed'):
            payload[name] = getattr(self, name)
        path = self.out / 'pending_checkpoint.pt'
        temporary = self.out / 'pending_checkpoint.pt.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, path)
        status = {**self.state(), 'last_step_diagnostics': self._step_diagnostics}
        try:
            self.store.commit(path, self.contract, status)
        except BaseException:
            self._persistence_failed = True
            print('DRIVE_SAVE_FAILED: 추가 optimizer 업데이트를 중단합니다.', flush=True)
            raise
        self._saved_signature = signature
        # CheckpointStore owns the local status too: do not erase its backup marker.

    def _step(self):
        self._unsafe_step = True
        start = time.monotonic()
        losses, successful = super()._step()
        torch.cuda.synchronize()
        finite_gradients = all(bool(torch.isfinite(p.grad).all())
                              for p in self.model.parameters() if p.grad is not None)
        finite_parameters = all(bool(torch.isfinite(p).all()) for p in self.model.parameters())
        if not finite_parameters or (successful and not finite_gradients):
            raise FloatingPointError('nonfinite model/gradient: 직전 Drive 검증본으로 복구해야 합니다.')
        self._step_diagnostics = {
            **losses, 'optimizer_updated': successful, 'finite_gradients': finite_gradients,
            'finite_model_parameters': finite_parameters,
            'step_seconds_excluding_save': time.monotonic() - start,
            'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        }
        self._unsafe_step = False
        return losses, successful

    def record_step(self, losses):
        if self.updates % self.plan.log_every == 0 or self.phase_step == 1:
            log = {'update': self.updates, 'phase': PHASES[self.phase],
                   'phase_step': self.phase_step, **losses}
            self.history.append(log)
            print(json.dumps(log), flush=True)

    def validate_phase(self):
        name = PHASES[self.phase]
        if self.gates and self.gates[-1]['phase'] == name and not self.gates[-1]['passed']:
            print('GATE_FAILED: 동일 계획을 추가 학습으로 변경하지 않습니다.', flush=True)
            return False
        result = evaluate(self.model, self.dev, self.device, self.plan.micro_batch, full=False)
        measure = result['ground_accuracy'] if self.phase == 0 else result['initializer']['accuracy']
        required = (self.plan.ground_gate, self.plan.semantic_gate, 0.)[self.phase]
        gate = {'phase': name, 'measured': measure, 'required': required,
                'passed': measure >= required}
        scope = {'source': self.contract['source'], 'wrapper': wrapper_digest(),
                 'updates': self.updates, 'phase': name}
        self.store.publish_artifact(json_file(self.out / (name + '_dev.json'), result),
                                    name + '_dev.json', scope)
        self.gates.append(gate)
        print(json.dumps({'gate': gate}, ensure_ascii=False), flush=True)
        if not gate['passed']:
            return False
        self._unsafe_step = True
        self.phase += 1
        self.phase_step = 0
        # Preserve the last optimizer at completion instead of silently discarding it.
        if self.phase < 3:
            self._optimizer()
        self._unsafe_step = False
        return True

    def finish(self, budget):
        if self.phase != 3 or self.store.committed is None:
            raise PersistenceError('완료 체크포인트 없이 최종 평가/내보내기를 하지 않습니다.')
        scope = {'checkpoint_sha256': self.store.committed['sha256'],
                 'source': self.contract['source'], 'wrapper': wrapper_digest(),
                 'updates': self.updates}
        runtime_marker = self.store.artifact_marker('runtime.pt', scope)
        if runtime_marker is None:
            if budget.expired():
                return False
            super().export()
            runtime_marker = self.store.publish_artifact(self.out / 'runtime.pt', 'runtime.pt', scope)
        if self.store.artifact_marker('runtime.json', scope) is None:
            if budget.expired():
                return False
            metadata = {'file': 'runtime.pt', 'sha256': runtime_marker['sha256'],
                        'parameters': self.model.counts()['parameters']}
            self.store.publish_artifact(json_file(self.out / 'runtime.json', metadata), 'runtime.json', scope)
        stages = [
            ('development_evaluation.json', lambda: self.dev),
            ('graph_test_evaluation.json', lambda: make_split('test', self.plan.test_graphs)),
            ('heldout_phrase_test_evaluation.json', lambda: make_split('test', 64, seed=20001, heldout=True)),
            ('larger_graph_test_evaluation.json', lambda: make_split('test', 64, seed=40001, n_range=(7, 8))),
        ]
        for name, items in stages:
            if self.store.artifact_marker(name, scope) is not None:
                continue
            if budget.expired():
                return False
            result = evaluate(self.model, items(), self.device, self.plan.micro_batch, (17, 29, 43))
            self.store.publish_artifact(json_file(self.out / name, result), name, scope)
            print('EVALUATION_STAGE_VERIFIED=' + name, flush=True)
        if budget.expired():
            return False
        final = {key: self.store.load_artifact(key + '_evaluation.json', scope)
                 for key in ('graph_test', 'heldout_phrase_test', 'larger_graph_test')}
        if self.store.artifact_marker('final_evaluation.json', scope) is None:
            self.store.publish_artifact(json_file(self.out / 'final_evaluation.json', final),
                                        'final_evaluation.json', scope)
        manifest = {'training_complete': True, 'evaluations_complete': True, 'scope': scope,
                    'runtime': runtime_marker,
                    'reports': {name: self.store.artifact_marker(name, scope)
                                for name in [row[0] for row in stages] + ['final_evaluation.json']}}
        self.store.publish_artifact(json_file(self.out / 'evaluation_complete.json', manifest),
                                    'evaluation_complete.json', scope)
        print('PLAN_AND_EVALUATION_COMPLETE: 기존 계획을 변경하거나 추가 학습하지 않습니다.', flush=True)
        return True

    def run(self, max_updates=None, wall_seconds=21600):
        return run_with_budget(self, wall_seconds, max_updates)


@contextlib.contextmanager
def local_writer_lock(local):
    local.mkdir(parents=True, exist_ok=True)
    with open(local / 'writer.lock', 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PersistenceError('이 런타임에서 다른 학습 셀이 실행 중입니다.') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def put_immutable(store, path, name):
    from checkpoint_store import hashes
    item = store.named(name)
    if item:
        digest = hashes(path)
        if int(item.get('size', -1)) != digest['size'] or item.get('md5Checksum') != digest['md5Checksum']:
            raise PersistenceError('보존된 소스/계획 파일과 다릅니다: ' + name)
        return
    store.put_verified(path, name)


def start(service, wall_seconds=21600):
    Budget(wall_seconds)  # Validate the launch setting before any remote writes.
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('T4 CUDA 필요. CPU/다른 GPU 대체 없이 중단합니다.')
    if source_digest() != EXPECTED_SOURCE:
        raise RuntimeError('모델 source fingerprint 불일치. 실행하지 않습니다.')
    local = pathlib.Path('/content/fdt100_durable_v2_state')
    with local_writer_lock(local):
        return _start_locked(service, local, wall_seconds)


def _start_locked(service, local, wall_seconds):
    launch_started = time.monotonic()
    folder_id = run_folder(service)
    store = CheckpointStore(GoogleDriveBackend(service, folder_id), local, wrapper_digest())
    marker = store.restore(EXPECTED_SOURCE)
    if marker:
        contract = marker['contract']
        plan = dict(contract['plan'])
        plan['phase_steps'] = tuple(plan['phase_steps'])
        cfg = Config(**contract['config'])
        trainer = DurableTrainer(local, Plan(**plan), cfg, resume=True, store=store)
        print('EXACT_RESUME: 저장된 source/config/data/plan/environment를 검증했습니다.', flush=True)
    else:
        print('FRESH_RECONSTRUCTION: 이전 2240-update 가중치 없음. 새 실행 step 0부터 시작합니다.', flush=True)
        trainer = DurableTrainer(local, Plan(), Config(), store=store)
    try:
        if marker is None:
            trainer.save()  # Persist initialization before the first optimizer update.
        return _execute(trainer, store, folder_id, wall_seconds, launch_started)
    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def _execute(trainer, store, folder_id, wall_seconds, launch_started):
    local = trainer.out
    for name in ('model.py', 'data.py', 'train.py', 'runtime.py',
                 'checkpoint_store.py', 'durable_runner.py', 'run_control.py'):
        put_immutable(store, pathlib.Path(__file__).parent / name, name)
    put_immutable(store, local / 'data_manifest.json', 'data_manifest.json')
    put_immutable(store, json_file(local / 'recovery_plan.json', {
        'run_kind': 'fresh_reconstruction_after_missing_weights',
        'previous_weights_recovered': False, 'contract': trainer.contract,
        'durability_wrapper_sha256': wrapper_digest(),
        'save_interval_optimizer_attempts': 1,
        'initial_phase_updates': [480, 320, 640],
        'future_60_15_25_plan_executed': False,
        'default_wall_seconds_per_launch': 21600,
        'colab_runtime_end_prevention': False,
        'durability_limit': 'An interruption can discard the single currently uncommitted update.',
    }), 'recovery_plan.json')
    remaining = max(0., wall_seconds - (time.monotonic() - launch_started))
    status = trainer.run(wall_seconds=remaining)
    store.put_verified(json_file(local / 'launch_status.json', {
        **status, 'wall_seconds_requested': wall_seconds,
        'launch_elapsed_seconds': time.monotonic() - launch_started,
        'weights_saved_to_drive': bool(store.committed), 'drive_backup': store.committed,
        'previous_2240_update_weights_recovered': False}), 'launch_status.json')
    print('DURABLE_TRAINING_STATUS=' + json.dumps({
        'updates': status['optimizer_updates'], 'phase': status['phase'],
        'drive_folder_id': folder_id, 'weights_saved_to_drive': bool(store.committed),
        'previous_2240_update_weights_recovered': False}), flush=True)
    return status
