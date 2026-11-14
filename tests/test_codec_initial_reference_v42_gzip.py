import copy
import gzip
import json
import random
import sys
import zlib
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
import codec_initial_reference_v42_gzip as storage
import text_generation_support_v1 as support


def fixture(tmp_path):
    shared = {'model': {'x': torch.arange(8, dtype=torch.float32)},
        'optimizer': {'state': {0: {'step': torch.tensor(7.), 'exp_avg': torch.ones(8)}}},
        'optimizer_parameter_names': [['x']], 'scaler': {'scale': 1024., '_growth_tracker': 7},
        'frozen_sha': 'CPU_FIXTURE_ONLY', 'torch_rng': torch.get_rng_state().clone(),
        'cuda_rng': [torch.tensor([1, 2, 3], dtype=torch.uint8)],
        'python_rng': random.getstate(), 'numpy_rng': {'keys': [1, 2, 3], 'position': 1}}
    parent = dict(shared, attempts=7, cursor=56, updates=7, history=[{'rows': [0, 1]}])
    state = dict(shared, schema='CPU_STORAGE_ONLY', attempts=0, updates=0, cursor=0, skips=0,
                 gradients={}, history=[], evaluations={}, allocation_calls={'encode': 0, 'decoder': 0, 'denoise': 0},
                 orders=[[0, 1, 2]], ancestry={'history': copy.deepcopy(parent['history'])})
    source = tmp_path/'immutable_parent.pt'; torch.save(parent, source)
    return state, parent, source, support.sha(source), support.state_sha(parent)


def test_fullstate_RNG_exact_roundtrip_and_global_RNG_unchanged(tmp_path):
    state, parent, source, sha, algorithm = fixture(tmp_path)
    before = support.state_sha(state); rng = torch.get_rng_state().clone()
    restored, sizes = storage.safe_memory_roundtrip(state, parent, source, sha, algorithm)
    assert support.state_sha(restored) == before == support.state_sha(state)
    assert torch.equal(torch.get_rng_state(), rng) and not torch.cuda.is_initialized()
    assert 0 < sizes['compressed_bytes'] <= storage.MAX_COMPRESSED_BYTES
    assert 0 < sizes['serialized_bytes'] <= storage.MAX_SERIALIZED_BYTES
    assert support.sha(source) == sha


def test_file_marker_parent_required_safeload_and_no_overwrite(tmp_path):
    state, parent, source, sha, algorithm = fixture(tmp_path)
    dest = tmp_path/'reference'
    marker = storage.save_initial_reference(dest, state, source, sha, algorithm)
    path = dest/'checkpoints'/marker['file']
    loaded = storage.load_initial_reference(path, source, sha, algorithm)
    assert support.state_sha(loaded) == support.state_sha(state)
    assert not marker['standalone'] and marker['compression'] == 'gzip'
    assert path.read_bytes()[:2] == b'\x1f\x8b' and support.sha(source) == sha
    with pytest.raises(FileExistsError):
        storage.save_initial_reference(dest, state, source, sha, algorithm)


def test_serialization_write_and_decompression_caps(monkeypatch):
    monkeypatch.setattr(storage, 'MAX_SERIALIZED_BYTES', 1024)
    out = storage._BoundedBytesIO()
    with pytest.raises(OSError):
        out.write(b'x'*1025)
    assert len(out.getvalue()) == 0
    with pytest.raises(ValueError, match='decompressed'):
        storage._unpack(gzip.compress(b'x'*1025, mtime=0))


@pytest.mark.parametrize('mutation', ['CRC', 'TRUNCATE', 'TRAILING', 'MEMBER'])
def test_corrupt_truncated_or_extra_gzip_data_rejected(tmp_path, mutation):
    state, parent, source, sha, algorithm = fixture(tmp_path)
    ref = storage.base.make_reference(state, parent, source, sha, algorithm)
    blob, _ = storage._pack(ref)
    if mutation == 'CRC':
        value = bytearray(blob); value[-8] ^= 1; blob = bytes(value)
    elif mutation == 'TRUNCATE':
        blob = blob[:-5]
    elif mutation == 'TRAILING':
        blob += b'UNEXPECTED_BYTES'
    else:
        blob += gzip.compress(b'SECOND_MEMBER', mtime=0)
    with pytest.raises((ValueError, zlib.error, OSError, EOFError)):
        storage._unpack(blob)


def test_wrong_parent_updated_state_and_modified_file_rejected(tmp_path):
    state, parent, source, sha, algorithm = fixture(tmp_path)
    with pytest.raises(ValueError):
        storage.safe_memory_roundtrip(dict(state, attempts=1), parent, source, sha, algorithm)
    changed = copy.deepcopy(parent); changed['model']['x'][0] += 1
    with pytest.raises(ValueError):
        storage.safe_memory_roundtrip(state, changed, source, sha, algorithm)
    marker = storage.save_initial_reference(tmp_path/'ref', state, source, sha, algorithm)
    path = tmp_path/'ref/checkpoints'/marker['file']
    with pytest.raises(ValueError):
        storage.load_initial_reference(path, source, 'WRONG_PARENT_SHA', algorithm)
    path.write_bytes(path.read_bytes()+b'CORRUPTION_CPU_FIXTURE_ONLY')
    with pytest.raises(ValueError):
        storage.load_initial_reference(path, source, sha, algorithm)


def test_compressed_limit_rejects_before_decode(monkeypatch):
    monkeypatch.setattr(storage, 'MAX_COMPRESSED_BYTES', 4)
    with pytest.raises(ValueError, match='input cap'):
        storage._unpack(b'12345')


def test_enabled_scaler_engine_restore_exact_next_update(tmp_path):
    # Same tiny populated91 engine as the closed V41 CPU contract; not ability
    # evidence, nor an actual full-size V42 worker or CUDA recovery replay.
    from test_three_memory_worker_v41 import new_epoch
    engine, parent = new_epoch(tmp_path)
    initial = copy.deepcopy(engine.export())
    source = tmp_path/'populated91_parent.pt'; torch.save(parent, source)
    sha, algorithm = support.sha(source), support.state_sha(parent)
    marker = storage.save_initial_reference(tmp_path/'gzip', initial, source, sha, algorithm)
    loaded = storage.load_initial_reference(tmp_path/'gzip/checkpoints'/marker['file'], source, sha, algorithm)
    expected = engine.step(); final = copy.deepcopy(engine.export())
    recovered, _ = new_epoch(tmp_path); recovered.restore(loaded)
    assert recovered.scaler.is_enabled() and len(recovered.optimizer.state) == 91
    assert recovered.step() == expected
    assert support.state_sha(recovered.export()) == support.state_sha(final)
