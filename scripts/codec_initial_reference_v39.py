"""Lossless initial-only checkpoint reference to an immutable preserved parent.

Updated checkpoints remain self-contained. No neural state is quantized,
removed, or recomputed by a model. This format requires its pinned parent.
"""
import io as binary_io
import os
from pathlib import Path
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

SHARED = ('model', 'optimizer', 'optimizer_parameter_names', 'scaler',
          'frozen_sha', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')
SCHEMA = 'fdt_initial_parent_reference_v39'
POLICY = {'format': SCHEMA, 'initial_only': True, 'parent_required': True,
          'shared_fields': list(SHARED), 'shared_state_must_be_identical': True,
          'updated_checkpoints': 'self_contained', 'quantization': False}
STORAGE_BOUND = {'maximum_full_files': 3, 'maximum_full_file_bytes': 850*1024**2,
                 'maximum_initial_reference_bytes': 32*1024**2,
                 'assessment_bytes': 128*1024**2, 'other_bytes': 64*1024**2,
                 'total_bound_bytes': 2774*1024**2, 'output_cap_bytes': 3*1024**3}


def make_reference(state, parent, source, source_sha, source_algorithm):
    if any(state[k] != 0 for k in ('attempts', 'updates', 'cursor', 'skips')):
        raise ValueError('reference only before any attempted update')
    if state['gradients'] or state['history'] or state['evaluations'] or any(state['allocation_calls'].values()):
        raise ValueError('nonempty initial state')
    if support.state_sha(parent) != source_algorithm:
        raise ValueError('parent algorithm')
    if support.state_sha({k:state[k] for k in SHARED}) != support.state_sha({k:parent[k] for k in SHARED}):
        raise ValueError('initial shared model/Adam/scaler/ALLstoredRNG differ')
    return {'reference_schema': SCHEMA, 'source_path': str(Path(source).resolve()),
            'source_sha256': source_sha, 'source_algorithm_sha256': source_algorithm,
            'shared_fields': list(SHARED), 'metadata': {k:v for k,v in state.items() if k not in SHARED},
            'reconstructed_algorithm_sha256': support.state_sha(state)}


def materialize(reference, parent, source, source_sha, source_algorithm):
    expected = {'reference_schema', 'source_path', 'source_sha256', 'source_algorithm_sha256',
                'shared_fields', 'metadata', 'reconstructed_algorithm_sha256'}
    if set(reference) != expected or reference['reference_schema'] != SCHEMA:
        raise ValueError('reference schema')
    if (reference['source_path'], reference['source_sha256'], reference['source_algorithm_sha256']) != (
            str(Path(source).resolve()), source_sha, source_algorithm):
        raise ValueError('pinned parent identity')
    if reference['shared_fields'] != list(SHARED) or set(SHARED) & set(reference['metadata']):
        raise ValueError('reference field ownership')
    if support.state_sha(parent) != source_algorithm:
        raise ValueError('parent content changed')
    state = dict(reference['metadata'], **{k:parent[k] for k in SHARED})
    if support.state_sha(state) != reference['reconstructed_algorithm_sha256']:
        raise ValueError('reconstructed fullstate digest')
    # The zero-work guard also applies to loads, not only the original save.
    make_reference(state, parent, source, source_sha, source_algorithm)
    return state


def safe_memory_roundtrip(state, parent, source, source_sha, source_algorithm):
    reference = make_reference(state, parent, source, source_sha, source_algorithm)
    stream = binary_io.BytesIO(); torch.save(reference, stream)
    if stream.tell() > STORAGE_BOUND['maximum_initial_reference_bytes']:
        raise ValueError('initial reference size cap')
    size = stream.tell(); stream.seek(0)
    restored = materialize(torch.load(stream, weights_only=True, map_location='cpu'),
                           parent, source, source_sha, source_algorithm)
    return restored, size


def save_initial_reference(output, state, source, source_sha, source_algorithm):
    source = Path(source).resolve()
    if support.sha(source) != source_sha:
        raise ValueError('parent file changed before read')
    parent = torch.load(source, weights_only=True, map_location='cpu')
    reference = make_reference(state, parent, source, source_sha, source_algorithm)
    directory = Path(output)/'checkpoints'; directory.mkdir(exist_ok=True)
    path = directory/'state_0000_reference.pt'; temp = path.with_suffix('.pt.tmp')
    if path.exists() or temp.exists(): raise FileExistsError('no reference overwrite')
    with temp.open('xb') as file:
        torch.save(reference, file); file.flush(); os.fsync(file.fileno())
    if temp.stat().st_size > STORAGE_BOUND['maximum_initial_reference_bytes']:
        raise ValueError('initial reference file cap')
    io.replace_with_retry(temp, path)
    restored = materialize(torch.load(path, weights_only=True, map_location='cpu'),
                           parent, source, source_sha, source_algorithm)
    if support.sha(source) != source_sha: raise ValueError('parent file changed during read')
    meta = {'file':path.name, 'sha256':support.sha(path), 'bytes':path.stat().st_size,
            'algorithm_sha256':support.state_sha(restored), 'attempts':0, 'updates':0,
            'sampler_cursor':0, 'full_state_verified':True, 'standalone':False,
            'checkpoint_format':SCHEMA, 'source_path':str(source), 'source_sha256':source_sha}
    io.atomic_json(path.with_suffix('.json'), meta); io.atomic_json(Path(output)/'latest.json', meta)
    return meta


def load_initial_reference(path, source, source_sha, source_algorithm):
    path, source = Path(path), Path(source).resolve()
    mark = io.load_json(path.with_suffix('.json'))
    if support.sha(path) != mark['sha256'] or support.sha(source) != source_sha:
        raise ValueError('reference/parent file SHA')
    parent = torch.load(source, weights_only=True, map_location='cpu')
    state = materialize(torch.load(path, weights_only=True, map_location='cpu'),
                        parent, source, source_sha, source_algorithm)
    if support.state_sha(state) != mark['algorithm_sha256']:
        raise ValueError('reference marker fullstate digest')
    return state
