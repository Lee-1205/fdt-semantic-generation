"""Lossless, bounded gzip storage for INITIAL parent-required references only.

Reuse the immutable format39 payload/parent/digest/zero-work semantics. Never
compress or rewrite updated fullstates, model tensors or optimizer trajectories.
Physical32MiB remains fixed; the distinct CPU serialization bound is64MiB.
This module is not a worker, execution plan, or authorization to resume a run.
"""
import gzip
import io as binary_io
import os
import zlib
from pathlib import Path
import torch
import codec_initial_reference_v39 as base
import learning_io_v2 as learning_io
import text_generation_support_v1 as support

SCHEMA = 'fdt_initial_parent_reference_v42_gzip_v1'
MAX_COMPRESSED_BYTES = 32*1024**2
MAX_SERIALIZED_BYTES = 64*1024**2
GZIP_LEVEL = 6
SHARED = base.SHARED
POLICY = {**base.POLICY, 'format': SCHEMA, 'shared_fields': list(SHARED),
          'payload_reference_schema': base.SCHEMA, 'compression': 'gzip',
          'maximum_compressed_bytes': MAX_COMPRESSED_BYTES,
          'maximum_serialized_bytes': MAX_SERIALIZED_BYTES,
          'gzip_level': GZIP_LEVEL, 'concatenated_or_trailing_data': 'reject',
          'bounded_decompression_and_CRC_required': True}
STORAGE_BOUND = {'maximum_initial_reference_bytes': MAX_COMPRESSED_BYTES,
                 'maximum_serialized_reference_bytes': MAX_SERIALIZED_BYTES}


class _BoundedBytesIO(binary_io.BytesIO):
    def write(self, data):
        if self.tell()+len(data) > MAX_SERIALIZED_BYTES:
            raise OSError('fixed serialized reference CPU size cap')
        return super().write(data)


def _pack(reference):
    stream = _BoundedBytesIO()
    torch.save(reference, stream)
    raw = stream.getvalue()
    blob = gzip.compress(raw, compresslevel=GZIP_LEVEL, mtime=0)
    if len(blob) > MAX_COMPRESSED_BYTES:
        raise ValueError('fixed compressed reference file cap')
    return blob, {'serialized_bytes': len(raw), 'compressed_bytes': len(blob),
                  'compression': 'gzip', 'gzip_level': GZIP_LEVEL}


def _unpack(blob):
    if len(blob) > MAX_COMPRESSED_BYTES:
        raise ValueError('fixed compressed reference input cap')
    decoder = zlib.decompressobj(wbits=16+zlib.MAX_WBITS)
    raw = decoder.decompress(blob, MAX_SERIALIZED_BYTES+1)
    if len(raw) > MAX_SERIALIZED_BYTES or decoder.unconsumed_tail:
        raise ValueError('fixed decompressed reference CPU cap')
    if not decoder.eof:
        raise ValueError('incomplete gzip reference or CRC trailer')
    if decoder.unused_data:
        raise ValueError('concatenated gzip or trailing data forbidden')
    # zlib has validated the gzip CRC/trailer before eof becomes true.
    reference = torch.load(binary_io.BytesIO(raw), map_location='cpu', weights_only=True)
    if not isinstance(reference, dict):
        raise ValueError('reference dictionary required')
    return reference, len(raw)


def _read_blob(path):
    if path.stat().st_size > MAX_COMPRESSED_BYTES:
        raise ValueError('fixed reference file size cap')
    with path.open('rb') as file:
        blob = file.read(MAX_COMPRESSED_BYTES+1)
    if len(blob) > MAX_COMPRESSED_BYTES:
        raise ValueError('reference grew beyond fixed read cap')
    return blob


def safe_memory_roundtrip(state, parent, source, source_sha, source_algorithm):
    """Return exact restored fullstate and compressed/serialized size evidence."""
    reference = base.make_reference(state, parent, source, source_sha, source_algorithm)
    blob, sizes = _pack(reference)
    loaded, raw_size = _unpack(blob)
    if raw_size != sizes['serialized_bytes']:
        raise ValueError('serialized reference length mismatch')
    restored = base.materialize(loaded, parent, source, source_sha, source_algorithm)
    if support.state_sha(restored) != support.state_sha(state):
        raise ValueError('gzip fullstate roundtrip mismatch')
    return restored, sizes


def save_initial_reference(output, state, source, source_sha, source_algorithm):
    directory = Path(output)/'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'state_0000_reference_gzip.pt'
    temp = path.with_suffix('.pt.tmp')
    if path.exists() or temp.exists():
        raise FileExistsError('reference overwrite or abandoned candidate forbidden')
    source = Path(source).resolve()
    if support.sha(source) != source_sha:
        raise ValueError('parent file changed before read')
    parent = torch.load(source, weights_only=True, map_location='cpu')
    reference = base.make_reference(state, parent, source, source_sha, source_algorithm)
    blob, sizes = _pack(reference)
    with temp.open('xb') as file:
        file.write(blob); file.flush(); os.fsync(file.fileno())
    learning_io.replace_with_retry(temp, path)
    loaded, raw_size = _unpack(_read_blob(path))
    restored = base.materialize(loaded, parent, source, source_sha, source_algorithm)
    if raw_size != sizes['serialized_bytes'] or support.state_sha(restored) != support.state_sha(state):
        raise ValueError('saved gzip reference roundtrip mismatch')
    if support.sha(source) != source_sha:
        raise ValueError('parent file changed during read')
    marker = {'file': path.name, 'sha256': support.sha(path), 'bytes': path.stat().st_size,
        'algorithm_sha256': support.state_sha(restored), 'attempts': 0, 'updates': 0,
        'sampler_cursor': 0, 'full_state_verified': True, 'standalone': False,
        'checkpoint_format': SCHEMA, 'source_path': str(source), 'source_sha256': source_sha,
        'source_algorithm_sha256': source_algorithm, **sizes}
    learning_io.atomic_json(path.with_suffix('.json'), marker)
    learning_io.atomic_json(Path(output)/'latest.json', marker)
    return marker


def load_initial_reference(path, source, source_sha, source_algorithm):
    path, source = Path(path), Path(source).resolve()
    marker = learning_io.load_json(path.with_suffix('.json'))
    if (marker.get('checkpoint_format') != SCHEMA or marker.get('compression') != 'gzip'
            or marker.get('standalone') is not False):
        raise ValueError('gzip parent-required reference marker')
    if (support.sha(path) != marker['sha256'] or path.stat().st_size != marker['bytes']
            or support.sha(source) != source_sha):
        raise ValueError('reference/parent file SHA or length')
    if (marker['source_path'], marker['source_sha256'], marker['source_algorithm_sha256']) != (
            str(source.resolve()), source_sha, source_algorithm):
        raise ValueError('pinned parent marker identity')
    reference, raw_size = _unpack(_read_blob(path))
    if raw_size != marker['serialized_bytes']:
        raise ValueError('serialized metadata size marker')
    parent = torch.load(source, weights_only=True, map_location='cpu')
    state = base.materialize(reference, parent, source, source_sha, source_algorithm)
    if support.state_sha(state) != marker['algorithm_sha256']:
        raise ValueError('reference marker fullstate digest')
    return state
