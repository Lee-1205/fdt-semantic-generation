"""Single-run immutable evidence with atomic publication and strict resume identity."""
import json
import os
import tempfile
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    # Serialize first: invalid numerical evidence must never create a result file.
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link publishes a complete file without replacing
        # an existing result, including if another writer won the race.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def completed(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, (dict, list)):
            raise ValueError('unexpected result type')
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f'incomplete evidence: {path}; preserve it and use a new run directory') from error
    return True


def ensure_run_identity(directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'run_identity.json'
    if completed(manifest):
        if json.loads(manifest.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('run identity changed; use a new output directory')
    else:
        if list(directory.glob('*.json')):
            raise RuntimeError('existing evidence has no run identity manifest; use a new output directory')
        save(manifest, identity)
