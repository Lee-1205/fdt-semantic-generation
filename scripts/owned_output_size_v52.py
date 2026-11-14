"""Count current files without treating an atomic JSON rename as a model failure."""
from pathlib import Path

def owned_output_bytes(directory):
    total=0
    for path in Path(directory).rglob('*'):
        try:
            info=path.stat()
        except FileNotFoundError:
            # The OS owner just replaced its own temporary snapshot. No deletion
            # or retry of a model attempt is performed here.
            continue
        if path.is_dir():continue
        total+=info.st_size
    return total

def cpu_regression():
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory);(base/'stable').write_bytes(b'abc')
        absent=base/'state.json.replaced.tmp'
        original=Path.rglob
        def racing(self,pattern):
            yield absent
            yield from original(self,pattern)
        with patch.object(Path,'rglob',racing):
            assert owned_output_bytes(base)==3
    return dict(atomic_missing_file_ignored=True,stable_bytes_still_counted=True)
