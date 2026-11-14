import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_resume_rejects_changed_code_or_checkpoint(tmp_path):
    from astra_evidence_io import ensure_run_identity
    identity = {'code': 'abc', 'checkpoint': '123', 'protocol': 'greedy'}
    ensure_run_identity(tmp_path, identity)
    ensure_run_identity(tmp_path, identity)
    for field in ('code', 'checkpoint', 'protocol'):
        with pytest.raises(RuntimeError, match='identity'):
            ensure_run_identity(tmp_path, {**identity, field: 'changed'})


def test_legacy_results_cannot_silently_resume(tmp_path):
    from astra_evidence_io import ensure_run_identity
    (tmp_path / 'case_000.json').write_text('{}')
    with pytest.raises(RuntimeError, match='manifest'):
        ensure_run_identity(tmp_path, {'code': 'abc'})


def test_partial_json_is_not_a_completed_result(tmp_path):
    from astra_evidence_io import completed
    path = tmp_path / 'case.json'
    assert not completed(path)
    path.write_text('{"partial":')
    with pytest.raises(RuntimeError, match='incomplete'):
        completed(path)


def test_atomic_publish_preserves_old_evidence_and_rejects_nan(tmp_path):
    from astra_evidence_io import save
    path = tmp_path / 'case.json'
    with pytest.raises(ValueError):
        save(path, {'bad': float('nan')})
    assert not path.exists()
    save(path, {'value': 7})
    assert json.loads(path.read_text()) == {'value': 7}
    with pytest.raises(FileExistsError):
        save(path, {'value': 8})
    assert json.loads(path.read_text()) == {'value': 7}
