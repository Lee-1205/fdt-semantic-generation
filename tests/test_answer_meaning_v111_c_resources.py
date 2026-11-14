import sys,time,collections
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_answer_meaning_v111_c_cpu as worker


def test_actual_C_budget_floor_pending_output_cap_and_closed_callcap(monkeypatch):
    tmp_path=Path(__file__).resolve().parent/'NONEXISTENT_answer_budget_fixture'
    assert not tmp_path.exists()  # no temporary file or default-temp ACL needed.
    budget=worker.Budget(dict(output=str(tmp_path),cutoff=time.time()+60))
    usage=collections.namedtuple('Usage','total used free')
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda _:usage(0,0,50*1024**3+1024-1))
    with pytest.raises(OSError):budget.before(pending=1024)
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda _:usage(0,0,110*1024**3))
    monkeypatch.setattr(worker,'owned_output_bytes',lambda _:worker.OUTPUT+1)
    with pytest.raises(OSError):budget.before()
    budget.calls['denoise']=41000
    with pytest.raises(RuntimeError):budget.charge('denoise')
    with pytest.raises(RuntimeError):budget.charge('encode')


def test_offline_semantic_score_accepts_inverse_expression_and_rejects_opposite():
    row=dict(answer='A is larger than B.')
    assert worker.score('B is smaller than A.',row)['partial_grammar_semantic_match']
    assert worker.score('A > B',row)['partial_grammar_semantic_match']
    assert not worker.score('A is smaller than B.',row)['partial_grammar_semantic_match']
    assert worker.score('It is unknown.',dict(answer='It is unknown.'))['partial_grammar_semantic_match']
