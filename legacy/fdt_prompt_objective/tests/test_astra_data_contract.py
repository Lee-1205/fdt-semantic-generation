import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from astra_data_contract import retrieval_contract,python_contract,require_semantic_clearance


def test_copy_target_must_exist_in_visible_keyed_record():
    q='Copy only the complete value stored at K00.\nAnswer:'
    assert retrieval_contract('Records:\nK00=AB1234\nK01=CD5678\n'+q,' AB1234')=='PASS'
    assert retrieval_contract('5678\nK01=CD5678\n'+q,' AB1234')=='SOURCE_NOT_VISIBLE'
    assert retrieval_contract('K00=XY9999\n'+q,' AB1234')=='VISIBLE_SOURCE_TARGET_MISMATCH'
    assert retrieval_contract('K00=AB1234\nK00=AB1234\n'+q,' AB1234')=='SOURCE_NOT_UNIQUE'


def test_partial_python_is_not_a_complete_code_target():
    assert python_contract(' def f():\n    """unfinished')=='INVALID_PYTHON_SYNTAX'
    assert python_contract(' def f():\n    return 7')=='PASS_SYNTAX_ONLY'


def test_quarantined_pilot_cannot_restart(tmp_path):
    require_semantic_clearance(tmp_path)
    (tmp_path/'DATA_SEMANTICS_STOP.json').write_text('{}')
    with pytest.raises(RuntimeError,match='quarantined'):require_semantic_clearance(tmp_path)


def test_actual_training_entrypoint_stops_before_loading_a_plan_or_gpu(tmp_path,monkeypatch):
    import run_astra_bounded_pilot as pilot
    (tmp_path/'DATA_SEMANTICS_STOP.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['pilot','--plan-dir',str(tmp_path),'--mode','train'])
    monkeypatch.setattr(pilot,'load_model',lambda *a:pytest.fail('model should not load'))
    with pytest.raises(RuntimeError,match='quarantined'):pilot.main()
