import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_astra_clean_pilot import remaining_limits

def test_prior_failed_attempts_reduce_both_matched_arms_without_new_authorization():
    stop={'reserved_supervised_all_attempts':15534,'B':'NOT_STARTED',
          'charged_execution_seconds_with_shutdown_allowances':1436.937567949295}
    tokens,seconds=remaining_limits(stop)
    assert tokens==234466 and seconds==pytest.approx(12963.062432050705)
    assert 15534+2*tokens<=500000
    with pytest.raises(ValueError):remaining_limits({**stop,'B':'STARTED'})
    with pytest.raises(ValueError):remaining_limits({**stop,'charged_execution_seconds_with_shutdown_allowances':14400})

def test_execution_segments_charge_previous_runs_and_fail_closed_on_crash(tmp_path,monkeypatch):
    import run_astra_clean_pilot as r
    from astra_evidence_io import save
    (tmp_path/'budget_accounting.json').write_text('{}')
    folder=tmp_path/'execution';folder.mkdir()
    budget={'remaining_gpu_seconds':1000}
    monkeypatch.setattr(r.time,'time',lambda:100.)
    segment,start,deadline=r.start_segment(folder,budget)
    assert deadline==1040.
    with pytest.raises(RuntimeError,match='unclosed'):r.start_segment(folder,budget)
    save(segment.with_suffix('.end.json'),{'charged_seconds':160})
    _,_,deadline=r.start_segment(folder,budget)
    assert deadline==880.
