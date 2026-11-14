from pathlib import Path
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_query_bound_alignment_20261027_v2 as worker


def test_actual_pinned_rows_build_initial_manifest_without_gpu_or_optimizer():
    folder = ROOT / 'research/artifacts/role8_query_mix_20261027_v1'
    train = json.loads((folder / 'train.json').read_text(encoding='utf-8'))
    dev = json.loads((folder / 'dev.json').read_text(encoding='utf-8'))
    result = worker.initial_stage_manifest(train, dev)
    assert result['splits']['train']['graphs'] == 1024
    assert result['splits']['dev']['graphs'] == 64
    assert result['splits']['train']['query_labels'] == [4667, 3820, 3801, 4096]
    assert result['splits']['dev']['query_labels'] == [322, 207, 239, 256]
    assert abs(sum(result['splits']['train']['class_ratio']) - 1) < 1e-12


def test_empty_placeholder_is_rejected_before_manifest_division():
    with pytest.raises(ValueError, match='actual nonempty'):
        worker.initial_stage_manifest([], [])
