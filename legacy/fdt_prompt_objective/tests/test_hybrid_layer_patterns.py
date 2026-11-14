import pytest

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.next_causal_lm import resolve_anchor_layer_indices


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("alternate", [1, 3, 5]),
        ("last_half", [3, 4, 5]),
        ("middle", [2, 3]),
        ("all_but_first_last", [1, 2, 3, 4]),
        ("all", [0, 1, 2, 3, 4, 5]),
        ("none", []),
    ],
)
def test_named_patterns(pattern, expected):
    assert resolve_anchor_layer_indices(ModelConfig(n_layers=6, anchor_layer_pattern=pattern)) == expected


def test_explicit_indices_override_pattern():
    cfg = ModelConfig(n_layers=6, anchor_layer_pattern="none", anchor_layer_indices=[5, 1, 3, 3])
    assert resolve_anchor_layer_indices(cfg) == [1, 3, 5]


def test_out_of_range_explicit_index_is_rejected():
    with pytest.raises(ValueError):
        resolve_anchor_layer_indices(ModelConfig(n_layers=2, anchor_layer_indices=[2]))
