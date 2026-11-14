import torch

from fdt_rlm.data.interleaving import (
    bucket_sequence_counts,
    interleave_remaining_indices,
    source_counts_for_prefix,
)


def test_interleave_remaining_is_deterministic_unique_and_no_replay():
    counts = {"edu": 6, "web": 3, "code": 3}
    first = interleave_remaining_indices(counts, consumed_prefix=4)
    second = interleave_remaining_indices(counts, consumed_prefix=4)

    assert torch.equal(first, second)
    assert first.numel() == 8
    assert first.unique().numel() == first.numel()
    assert int(first.min()) >= 4
    assert set(first.tolist()) == set(range(4, 12))


def test_source_mix_is_balanced_in_early_prefixes():
    counts = {"edu": 60, "web": 15, "code": 25}
    order = interleave_remaining_indices(counts)

    for prefix in (20, 40, 100):
        actual = source_counts_for_prefix(order, counts, prefix)
        for source, total in counts.items():
            expected = prefix * total / sum(counts.values())
            assert abs(actual[source] - expected) <= 1


def test_shared_prefix_produces_identical_recovery_stream_for_both_models():
    counts = {"edu": 60, "web": 15, "code": 25}
    transformer_cursor = 44
    fdt_cursor = 37
    shared_prefix = max(transformer_cursor, fdt_cursor)

    transformer_order = interleave_remaining_indices(counts, shared_prefix)
    fdt_order = interleave_remaining_indices(counts, shared_prefix)

    assert torch.equal(transformer_order, fdt_order)
    assert int(transformer_order.min()) >= transformer_cursor
    assert int(fdt_order.min()) >= fdt_cursor


def test_manifest_bucket_sequence_counts():
    manifest = {
        "seq_len": 4,
        "shards": [
            {
                "split": "train",
                "sequences": 3,
                "buckets": {"edu": 8, "code": 4},
            },
            {
                "split": "validation",
                "sequences": 1,
                "buckets": {"edu": 4},
            },
        ],
    }
    assert bucket_sequence_counts(manifest, bucket_order=("edu", "code")) == {
        "edu": 2,
        "code": 1,
    }
