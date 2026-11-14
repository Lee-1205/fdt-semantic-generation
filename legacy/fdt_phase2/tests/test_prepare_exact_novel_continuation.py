import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "prepare_exact_novel_continuation.py"
    spec = importlib.util.spec_from_file_location("prepare_exact_novel_continuation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_document_hash_is_dtype_stable():
    module = load_module()
    assert module.document_hash(torch.tensor([1, 2, 3], dtype=torch.int64)) == module.document_hash(
        torch.tensor([1, 2, 3], dtype=torch.int32)
    )
    assert module.document_hash(torch.tensor([1, 2, 3])) != module.document_hash(
        torch.tensor([3, 2, 1])
    )


def test_compact_writer_discards_bucket_tail_and_uses_compact_dtypes(tmp_path):
    module = load_module()
    output = tmp_path / "dataset"
    writer = module.CompactShardWriter(output, seq_len=4, shard_tokens=8)
    writer.add_document("alpha", torch.tensor([10, 11]), eos_token_id=2)
    writer.add_document("alpha", torch.tensor([12, 13]), eos_token_id=2)
    writer.add_document("beta", torch.tensor([20, 21, 22]), eos_token_id=2)
    shards, discarded = writer.close()

    assert len(shards) == 1
    assert discarded == 0
    payload = torch.load(output / "shards" / "train" / "shard_000000.pt", weights_only=True)
    assert payload["input_ids"].dtype == torch.int32
    assert payload["attention_mask"].dtype == torch.bool
    assert shards[0]["buckets"] == {"alpha": 4, "beta": 4}


def test_prefix_hashes_include_document_crossing_consumed_boundary(tmp_path):
    module = load_module()
    dataset = tmp_path / "source"
    shard_dir = dataset / "shards" / "train"
    shard_dir.mkdir(parents=True)
    ids = torch.tensor([[10, 11, 12, 13], [14, 2, 20, 2]], dtype=torch.int32)
    torch.save({"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}, shard_dir / "shard_000000.pt")
    (dataset / "manifest.json").write_text(
        '{"seq_len":4,"bucket_order":["cosmopedia"],"plan":{"cosmopedia":8},'
        '"shards":[{"path":"shards/train/shard_000000.pt","split":"train",'
        '"tokens":8,"sequences":2,"buckets":{"cosmopedia":8}}]}',
        encoding="utf-8",
    )

    hashes, extended_tokens = module.collect_prefix_hashes(
        dataset, "cosmopedia", consumed_rows=1, eos_token_id=2, minimum_tokens=1
    )
    assert extended_tokens == 6
    assert hashes == {module.document_hash(torch.tensor([10, 11, 12, 13, 14]))}


def test_single_bucket_manifest_can_omit_per_shard_bucket_counts(tmp_path):
    module = load_module()
    dataset = tmp_path / "single_bucket"
    shard_dir = dataset / "shards" / "train"
    shard_dir.mkdir(parents=True)
    ids = torch.tensor([[10, 11, 2, 12], [13, 2, 14, 2]], dtype=torch.int32)
    torch.save(
        {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)},
        shard_dir / "shard_000000.pt",
    )
    (dataset / "manifest.json").write_text(
        '{"seq_len":4,"bucket_order":["only"],"plan":{"only":8},'
        '"shards":[{"path":"shards/train/shard_000000.pt","split":"train",'
        '"tokens":8,"sequences":2}]}',
        encoding="utf-8",
    )

    chunks = list(module.iter_bucket_token_chunks(dataset))
    assert len(chunks) == 1
    assert chunks[0][0] == "only"
    assert chunks[0][1].tolist() == ids.reshape(-1).tolist()
