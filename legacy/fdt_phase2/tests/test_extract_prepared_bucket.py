import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extract_prepared_bucket import extract_bucket


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_extract_bucket_hard_links_unmixed_shards(tmp_path):
    source = tmp_path / "source"
    shards = []
    for split in ("train", "validation", "test"):
        directory = source / "shards" / split
        directory.mkdir(parents=True)
        path = directory / "shard_000000.pt"
        path.write_bytes(f"{split}-payload".encode("ascii"))
        shards.append(
            {
                "path": str(path),
                "split": split,
                "tokens": 2048,
                "sequences": 1,
                "buckets": {"cosmopedia": 2048},
                "sha256": sha256(path),
            }
        )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer": "tokenizers/test",
                "tokenizer_vocab_size": 24576,
                "seq_len": 2048,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    result = extract_bucket(source, output, "cosmopedia")

    assert result["packed_tokens"] == 6144
    assert result["split_tokens"] == {
        "train": 2048,
        "validation": 2048,
        "test": 2048,
    }
    for split in ("train", "validation", "test"):
        original = source / "shards" / split / "shard_000000.pt"
        linked = output / "shards" / split / "shard_000000.pt"
        assert os.path.samefile(original, linked)


def test_extract_bucket_rejects_mixed_shard(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer": "tokenizers/test",
                "tokenizer_vocab_size": 24576,
                "seq_len": 2048,
                "shards": [
                    {
                        "path": "mixed.pt",
                        "split": "train",
                        "tokens": 4096,
                        "buckets": {"cosmopedia": 2048, "fineweb": 2048},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        extract_bucket(source, tmp_path / "output", "cosmopedia")
    except ValueError as exc:
        assert "mixed shard" in str(exc)
    else:
        raise AssertionError("Mixed shard was accepted")
