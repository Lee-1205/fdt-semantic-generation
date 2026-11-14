from scripts.prepare_balanced_novel_pilot import (
    MIX_PROFILES,
    NEAR_DEDUP_EXEMPT_BUCKETS,
    NearDeduper,
    build_filler_to_token_distance,
    build_wikipedia_short_fact,
    compact_json_documents,
    json_documents,
    long_retrieval_documents,
    prompted_retrieval_documents,
    retrieval_documents,
    repeated_ngram_fraction,
    simhash64,
    split_long_document,
)


def test_synthetic_documents_are_deterministic_and_structurally_valid():
    import json

    first = next(json_documents(1234))[0]
    second = next(json_documents(1234))[0]
    assert first == second
    payload = json.loads(first)
    assert payload["records"]
    assert payload["batch_id"].startswith("batch-")

    retrieval = next(retrieval_documents(5678))[0]
    lines = retrieval.splitlines()
    target = lines[-1].split()[-1].rstrip(".")
    assert target in retrieval
    assert "Recorded key-value entries:" in retrieval


def test_near_deduper_rejects_small_edit_but_keeps_distinct_text():
    deduper = NearDeduper(max_hamming=3)
    base = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    assert deduper.add_if_novel(base)
    assert not deduper.add_if_novel(base + " ")
    assert deduper.add_if_novel("python function returns a sorted collection of unique records")
    assert simhash64(base) == simhash64(base)


def test_split_long_document_preserves_tokens():
    tokens = list(range(25))
    parts = list(split_long_document(tokens, 8))
    assert [len(part) for part in parts] == [8, 8, 8, 1]
    assert [token for part in parts for token in part] == tokens


def test_v2_mix_and_compact_json_match_the_capability_shape():
    import json

    assert abs(sum(MIX_PROFILES["retention_capability_v2"].values()) - 1.0) < 1e-9
    assert abs(sum(MIX_PROFILES["retention_capability_v2_scale"].values()) - 1.0) < 1e-9
    assert abs(sum(MIX_PROFILES["retention_diversity_v3_pilot"].values()) - 1.0) < 1e-9
    assert abs(sum(MIX_PROFILES["retention_alignment_v4_pilot"].values()) - 1.0) < 1e-9
    assert abs(sum(MIX_PROFILES["code_growth_v5_pilot"].values()) - 1.0) < 1e-9
    assert abs(sum(MIX_PROFILES["general_growth_v6_scale"].values()) - 1.0) < 1e-9
    assert sum(
        MIX_PROFILES["general_growth_v6_scale"][bucket]
        for bucket in ("fineweb_edu", "wikipedia", "code_python")
    ) == 0.80
    assert abs(sum(MIX_PROFILES["retrieval_repetition_repair_v7"].values()) - 1.0) < 1e-9
    assert MIX_PROFILES["retrieval_repetition_repair_v7"]["retrieval_prompted"] == 0.10
    assert MIX_PROFILES["retrieval_repetition_repair_v7"]["retrieval_long"] == 0.10
    assert abs(sum(MIX_PROFILES["factual_retrieval_balance_v9"].values()) - 1.0) < 1e-9
    assert MIX_PROFILES["factual_retrieval_balance_v9"]["factual_cloze"] == 0.30
    assert (
        MIX_PROFILES["factual_retrieval_balance_v9"]["retrieval_prompted"]
        + MIX_PROFILES["factual_retrieval_balance_v9"]["retrieval_long"]
        == 0.15
    )
    assert abs(sum(MIX_PROFILES["natural_retrieval_balance_v10"].values()) - 1.0) < 1e-9
    assert sum(
        MIX_PROFILES["natural_retrieval_balance_v10"][bucket]
        for bucket in ("fineweb_edu_dedup", "cosmopedia", "tinystories", "wikipedia")
    ) == 0.70
    assert "factual_cloze" not in MIX_PROFILES["natural_retrieval_balance_v10"]
    assert abs(sum(MIX_PROFILES["short_fact_repair_v12"].values()) - 1.0) < 1e-9
    assert MIX_PROFILES["short_fact_repair_v12"]["wikipedia_short_fact"] == 0.25
    assert "factual_cloze" not in MIX_PROFILES["short_fact_repair_v12"]
    assert abs(sum(MIX_PROFILES["long_horizon_general_v13"].values()) - 1.0) < 1e-9
    assert abs(sum(
        MIX_PROFILES["long_horizon_general_v13"][bucket]
        for bucket in ("fineweb_edu_dedup", "cosmopedia", "tinystories", "wikipedia")
    ) - 0.60) < 1e-9
    assert MIX_PROFILES["long_horizon_general_v13"]["code_python"] == 0.20
    assert "factual_cloze" not in MIX_PROFILES["long_horizon_general_v13"]
    assert "wikipedia_short_fact" not in MIX_PROFILES["long_horizon_general_v13"]
    assert abs(sum(MIX_PROFILES["capability_retention_v14"].values()) - 1.0) < 1e-9
    assert abs(sum(
        MIX_PROFILES["capability_retention_v14"][bucket]
        for bucket in ("fineweb_edu_dedup", "cosmopedia", "tinystories", "wikipedia")
    ) - 0.45) < 1e-9
    assert MIX_PROFILES["capability_retention_v14"]["factual_cloze"] == 0.05
    assert MIX_PROFILES["capability_retention_v14"]["code_python"] == 0.158
    assert MIX_PROFILES["capability_retention_v14"]["code_mbpp"] == 0.002
    assert MIX_PROFILES["capability_retention_v14"]["retrieval_prompted"] == 0.10
    assert MIX_PROFILES["capability_retention_v14"]["retrieval_long"] == 0.20
    assert {"json_compact", "retrieval_prompted", "retrieval_long"} <= NEAR_DEDUP_EXEMPT_BUCKETS
    document, metadata = next(compact_json_documents(20260810))
    prompt, payload = document.split("JSON: ", 1)
    record = json.loads(payload)
    assert prompt.startswith("Convert the record")
    assert metadata["schema"] == "compact_record_v1"
    assert record["record_id"].startswith("R20260810-")
    assert "\n" not in payload
    assert '"record_id":' in payload


def test_repeated_ngram_fraction_detects_boilerplate_without_penalizing_short_text():
    repetitive = " ".join(["alpha beta gamma"] * 40)
    diverse = " ".join(f"token_{index}" for index in range(120))
    repetitive_fraction, repetitive_count = repeated_ngram_fraction(repetitive)
    diverse_fraction, diverse_count = repeated_ngram_fraction(diverse)
    assert repetitive_count == diverse_count == 118
    assert repetitive_fraction > 0.90
    assert diverse_fraction == 0.0
    assert repeated_ngram_fraction("alpha beta") == (0.0, 0)


def test_wikipedia_short_fact_masks_the_short_answer_and_rejects_long_titles():
    text = (
        "Ada Lovelace was an English mathematician and writer known for her work on "
        "Charles Babbage's proposed mechanical general-purpose computer. She was the "
        "first to recognize that the machine had applications beyond pure calculation."
    )
    document = build_wikipedia_short_fact("Ada Lovelace", text)
    assert document is not None
    assert document.endswith("Answer: Ada Lovelace")
    assert "Ada Lovelace" not in document.split("Answer:", 1)[0]
    assert "This subject" in document
    assert build_wikipedia_short_fact("An Excessively Long Multi Word Article Title", text) is None


def test_long_retrieval_v2_prioritizes_distance_1400():
    class WhitespaceTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return text.split()

    generator = long_retrieval_documents(WhitespaceTokenizer(), 123)
    rows = [next(generator) for _ in range(8)]
    assert [metadata["distance_class"] for _, metadata in rows] == [
        64,
        512,
        1400,
        1400,
        64,
        512,
        1400,
        1400,
    ]
    for document, metadata in rows:
        assert metadata["id"].startswith("long-retrieval-")
        assert document.count(metadata["code"]) == 2
        assert document.endswith(metadata["code"] + ".")


def test_batched_retrieval_filler_reaches_minimum_distance():
    import random

    class WhitespaceTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return text.split()

    filler = build_filler_to_token_distance(
        WhitespaceTokenizer(), ["alpha", "beta", "gamma"], random.Random(17), 1400
    )
    assert len(WhitespaceTokenizer.encode(" ".join(filler))) == 1400


def test_prompted_retrieval_is_deterministic_and_ends_with_the_target():
    first_document, first_metadata = next(prompted_retrieval_documents(20260810))
    second_document, second_metadata = next(prompted_retrieval_documents(20260810))
    assert first_document == second_document
    assert first_metadata == second_metadata
    assert first_document.endswith(first_metadata["code"])
    assert first_document.count(first_metadata["code"]) == 2
    assert first_metadata["id"].startswith("prompted-retrieval-")
