from __future__ import annotations

import prepare_fdt_v21_knowledge_scale as base


# The globally disjoint pool now contains fewer long HotpotQA answers than the
# historical V21 quota. Preserve 20,000 factual rows while keeping a 5% long
# answer stratum and reallocating the remaining quota to short/medium answers.
base.FACTUAL_LENGTH_QUOTAS = {
    "short_1_4": 14_500,
    "medium_5_7": 4_500,
    "long_8_plus": 1_000,
}


if __name__ == "__main__":
    if sum(base.FACTUAL_LENGTH_QUOTAS.values()) != base.ROWS_PER_CATEGORY:
        raise RuntimeError("FDT v4.1 factual quota must preserve 20,000 rows")
    base.main()
