# FDT: conditional answer-meaning generation

Research snapshot through **2026-11-14**, including the language codec,
three-memory interface, conditional latent diffusion, recovery tests, and
successful and unsuccessful experiment records.

## Research contract

```text
Language X → frozen encoder E → input representation Zx
Zx + persistent fuzzy prior K_P + contextual fuzzy snapshot K_C
   + dormant/null Exact memory K_E + independent noise
    → learned 16-step conditional latent diffusion Rθ
    → committed answer representation → frozen language realizer D → text
```

Only Rθ was trained in the latest controlled trials. E, D, existing memory
formation/binding, and all codec buffers remained fixed. The target is E(Y),
not identity reconstruction E(X). Gold answers, answer length and answer
lookup are absent from source-only inference. Sixteen denoising steps are not
sixteen logical reasoning steps. The current D is a parallel byte/EOS realizer,
not the older V7/Diana autoregressive decoder.

## Current measured outcome

| Trial | Change | Actual updates | Outcome |
|---|---|---:|---|
| V111 | hidden520, answer endpoint + VP loss | 268 | Correct positive relation generation not established |
| V112 | Common noise for real reverse pairs + pair-difference loss | 268 | Stronger conditional latent response; language remains damaged |
| V113 | Fixed-D content-byte loss backpropagated to Rθ only | 268 | Partial TRAIN relation generation; DEV generalization unresolved |

V113 used **134 selected TRAIN rows**, 31 fact keys and 27 answer strings,
repeated across 16 finite epochs (2,144 row exposures, not independent tasks).
TRAIN byte accuracy increased from **76.84% to 96.65%**. Correct normal-grammar
positive relation cores were observed in **34/80** TRAIN outputs; manual reading
identified one additional unambiguous typo case, giving a **lower bound of 35/80
readable cores**. These are core-content observations, not certified complete-output
semantic accuracy: malformed suffixes remain, and positive whole-string exact
was **0/80**. Both members were correct for 8/32 positive reverse-question pairs.

Positive relation correctness on the observed DEV panels remains **unresolved
(0/48 verified correct cores)**. DEV NLL worsened. Novel fact/entity combinations,
surface wording, and position changes remain substantial limitations. This is
neither a general LLM nor proof of independent abstract reasoning, memory
causality, composition, or lossless semantics. Each trial used a fresh optimizer;
their loss curves must not be merged into one optimizer trajectory.

Latest trials ran on **CPU**, not GPU. All worker/owner processes were confirmed
terminated before independent saved-data analysis. Historical failures and
limitations are retained rather than hidden behind the latest result.

## Start here

- [Latest detailed results: seven reports](evidence/v113/README_RESULTS_KR.md)
- [Semantic evaluation](evidence/v113/05_SEMANTIC_EVALUATION_KR.md)
- [Next research decision](evidence/v113/07_NEXT_RESEARCH_DECISION_KR.md)
- [V112 analysis](evidence/v112/RESULT_AND_NEXT_KR.md)
- [Architecture and scope](docs/ARCHITECTURE_KR.md)
- [Local asset and reproducibility boundaries](docs/LOCAL_ASSETS.md)
- [Snapshot provenance](provenance/export_manifest.json)

## Lightweight verification

Python 3.10+ and the dependencies in `pyproject.toml` are required. Example in a
fresh environment (choose the appropriate PyTorch build for your platform):

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e . pytest
.venv/Scripts/python -B -m pytest -q -p no:cacheprovider tests/test_latent_identity_v110.py tests/test_answer_meaning_v111.py tests/test_answer_pair_contrast_v112.py tests/test_answer_codec_fidelity_v113.py
```

The focused suite uses tiny neural fixtures and the included authored synthetic
metadata. It is not a new full-size training/evaluation run. Most historical
workers require local binary checkpoints and caches omitted from this snapshot.
**Do not restart closed run profiles or treat old plans as current authorization.**
Historical `README_LOCAL_KR.md` and setup scripts describe earlier experiments;
this README and the dated V113 reports take precedence for current claims.

## Contents and exclusions

`src/`, `research/`, `scripts/`, `tests/`, `configs/`, `archive/`, and `legacy/`
preserve the project implementation and earlier source branches. `docs/history/`
retains dated research reports; `evidence/v111` through `v113` retain text results
and historical plans. The private repository has no new open-source license grant.

Weights, optimizer/scaler/RNG binary states, raw tensor traces, virtual environments,
credentials, third-party corpora, final holdouts and evaluation registries are
**not uploaded**. They remain unchanged locally. Absolute workstation/user paths
and the owner's private email are normalized in exported text. Original-file
and exported-file SHA256 values are both recorded; old sealed plan hashes refer
to the originals, not transformed GitHub copies. This is a research source-and-text
evidence snapshot, not a complete off-machine recovery backup.
