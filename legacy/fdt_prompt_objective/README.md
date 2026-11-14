# FDT-RLM Research Branch

This directory is the new English/code FDT-RLM research branch.

The Korean FDT2 model is preserved only as legacy history. New checkpoints, tokenizers, datasets, and model configs are intentionally separate.

## Current Stage

Completed:

- FDT2 audit.
- Legacy copy.
- Migration notes.
- Minimal causal tests for current FDT2.
- New causal Transformer baseline.
- New causal FDT model with prefix-safe sparse anchor routing.
- English/code tokenizer and packed text data path.
- Micro configs for matched FDT and Transformer experiments.

Next:

- Add a validation split and validation loss.
- Add tokenizer training for a project-owned byte-level BPE tokenizer.
- Add uncertainty calibration and confidence heads.
- Add RLM runtime only after base models pass causal tests.

## Why Separate From FDT2

The master prompt proposes an English/code FDT-RLM system with a new tokenizer and recursive runtime. That is not compatible with current Korean FDT2 checkpoints.

Keeping them separate protects the useful Korean experiment while allowing the larger research program to move forward honestly.

## Test Command

From the repository root:

```powershell
python -m pytest outputs/FDT_RLM/tests
```

## Bounded Fuzzy Entropy Repair

FDT2's capped entropy reward is restored and made layer-aware. Early layers
target normalized entropy 0.65, while later layers decay to 0.45 so ambiguity
can be retained first and resolved with context. Entropy above the target is
penalized instead of being rewarded toward uniform Top-K routing.

Continue only the existing FDT checkpoint for a short 500-step repair stage:

```powershell
python scripts/run_first_experiment.py --phase full --models fdt --resume --additional_optimizer_steps 500 --noise_weight 0 --no-save_generations --eval_every 100 --save_every 100 --log_every 20
```

The runner preserves the old latest checkpoint as
`pre_entropy_repair_step_<step>.pt` before writing repaired checkpoints.
Watch `entropy_normalized`, `entropy_target`, `entropy_over`,
`effective_anchors`, `mean_top1_membership`, and validation loss. Do not scale
the model until learned membership beats the uniform-Top-K counterfactual.

## Smoke Training

From the repository root:

```powershell
python outputs/FDT_RLM/scripts/train_base.py --text_file outputs/FDT_RLM/data/smoke_english_code.txt --model_type fdt --max_steps 10 --batch_size 2 --grad_accum_steps 1 --output_dir outputs/FDT_RLM/runs/smoke_fdt
```

Transformer baseline smoke:

```powershell
python outputs/FDT_RLM/scripts/train_base.py --text_file outputs/FDT_RLM/data/smoke_english_code.txt --model_type transformer --max_steps 10 --batch_size 2 --grad_accum_steps 1 --output_dir outputs/FDT_RLM/runs/smoke_transformer
```

## Start Here: From Fresh Environment to Training

### 1. Install

```powershell
cd <USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM
pip install -r requirements.txt
```

### 2. Hugging Face Auth

Optional for the current 10M smoke corpus, required for some gated BigCode datasets:

```powershell
hf auth login
```

Never paste a token into source files or configs.

### 3. Probe / Resolve Sources

```powershell
python scripts/probe_datasets.py --config configs/datasets.yaml
python scripts/resolve_code_source.py
```

Current selected code source: `bigcode/the-stack-smol-xl`.

### 4. Dataset Plan

```powershell
python scripts/plan_dataset.py --total-tokens 10000000
```

### 5. Tokenizer

Already created at `tokenizers/fdt_bpe`.

Recreate:

```powershell
python scripts/prepare_tokenizer_corpus.py --config configs/datasets.yaml --target-tokens 1000000 --output artifacts/tokenizer_corpus_1m.txt
python scripts/train_tokenizer.py --input artifacts/tokenizer_corpus_1m.txt --output-dir tokenizers/fdt_bpe --vocab-size 49152
python scripts/validate_tokenizer.py --tokenizer tokenizers/fdt_bpe
```

### 6. Prepare / Validate 10M Corpus

Already created at `prepared_data/fdt_rlm_10m`.

Recreate:

```powershell
python scripts/prepare_dataset.py --config configs/datasets.yaml --total-tokens 10000000 --output-dir prepared_data/fdt_rlm_10m --tokenizer-name tokenizers/fdt_bpe
python scripts/validate_dataset.py --dataset-dir prepared_data/fdt_rlm_10m
```

### 7. Tests

```powershell
pytest tests -q
```

### 8. Readiness

```powershell
python scripts/check_training_readiness.py --dataset-dir prepared_data/fdt_rlm_10m --tokenizer tokenizers/fdt_bpe
```

Expected:

```text
TRAINING READINESS: PASS
```

### 9. Start Transformer Baseline

```powershell
python scripts/train_base.py --dataset_dir prepared_data/fdt_rlm_10m --tokenizer_name tokenizers/fdt_bpe --model_type transformer --max_steps 1000 --batch_size 1 --grad_accum_steps 8 --dim 384 --n_layers 6 --n_heads 6 --max_seq_len 512 --output_dir runs/micro_transformer
```

### 10. Start FDT

```powershell
python scripts/train_base.py --dataset_dir prepared_data/fdt_rlm_10m --tokenizer_name tokenizers/fdt_bpe --model_type fdt --max_steps 1000 --batch_size 1 --grad_accum_steps 8 --dim 384 --n_layers 6 --n_heads 6 --max_seq_len 512 --num_anchors 128 --top_k 8 --output_dir runs/micro_fdt
```

## Dataset Pipeline

Default mixture:

- FineWeb-Edu: 60%
- FineWeb: 15%
- code/structured: 25%

Commands:

```powershell
python outputs/FDT_RLM/scripts/probe_datasets.py --config outputs/FDT_RLM/configs/datasets.yaml
python outputs/FDT_RLM/scripts/plan_dataset.py --total-tokens 10000000
python outputs/FDT_RLM/scripts/prepare_tokenizer_corpus.py --target-tokens 10000000
python outputs/FDT_RLM/scripts/train_tokenizer.py
python outputs/FDT_RLM/scripts/prepare_dataset.py --config outputs/FDT_RLM/configs/datasets.yaml --total-tokens 10000000 --output-dir outputs/FDT_RLM/prepared_data/fdt_rlm_10m
python outputs/FDT_RLM/scripts/validate_dataset.py --dataset-dir outputs/FDT_RLM/prepared_data/fdt_rlm_10m
```

Do not start base training from prepared shards until validation prints `PASS`.
