from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_base_model_strict as v2  # noqa: E402
import evaluate_base_model_strict_v3 as v3  # noqa: E402
from fdt_rlm.next_tools import safe_torch_load  # noqa: E402
from prepare_capability_completion_v15 import require_c_path  # noqa: E402
from train_capability_completion_v20 import factual_short_eos_metrics  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mmap(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return safe_torch_load(path, "cpu")


@torch.no_grad()
def set_interpolation(model, parent_state: dict, candidate_state: dict, alpha: float) -> None:
    target_state = model.state_dict()
    if target_state.keys() != parent_state.keys() or target_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint model-state keys differ")
    for name, target in target_state.items():
        parent = parent_state[name]
        candidate = candidate_state[name]
        if parent.shape != candidate.shape or parent.dtype != candidate.dtype:
            raise ValueError(f"checkpoint tensor mismatch: {name}")
        if target.is_floating_point():
            target.copy_(parent.to(device=target.device, dtype=target.dtype))
            target.lerp_(candidate.to(device=target.device, dtype=target.dtype), alpha)
        else:
            if not torch.equal(parent, candidate):
                raise ValueError(f"non-floating checkpoint tensor differs: {name}")
            target.copy_(parent.to(device=target.device))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only fixed-grid FDT checkpoint interpolation diagnostic.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    parent_path = require_c_path(args.parent)
    candidate_path = require_c_path(args.candidate)
    output = require_c_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    alphas = [float(value) for value in args.alphas]
    if sorted(set(alphas)) != alphas or any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("alphas must be unique, increasing, and strictly between zero and one")

    device = torch.device(args.device)
    model, tokenizer, config, parent_metadata = v2.load_checkpoint_model(parent_path, device)
    parent_checkpoint = load_mmap(parent_path)
    candidate_checkpoint = load_mmap(candidate_path)
    parent_state = parent_checkpoint["model_state_dict"]
    candidate_state = candidate_checkpoint["model_state_dict"]
    if parent_checkpoint["model_config"] != candidate_checkpoint["model_config"]:
        raise ValueError("checkpoint model configs differ")

    broad_dir = ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048"
    capability_dir = ROOT / "prepared_data" / "capability_curriculum_5k"
    broad_manifest = json.loads((broad_dir / "manifest.json").read_text(encoding="utf-8"))
    capability_manifest = json.loads((capability_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ids, broad_masks = v2.split_rows(broad_dir, "validation")
    broad_ranges = v2.bucket_ranges(broad_manifest, "validation")
    payload = v3.load_capability_payload(capability_dir)
    ranges = v3.capability_ranges(capability_manifest)
    python_range = [entry for entry in ranges if entry[0] == "python_code"]
    factual_range = next((start, end) for name, start, end in ranges if name == "factual_qa")

    results = []
    for alpha in alphas:
        set_interpolation(model, parent_state, candidate_state, alpha)
        model.eval()
        heldout = v2.heldout_metrics(model, config, broad_ids, broad_masks, broad_ranges, 4, device)["overall"]
        python = v3.evaluate_capability_generation(
            model, tokenizer, config, payload, python_range, {"python_code": 25}, device
        )["python_code"]
        prefix = v3.evaluate_heldout_generation(
            model, tokenizer, config, broad_ids, broad_masks, broad_ranges, 10, device
        )["overall"]
        factual = factual_short_eos_metrics(
            model,
            payload["input_ids"],
            payload["labels"],
            payload["attention_mask"],
            factual_range,
            4,
            device,
            config.eos_token_id,
        )
        results.append(
            {
                "alpha": alpha,
                "heldout": heldout,
                "python": {key: value for key, value in python.items() if key != "rows"},
                "prefix": prefix,
                "factual_short_eos": factual,
            }
        )
        print(json.dumps({"event": "alpha_complete", "alpha": alpha, "python_structural": python["usable_structure"]["successes"], "loop_free": prefix["loop_free"]["successes"]}), flush=True)

    report = {
        "status": "DIAGNOSTIC_COMPLETED",
        "performs_training": False,
        "selection_role": "development_only_fixed_grid",
        "parent": {"path": str(parent_path.resolve()), "sha256": sha256(parent_path), **parent_metadata},
        "candidate": {"path": str(candidate_path.resolve()), "sha256": sha256(candidate_path)},
        "alphas": alphas,
        "results": results,
        "limitations": [
            "The fixed capability and prefix rows are repeatedly consulted development diagnostics.",
            "Python structural usability is not semantic correctness.",
            "Any selected interpolation requires a fresh audited standard evaluation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
