from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.next_tools import safe_torch_load  # noqa: E402
from prepare_capability_completion_v15 import require_c_path  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one predeclared model-only checkpoint interpolation.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    parent_path = require_c_path(args.parent)
    candidate_path = require_c_path(args.candidate)
    output_dir = require_c_path(args.output_dir)
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_hash = sha256(parent_path)
    candidate_hash = sha256(candidate_path)
    if parent_hash.lower() != args.expected_parent_sha256.lower():
        raise ValueError("parent checkpoint hash mismatch")
    if candidate_hash.lower() != args.expected_candidate_sha256.lower():
        raise ValueError("candidate checkpoint hash mismatch")

    parent = load_mmap(parent_path)
    candidate = load_mmap(candidate_path)
    if parent["model_config"] != candidate["model_config"]:
        raise ValueError("checkpoint model configs differ")
    parent_state = parent["model_state_dict"]
    candidate_state = candidate["model_state_dict"]
    if parent_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint model-state keys differ")

    model_state = {}
    for name, parent_tensor in parent_state.items():
        candidate_tensor = candidate_state[name]
        if parent_tensor.shape != candidate_tensor.shape or parent_tensor.dtype != candidate_tensor.dtype:
            raise ValueError(f"checkpoint tensor mismatch: {name}")
        if parent_tensor.is_floating_point():
            value = parent_tensor.clone()
            value.lerp_(candidate_tensor, args.alpha)
        else:
            if not torch.equal(parent_tensor, candidate_tensor):
                raise ValueError(f"non-floating checkpoint tensor differs: {name}")
            value = parent_tensor.clone()
        model_state[name] = value

    train_config = dict(parent.get("train_config", {}))
    train_config.update(
        {
            "stage": args.stage,
            "interpolation_alpha": args.alpha,
            "interpolation_parent": str(parent_path.resolve()),
            "interpolation_parent_sha256": parent_hash,
            "interpolation_candidate": str(candidate_path.resolve()),
            "interpolation_candidate_sha256": candidate_hash,
            "performs_training": False,
        }
    )
    candidate_step = int(candidate.get("global_step", candidate.get("optimizer_step", 0)))
    payload = {
        "model_state_dict": model_state,
        "model_config": parent["model_config"],
        "train_config": train_config,
        "global_step": candidate_step,
        "optimizer_step": int(candidate.get("optimizer_step", candidate_step)),
        "sample_cursor": int(candidate.get("sample_cursor", 0)),
        "tokens_seen": int(candidate.get("tokens_seen", parent.get("tokens_seen", 0))),
        "stage": args.stage,
        "stage_status": "INTERPOLATED_DIAGNOSTIC_CANDIDATE",
        "optimizer_state_saved": False,
        "interpolation": {
            "alpha": args.alpha,
            "parent": str(parent_path.resolve()),
            "parent_sha256": parent_hash,
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": candidate_hash,
        },
    }
    output = output_dir / "latest.pt"
    temporary = output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    report = {
        "status": "INTERPOLATION_CREATED",
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "alpha": args.alpha,
        "parent_sha256": parent_hash,
        "candidate_sha256": candidate_hash,
        "optimizer_state_saved": False,
    }
    report_path = output_dir / "interpolation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
