from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(r"<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM").resolve()
TARGETS = {
    "runs/fdt_v4_1_free_run_diverse_pilot_16m_20260901_r1/latest_recovery.pt": "16B9947CB8E5919D252AA96AD5750D932FE81F80773FEBF18B3C673F61F59229",
    "runs/fdt_v4_1_free_run_margin_pilot_8m_20260901_r1/latest_recovery.pt": "E9DD8D6C8942A002B4DE42702CBA71DEFFEEDFEFC3F66AFB035B7FB59EB95F4A",
    "runs/fdt_v4_1_free_run_recovery_pilot_2m_20260901_r1/latest_recovery.pt": "F1105AB8420066270616C3395A6BCE986CA4E8FB20A16F8400EE4F2A9C7B4E13",
    "runs/fdt_v4_1_424m_rope_stage_a_20260830_r2_pinned_validation_resume/latest_recovery.pt": "5C955325B50D4E0DB8AC6AF8E2F91EC41A249787547ED8E0418E0F364D5736DC",
    "runs/fdt_v4_1_424m_rope_stage_a_20260830_r1/latest_recovery.pt": "E1BAED951F17BA1343E83C47D2DEAE3A984900111AD5816DA881B267ACA2A6C5",
    "runs/fdt_v4_1_424m_transition_20260829_r2/latest_recovery.pt": "378DBD95CFE4461288B3FF26853BCE6D1908DA81D03CE730ACD692927D3C14D7",
    "runs/fdt_v4_main_424m_curriculum_speed_20260827_r8_fast_backend/latest_recovery.pt": "D91661A3F90E67D9E71ED69DBD920BA2A990B5EAFA18A977EF1155B0C7196F5C",
    "runs/fdt_v4_main_424m_curriculum_speed_20260827_r9_batch4/latest_recovery.pt": "C066C6526AA3DC2ABFA0BE5F5DCAF7323F0BBEDC8F22A0B6A6F1197BC5FAD670",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    validated: list[dict[str, object]] = []
    for relative, expected_sha in TARGETS.items():
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents:
            raise RuntimeError(f"target escaped the workspace: {target}")
        if target.name != "latest_recovery.pt" or not target.is_file():
            raise RuntimeError(f"invalid or missing target: {target}")
        model_only = target.with_name("latest.pt")
        if not model_only.is_file():
            raise RuntimeError(f"model-only counterpart is missing: {model_only}")
        actual_sha = sha256(target)
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch: {target}")
        validated.append(
            {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": actual_sha,
                "model_only_retained": str(model_only),
            }
        )

    for entry in validated:
        Path(str(entry["path"])).unlink()
    remaining = [entry["path"] for entry in validated if Path(str(entry["path"])).exists()]
    if remaining:
        raise RuntimeError(f"targets remain after removal: {remaining}")

    free_bytes = shutil.disk_usage(ROOT).free
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "removed": validated,
                "removed_bytes": sum(int(entry["bytes"]) for entry in validated),
                "free_bytes": free_bytes,
                "free_gib": free_bytes / (1024**3),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
