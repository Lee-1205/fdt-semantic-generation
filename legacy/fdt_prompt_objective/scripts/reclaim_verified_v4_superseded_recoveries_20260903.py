from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(r"<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM").resolve()
TARGETS = {
    "runs/fdt_v4_main_424m_curriculum_speed_20260827_r3/latest_recovery.pt": "B6B3734E5CF929B09F7F1975F4B136EC390C98847DAB4C27EC84C7B70EE90471",
    "runs/fdt_v4_main_424m_curriculum_speed_20260827_r4/latest_recovery.pt": "1F20C75E4B5F2A3B14CC6D707A95C17C49A58E1FEC3DC3138002E33381B456FE",
    "runs/fdt_v4_main_424m_curriculum_speed_20260827_r5/latest_recovery.pt": "323E9C9E9B5CC3FB5C79197F5C06CE9825CB8C9711921E130FE971B0E4577731",
    "runs/fdt_v4_main_424m_curriculum_speed_20260827_r6_fixed_validation/latest_recovery.pt": "A04344A0E217C807DE8A9307639BC3088268D06F81F5C4E5A113A5D0D4A8B561",
    "runs/fdt_v4_main_424m_curriculum_speed_20260827_r7_syncfix/latest_recovery.pt": "9C9BDD50AB7C083E3E9808F99988495E9BDAEC29647A78662ABFB9EF0DE66232",
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
        if ROOT not in target.parents or target.name != "latest_recovery.pt":
            raise RuntimeError(f"invalid target: {target}")
        model_only = target.with_name("latest.pt")
        if not target.is_file() or not model_only.is_file():
            raise RuntimeError(f"recovery or model-only counterpart missing: {target}")
        actual_sha = sha256(target)
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch: {target}")
        validated.append({
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": actual_sha,
            "model_only_retained": str(model_only),
        })

    for entry in validated:
        Path(str(entry["path"])).unlink()
    if any(Path(str(entry["path"])).exists() for entry in validated):
        raise RuntimeError("one or more verified targets remain")

    free_bytes = shutil.disk_usage(ROOT).free
    print(json.dumps({
        "status": "COMPLETE",
        "removed": validated,
        "removed_bytes": sum(int(entry["bytes"]) for entry in validated),
        "free_gib": free_bytes / (1024**3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
