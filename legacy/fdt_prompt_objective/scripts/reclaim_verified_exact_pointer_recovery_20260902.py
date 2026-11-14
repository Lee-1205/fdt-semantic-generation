from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(r"<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM").resolve()
TARGET = (ROOT / "runs/fdt_v4_1_exact_pointer_pilot_20260831_r3/latest_recovery.pt").resolve()
EXPECTED_SHA256 = "58610A064AA6F4C181154073467C7FDB0ABCF0C9E89EE3110E3B40932551F782"


def main() -> int:
    if ROOT not in TARGET.parents or TARGET.name != "latest_recovery.pt":
        raise RuntimeError("target escaped the verified workspace contract")
    model_only = TARGET.with_name("latest.pt")
    if not TARGET.is_file() or not model_only.is_file():
        raise RuntimeError("recovery or retained model-only counterpart is missing")
    digest = hashlib.sha256()
    with TARGET.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest().upper()
    if actual != EXPECTED_SHA256:
        raise RuntimeError("recovery SHA-256 mismatch")
    removed_bytes = TARGET.stat().st_size
    TARGET.unlink()
    if TARGET.exists():
        raise RuntimeError("recovery checkpoint remains after removal")
    free_bytes = shutil.disk_usage(ROOT).free
    print(json.dumps({
        "status": "COMPLETE",
        "removed": str(TARGET),
        "removed_bytes": removed_bytes,
        "sha256": actual,
        "model_only_retained": str(model_only),
        "free_gib": free_bytes / (1024**3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
