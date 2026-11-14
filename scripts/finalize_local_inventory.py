"""Record source/config/data evidence without modifying pinned run state."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil

from fdt_local.runner import source_digest

root = Path(__file__).resolve().parents[1]
floor = 20 * 1024**3
free = shutil.disk_usage(root).free
if free < floor + 1024**2:
    raise RuntimeError("20GiB reserve would not be protected")
contract = json.loads((root / "runs/smoke_102m/contract.json").read_text(encoding="utf-8"))
if source_digest() != contract["source"]:
    raise ValueError("frozen runtime changed")
paths = [root / name for name in ("pyproject.toml", "requirements-lock.txt", "setup_local.ps1", "README_LOCAL_KR.md")]
for folder, extensions in (("src/fdt_local", (".py",)), ("configs", (".yaml",)),
                           ("tests", (".py",)), ("scripts", (".py",)),
                           ("research", (".py", ".json")), ("docs", (".md", ".json"))):
    paths.extend(p for p in (root / folder).rglob("*")
                 if p.is_file() and p.suffix in extensions
                 and p.name != "artifact_manifest.json" and "__pycache__" not in p.parts)
files = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))]
latest = json.loads((root / "runs/smoke_102m/complete_000002.json").read_text(encoding="utf-8"))
record = {"status": "LOCAL_PREPARATION_AND_E2_CPU_CANDIDATE_COMPLETE",
          "observed_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
          "c_free_bytes": free, "required_reserve_bytes": floor,
          "runtime_source_sha256": contract["source"], "files": files,
          "checkpoint_reference": {"marker": "runs/smoke_102m/complete_000002.json", "contents": latest,
                                   "payload_rehashed_this_inventory": False},
          "smoke_budget": json.loads((root / "runs/smoke_budget.json").read_text(encoding="utf-8")),
          "resume_proof": json.loads((root / "runs/smoke_102m/resume_proof.json").read_text(encoding="utf-8")),
          "new_gpu_or_optimizer_attempts": 0,
          "e2_cpu_pytest_observation": {"command": ".venv/Scripts/python.exe -B -X utf8 -m pytest -q research/test_e2_data_v1.py --basetemp .pytest_e2_resume_20261026",
                                        "passed": 5, "elapsed_seconds": 4.96,
                                        "source": "actual tool execution output, recorded after execution"},
          "long_training_started": False, "drive_backup_verified": False}
target = root / "docs/artifact_manifest.json"
target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": record["status"], "tracked_files": len(files), "c_free_gib": round(free / 1024**3, 2),
                  "runtime_source_unchanged": True, "gpu_attempts_added": 0}))
