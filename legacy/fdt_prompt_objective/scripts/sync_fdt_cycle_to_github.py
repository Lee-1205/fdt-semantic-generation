from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_RELEASE_ASSET_BYTES = 1_900_000_000
DEFAULT_REMOTE = "Lee-1205/pure-fdt-research-archive"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )


def find_gh() -> str:
    direct = shutil.which("gh")
    if direct:
        return direct
    root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(root.glob("GitHub.cli_*/*/gh.exe")) + sorted(root.glob("GitHub.cli_*/gh.exe"))
    if matches:
        return str(matches[-1])
    raise FileNotFoundError("GitHub CLI was not found. Install gh and run gh auth login.")


def copy_run_evidence(run_dir: Path, record_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for source in sorted(run_dir.rglob("*")):
        if not source.is_file() or source.suffix.lower() in {".pt", ".tmp", ".partial"}:
            continue
        relative = source.relative_to(run_dir)
        target = record_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({
            "path": relative.as_posix(),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        })
    return copied


def release_assets(gh: str, remote: str, tag: str) -> set[str]:
    result = run([gh, "release", "view", tag, "--repo", remote, "--json", "assets"], check=False)
    if result.returncode != 0:
        return set()
    payload = json.loads(result.stdout)
    return {item["name"] for item in payload.get("assets", [])}


def ensure_release(gh: str, remote: str, tag: str, title: str, notes: str) -> None:
    view = run([gh, "release", "view", tag, "--repo", remote], check=False)
    if view.returncode == 0:
        return
    run([gh, "release", "create", tag, "--repo", remote, "--title", title, "--notes", notes])


def upload(gh: str, remote: str, tag: str, path: Path, uploaded: set[str]) -> None:
    if path.name in uploaded:
        return
    run([gh, "release", "upload", tag, str(path), "--repo", remote])
    uploaded.add(path.name)


def upload_checkpoint(gh: str, remote: str, tag: str, checkpoint: Path, record_dir: Path) -> dict[str, Any]:
    checkpoint_info = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256(checkpoint),
        "parts": [],
    }
    uploaded = release_assets(gh, remote, tag)
    if checkpoint.stat().st_size <= MAX_RELEASE_ASSET_BYTES:
        upload(gh, remote, tag, checkpoint, uploaded)
        checkpoint_info["parts"].append({"asset": checkpoint.name, "bytes": checkpoint.stat().st_size})
        return checkpoint_info

    with tempfile.TemporaryDirectory(prefix="fdt-gh-parts-") as temporary:
        part_dir = Path(temporary)
        with checkpoint.open("rb") as source:
            index = 1
            while True:
                block = source.read(MAX_RELEASE_ASSET_BYTES)
                if not block:
                    break
                part = part_dir / f"{checkpoint.name}.part{index:03d}"
                part.write_bytes(block)
                upload(gh, remote, tag, part, uploaded)
                checkpoint_info["parts"].append({
                    "asset": part.name,
                    "bytes": part.stat().st_size,
                    "sha256": sha256(part),
                })
                part.unlink()
                index += 1
    parts_manifest = record_dir / "checkpoint_parts.json"
    atomic_json(parts_manifest, checkpoint_info)
    upload(gh, remote, tag, parts_manifest, uploaded)
    return checkpoint_info


def git_sync(repo: Path, message: str) -> str:
    run(["git", "add", "records/runs"], cwd=repo)
    status = run(["git", "status", "--porcelain"], cwd=repo).stdout.strip()
    if status:
        run(["git", "commit", "-m", message], cwd=repo)
        run(["git", "pull", "--rebase", "origin", "main"], cwd=repo)
        run(["git", "push", "origin", "main"], cwd=repo)
    return run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an FDT cycle in GitHub and optionally publish its checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--upload-checkpoint", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    repo = args.repo.resolve()
    if run_dir.drive.upper() != "C:" or repo.drive.upper() != "C:":
        raise ValueError("FDT GitHub synchronization is C:-only")
    checkpoint = run_dir / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    state_path = run_dir / "analysis" / "github_sync_state.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "SYNCING",
        "started_at": now(),
        "run": str(run_dir),
        "remote": args.remote,
    }
    atomic_json(state_path, state)
    try:
        record_dir = repo / "records" / "runs" / run_dir.name
        record_dir.mkdir(parents=True, exist_ok=True)
        files = copy_run_evidence(run_dir, record_dir)
        verification_path = run_dir / "checkpoint_verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8-sig")) if verification_path.exists() else {}
        cycle_manifest = {
            "schema_version": 1,
            "recorded_at": now(),
            "run_name": run_dir.name,
            "source_run": str(run_dir),
            "checkpoint": {
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "verification": verification,
            },
            "evidence_files": files,
        }
        manifest_path = record_dir / "cycle_manifest.json"
        atomic_json(manifest_path, cycle_manifest)
        commit = git_sync(repo, f"Record FDT cycle {run_dir.name}")

        release = None
        if args.upload_checkpoint:
            gh = find_gh()
            tag = f"cycle-{run_dir.name.lower()}"
            ensure_release(
                gh,
                args.remote,
                tag,
                f"FDT cycle: {run_dir.name}",
                "Atomic checkpoint and audited run evidence. See records/runs in the repository for provenance.",
            )
            checkpoint_info = upload_checkpoint(gh, args.remote, tag, checkpoint, record_dir)
            uploaded = release_assets(gh, args.remote, tag)
            upload(gh, args.remote, tag, manifest_path, uploaded)
            release = {"tag": tag, "checkpoint": checkpoint_info}

        state.update({
            "status": "COMPLETE",
            "completed_at": now(),
            "commit": commit,
            "release": release,
        })
        atomic_json(state_path, state)
    except Exception as error:
        state.update({"status": "FAILED", "failed_at": now(), "error": str(error)})
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
