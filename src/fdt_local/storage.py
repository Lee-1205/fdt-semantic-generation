"""Local single-writer, payload-first checkpoint transactions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from filelock import FileLock
import torch

MIN_FREE_BYTES = 20 * 2**30


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_space(directory, additional_bytes=0, min_free_bytes=MIN_FREE_BYTES):
    free = shutil.disk_usage(Path(directory)).free
    if free - int(additional_bytes) < min_free_bytes:
        raise OSError(f"disk reserve: free={free}, proposed={additional_bytes}, floor={min_free_bytes}")
    return free


class CheckpointStore:
    schema = "fdt_local_checkpoint_v1"

    def __init__(self, directory, keep=3):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def lock(self):
        return FileLock(str(self.directory / "writer.lock"), timeout=0)

    def markers(self):
        candidates = []
        for path in self.directory.glob("complete_*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                if meta.get("schema") == self.schema and isinstance(meta.get("generation"), int):
                    candidates.append((meta["generation"], path, meta))
            except (OSError, ValueError):
                pass
        return sorted(candidates, key=lambda value: value[0], reverse=True)

    def verify(self, marker, expected_contract=None, load=True):
        marker = Path(marker).resolve()
        if marker.parent != self.directory:
            raise ValueError("marker outside run")
        meta = json.loads(marker.read_text(encoding="utf-8"))
        required = {"schema", "file", "sha256", "bytes", "contract", "attempts", "updates", "generation", "run_id"}
        if not required <= meta.keys() or meta["schema"] != self.schema:
            raise ValueError("incomplete checkpoint marker")
        name = meta["file"]
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("payload_"):
            raise ValueError("unsafe checkpoint payload name")
        path = self.directory / name
        if path.stat().st_size != meta["bytes"] or sha256(path) != meta["sha256"]:
            raise ValueError("checkpoint size/hash mismatch")
        if expected_contract is not None and canonical(meta["contract"]) != canonical(expected_contract):
            raise ValueError("source/config/data/plan/environment mismatch; new run required")
        payload = torch.load(path, map_location="cpu", weights_only=True) if load else None
        if payload is not None:
            if payload.get("schema") != self.schema:
                raise ValueError("payload schema mismatch")
            for field in ("contract", "attempts", "updates", "run_id"):
                if canonical(payload[field]) != canonical(meta[field]):
                    raise ValueError(f"payload/marker mismatch: {field}")
            if not {"model", "optimizer", "scaler", "sampler_rng", "noise_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "phase", "phase_step", "history"} <= payload.keys():
                raise ValueError("checkpoint lacks full resume state")
        return meta, payload

    def recover(self, expected_contract=None):
        errors = []
        for _, marker, _ in self.markers():
            try:
                return (*self.verify(marker, expected_contract), marker)
            except (OSError, ValueError, RuntimeError) as exc:
                if "environment mismatch" in str(exc):
                    raise
                errors.append({"marker": marker.name, "error": str(exc)})
        raise ValueError(f"no verified checkpoint: {errors}")

    def commit(self, payload, estimated_bytes, tag=None):
        ensure_space(self.directory, estimated_bytes)
        generation = max((item[0] for item in self.markers()), default=-1) + 1
        path = self.directory / f"payload_{generation:06d}.pt"
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=self.directory)
        start = time.monotonic()
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if Path(temporary).stat().st_size > estimated_bytes:
                ensure_space(self.directory)
            os.replace(temporary, path)
            meta = {"schema": self.schema, "generation": generation, "file": path.name,
                    "bytes": path.stat().st_size, "sha256": sha256(path),
                    "contract": payload["contract"], "run_id": payload["run_id"],
                    "attempts": payload["attempts"], "updates": payload["updates"],
                    "phase": payload["phase"], "phase_step": payload["phase_step"],
                    "tag": tag, "directory_fsync_supported": False,
                    "local_checkpoint_verified": True, "backup_to_drive_verified": False}
            # Read the whole payload before publishing its completion marker.
            check = torch.load(path, map_location="cpu", weights_only=True)
            if canonical(check["contract"]) != canonical(payload["contract"]) or check["attempts"] != payload["attempts"]:
                raise ValueError("fresh payload verification failed")
            del check
            marker = path.with_name(f"complete_{generation:06d}.json")
            atomic_json(marker, meta)
            atomic_json(self.directory / "latest.json", {"marker": marker.name, "sha256": meta["sha256"]})
            self._rotate()
            meta["save_seconds"] = time.monotonic() - start
            return meta, marker
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _rotate(self):
        # Only this new run's verified, untagged generations may be retired.
        untagged = [(generation, marker, meta) for generation, marker, meta in self.markers() if not meta.get("tag")]
        for _, marker, meta in untagged[self.keep:]:
            payload = self.directory / meta["file"]
            if payload.exists() and sha256(payload) == meta["sha256"]:
                with (self.directory / "rotation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(canonical({"path": str(payload), "bytes": meta["bytes"], "sha256": meta["sha256"]}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                payload.unlink()
                marker.unlink()
