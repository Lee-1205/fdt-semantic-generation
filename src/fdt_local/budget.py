"""One shared cap across every smoke process in this handoff."""
from pathlib import Path
import time
from filelock import FileLock
from .storage import atomic_json
import json


class SmokeBudget:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=0)

    def read(self):
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"max_attempts": 5, "max_gpu_seconds": 600, "attempts": 0,
                "gpu_seconds": 0, "runs": [], "status": "NEW"}

    def begin(self):
        with self.lock:
            value = self.read()
            if value["status"] != "NEW":
                raise ValueError("smoke suite already started; budget cannot be reset or automatically retried")
            value.update(status="ACTIVE", started_unix=time.time(), deadline_unix=time.time() + 600)
            atomic_json(self.path, value)
            return value

    def reserve_attempt(self, label):
        with self.lock:
            value = self.read()
            if value["status"] != "ACTIVE" or time.time() >= value["deadline_unix"] or value["attempts"] >= 5:
                raise RuntimeError("global smoke cap reached")
            value["attempts"] += 1
            value["runs"].append({"label": label, "attempt": value["attempts"], "reserved_unix": time.time()})
            atomic_json(self.path, value)
            return value["attempts"]

    def finish(self, elapsed, result):
        with self.lock:
            value = self.read()
            value.update(status="COMPLETE" if result else "FAILED", gpu_seconds=elapsed,
                         completed_unix=time.time())
            atomic_json(self.path, value)
