import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import yaml

from .model import Config, FDT100
from .runner import Engine, environment, source_digest
from .storage import CheckpointStore, atomic_json, ensure_space

ROOT = Path(__file__).resolve().parents[2]


def read_spec(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--max-attempts", type=int, required=True)
    smoke.add_argument("--wall-seconds", type=int, required=True)
    for name in ("train", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--wall-seconds", type=int, required=True)
        command.add_argument("--max-attempts", type=int, required=True)
        if name == "train":
            command.add_argument("--config", required=True)
    verify = sub.add_parser("verify-checkpoint")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--fresh-process", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True)
    stop = sub.add_parser("stop")
    stop.add_argument("--run-dir", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True, help="complete marker JSON, never an unverified raw payload")
    evaluate.add_argument("--suite", required=True)
    probe = sub.add_parser("_resume-probe")
    for argument in ("run-dir", "marker", "budget", "expected"):
        probe.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps({**environment(), "python_executable": sys.executable,
                          "project_free_bytes": shutil.disk_usage(ROOT).free,
                          "minimum_free_bytes": 20 * 2**30, "source": source_digest()}, ensure_ascii=False, indent=2))
    elif args.command == "preflight":
        spec = read_spec(args.config)
        ensure_space(ROOT, 6 * 2**30)
        if not torch.cuda.is_available() or "sm_120" not in torch.cuda.get_arch_list() and torch.cuda.get_device_capability() == (12, 0):
            raise RuntimeError("CUDA build does not support the selected GPU")
        model = FDT100(Config.smoke() if spec.get("profile") == "tiny" else Config())
        result = {"cuda_available": True, "environment": environment(), "model": model.counts(),
                  "disk_reserve_checked": True, "gpu_tensor_check": "deferred to budgeted smoke", "source": source_digest()}
        atomic_json(ROOT / "docs" / "local_preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "smoke":
        from .smoke import run_suite
        report = run_suite(read_spec(args.config), ROOT, args.max_attempts, args.wall_seconds)
        print(json.dumps({"complete": report["completed"], "attempts": report["budget"]["attempts"], "seconds": report["total_gpu_window_seconds"]}))
    elif args.command == "_resume-probe":
        from .smoke import resume_probe
        resume_probe(args.run_dir, args.marker, args.budget, args.expected)
    elif args.command in {"train", "resume"}:
        if args.wall_seconds <= 0 or args.max_attempts <= 0:
            raise ValueError("explicit positive time and attempt budgets required")
        out = Path(args.run_dir).resolve()
        if args.command == "train" and out.exists():
            raise FileExistsError("new run directory required")
        start = time.monotonic()
        spec = read_spec(args.config) if args.command == "train" else json.loads((out / "resolved_config.json").read_text())
        ensure_space(ROOT, 6 * 2**30)
        with CheckpointStore(out).lock():
            engine = Engine(spec, out, resume=args.command == "resume")
            import signal
            signal.signal(signal.SIGINT, lambda *_: (out / "STOP").write_text("SIGINT cooperative stop", encoding="utf-8"))
            signal.signal(signal.SIGTERM, lambda *_: (out / "STOP").write_text("SIGTERM cooperative stop", encoding="utf-8"))
            engine.run(args.max_attempts, start + args.wall_seconds)
    elif args.command == "verify-checkpoint":
        if args.fresh_process:
            result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "fdt_local", "verify-checkpoint", "--run-dir", args.run_dir], capture_output=True, text=True, encoding="utf-8", check=True)
            print(result.stdout)
        else:
            store = CheckpointStore(args.run_dir)
            expected = json.loads((Path(args.run_dir) / "contract.json").read_text()) if (Path(args.run_dir) / "contract.json").exists() else None
            meta, payload, marker = store.recover(expected)
            module = __import__("fdt_local.model_masked" if payload["contract"]["variant"] == "masked" else "fdt_local.model", fromlist=["FDT100"])
            model = module.FDT100(module.Config(**payload["contract"]["config"]))
            model.load_state_dict(payload["model"], strict=True)
            print(json.dumps({"verified": True, "marker": str(marker), "sha256": meta["sha256"],
                              "bytes": meta["bytes"], "parameters": model.counts()["parameters"],
                              "attempts": payload["attempts"], "updates": payload["updates"], "cpu_metadata_reload": True}))
    elif args.command == "status":
        path = Path(args.run_dir) / "status.json"
        print(path.read_text(encoding="utf-8") if path.exists() else '{"status":"not_started"}')
    elif args.command == "stop":
        directory = Path(args.run_dir).resolve()
        if not (directory / "contract.json").exists():
            raise ValueError("existing run required")
        (directory / "STOP").write_text("User cooperative stop", encoding="utf-8")
        print("Stop requested; current kernel and durable save must finish before exit")
    elif args.command == "evaluate":
        from .objectives import evaluate as evaluate_model
        from .data import make_split
        marker = Path(args.checkpoint).resolve()
        run = marker.parent
        spec = json.loads((run / "resolved_config.json").read_text())
        suite = read_spec(args.suite)
        with CheckpointStore(run).lock():
            engine = Engine(spec, run, resume=True, marker=marker)
            meta, _ = engine.store.verify(marker, engine.contract, load=False)
            output = run / (Path(args.suite).stem + "_evaluation.json")
            from .evaluation import complete_missing
            def evaluate_task(values):
                items = make_split("test", values["graphs"], seed=values["seed"], heldout=values.get("heldout", False), n_range=tuple(values.get("n_range", [4, 6])))
                return evaluate_model(engine.model, items, "cuda", 1, tuple(suite["noise_seeds"]))
            complete = complete_missing(output, meta["sha256"], suite, evaluate_task)
            print(json.dumps({"evaluation_complete": complete, "output": str(output)}))


if __name__ == "__main__":
    main()
