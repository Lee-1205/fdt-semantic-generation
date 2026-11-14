from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from fdt_rlm.config import load_yaml_like
from fdt_rlm.data.manifest import write_json
from fdt_rlm.data.source_adapters import probe_hf_dataset


def probe_source(name: str, spec: dict):
    dataset = spec.get("dataset") or spec.get("preferred", {}).get("dataset")
    config = spec.get("preferred_config") or spec.get("dataset_config") or spec.get("config") or ""
    split = spec.get("split") or spec.get("preferred", {}).get("split", "train")
    streaming = bool(spec.get("streaming", spec.get("preferred", {}).get("streaming", True)))
    if not dataset:
        return None
    return probe_hf_dataset(dataset=dataset, config=config, split=split, streaming=streaming)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    p.add_argument("--output", default="")
    args = p.parse_args()

    cfg = load_yaml_like(args.config)
    results = {}
    for name, spec in cfg.get("sources", {}).items():
        if name == "code":
            candidates = [spec.get("preferred", {})] + list(spec.get("fallbacks", []))
            code_results = []
            for cand in candidates:
                res = probe_source(cand.get("dataset", "code"), cand)
                if res:
                    code_results.append(res.__dict__)
                    status = "OK" if res.ok else "FAIL"
                    print(f"[{status}] {res.name} field={res.content_field or '-'} warning={res.warning or '-'}")
            results[name] = code_results
        else:
            res = probe_source(name, spec)
            if res:
                results[name] = res.__dict__
                status = "OK" if res.ok else "FAIL"
                print(f"[{status}] {res.name} field={res.content_field or '-'} warning={res.warning or '-'}")
    out = Path(args.output or ROOT / "artifacts" / "probe_report.json")
    write_json(out, {"results": results})
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

