from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from fdt_rlm.data.manifest import write_json
from fdt_rlm.data.source_adapters import probe_hf_dataset, stream_stack_smol_xl_group


def auth_status():
    try:
        from huggingface_hub import get_token, whoami

        token = get_token()
        if not token:
            return False, "NOT_AUTHENTICATED"
        try:
            whoami(token=token)
            return True, "AUTHENTICATED"
        except Exception:
            return True, "TOKEN_PRESENT_WHOAMI_FAILED"
    except Exception as exc:
        return False, f"AUTH_CHECK_FAILED:{type(exc).__name__}"


def try_stack_smol_xl():
    groups = ["python", "javascript_typescript", "c_cpp", "java", "rust_go", "structured_docs"]
    samples = {}
    for group in groups:
        try:
            doc = next(iter(stream_stack_smol_xl_group(group)))
            samples[group] = {
                "ok": True,
                "content_available": bool(doc.text),
                "content_field": "content",
                "language": doc.metadata.get("language"),
                "path": doc.metadata.get("path"),
                "sample_chars": len(doc.text),
            }
        except Exception as exc:
            samples[group] = {"ok": False, "reason": type(exc).__name__ + ": " + str(exc)[:200]}
    ready = all(row.get("ok") and row.get("content_available") for row in samples.values())
    return ready, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(ROOT / "artifacts" / "code_source_resolution.json"))
    args = p.parse_args()

    authed, auth = auth_status()
    sources = {}

    for name in ["bigcode/starcoderdata", "bigcode/the-stack-smol"]:
        res = probe_hf_dataset(name, split="train", streaming=True)
        status = "READY" if res.ok and res.content_field else "AUTH_REQUIRED_OR_GATED" if "gated" in res.warning.lower() or "authenticated" in res.warning.lower() else "OTHER"
        sources[name] = {
            "status": status,
            "content_available": bool(res.ok and res.content_field),
            "content_field": res.content_field,
            "reason": res.warning,
        }

    ready, samples = try_stack_smol_xl()
    sources["bigcode/the-stack-smol-xl"] = {
        "status": "READY" if ready else "SCHEMA_ERROR",
        "content_available": ready,
        "content_field": "content",
        "language_field": "lang",
        "access_method": "generic_json_loader_language_files",
        "language_samples": samples,
    }

    res = probe_hf_dataset("bigcode/the-stack-v2", split="train", streaming=True)
    sources["bigcode/the-stack-v2"] = {
        "status": "METADATA_ONLY_OR_CONTENT_NOT_AVAILABLE" if res.ok else "AUTH_REQUIRED_OR_GATED",
        "content_available": bool(res.ok and res.content_field == "content"),
        "content_field": res.content_field,
        "reason": res.warning,
    }

    for name in ["codeparrot/github-code", "codeparrot/github-code-clean"]:
        try:
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(name, repo_type="dataset")
            parquet = [f for f in files if f.endswith(".parquet")]
            sample_ok = False
            fields = []
            if parquet:
                try:
                    from datasets import load_dataset

                    ds = load_dataset("parquet", data_files=f"hf://datasets/{name}/{parquet[0]}", split="train", streaming=True)
                    row = next(iter(ds))
                    fields = list(row.keys())
                    sample_ok = isinstance(row.get("content"), str) and bool(row.get("content"))
                except Exception:
                    sample_ok = False
            status = "GENERIC_PARQUET_AVAILABLE" if parquet else "LEGACY_SCRIPT_ONLY"
            sources[name] = {
                "status": status,
                "content_available": sample_ok,
                "access_method": "generic_parquet_loader" if parquet else "",
                "content_field": "content" if "content" in fields else "",
                "fields": fields,
                "parquet_files_seen": len(parquet),
            }
        except Exception as exc:
            sources[name] = {"status": "OTHER", "content_available": False, "reason": type(exc).__name__ + ": " + str(exc)[:200]}

    selected = "bigcode/the-stack-smol-xl" if sources["bigcode/the-stack-smol-xl"]["status"] == "READY" else ""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "huggingface_authenticated": authed,
        "auth_status": auth,
        "sources": sources,
        "selected_source": selected,
        "selected_for": "10M smoke corpus" if selected else "",
    }
    write_json(args.output, report)
    print(f"selected_source={selected or 'NONE'}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
