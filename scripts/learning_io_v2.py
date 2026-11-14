"""Retry-bounded JSON I/O for Windows learning progress and supervisor state."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

RETRY_WINERRORS = {5, 32, 33}
RETRY_SECONDS = 3.0


def _winerror(exc: OSError) -> int | None:
    return getattr(exc, "winerror", None)


def _retryable(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) and _winerror(exc) in RETRY_WINERRORS


def replace_with_retry(source, destination, *, monotonic=time.monotonic, sleep=time.sleep,
                       retry_seconds=RETRY_SECONDS) -> None:
    source, destination = Path(source), Path(destination)
    deadline = monotonic() + min(max(0.0, retry_seconds), RETRY_SECONDS)
    delay = 0.025
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            remaining = deadline - monotonic()
            if not _retryable(exc) or remaining <= 0:
                raise
            sleep(min(delay, remaining))
            delay = min(delay * 2, 0.25)


def atomic_json(path, value, *, monotonic=time.monotonic, sleep=time.sleep,
                retry_seconds=RETRY_SECONDS) -> None:
    """Durably write a temporary JSON then replace, retrying transient Windows sharing denial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path, monotonic=monotonic, sleep=sleep, retry_seconds=retry_seconds)
    finally:
        if os.path.exists(temporary):
            try:
                bounded_unlink(temporary)
            except OSError:
                pass


def atomic_text(path, text, *, monotonic=time.monotonic, sleep=time.sleep,
                retry_seconds=RETRY_SECONDS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(str(text))
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path, monotonic=monotonic, sleep=sleep, retry_seconds=retry_seconds)
    finally:
        if os.path.exists(temporary):
            try:
                bounded_unlink(temporary)
            except OSError:
                pass


def read_text_shared(path: Path, encoding="utf-8") -> str:
    """Read with DELETE sharing on Windows so atomic replace can proceed concurrently."""
    path = Path(path)
    if os.name != "nt":
        return path.read_text(encoding=encoding)
    import msvcrt
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        code = ctypes.get_last_error()
        raise ctypes.WinError(code, str(path))
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        return stream.read().decode(encoding)


def load_json(path, *, retries=3, sleep=time.sleep):
    """Read and parse a JSON snapshot, closing the shared file handle before returning."""
    path = Path(path)
    last_error = None
    for attempt in range(retries):
        try:
            return json.loads(read_text_shared(path))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if isinstance(exc, FileNotFoundError):
                return None
            last_error = exc
            if attempt + 1 < retries and _retryable(exc):
                sleep(min(0.025 * (2 ** attempt), 0.2))
                continue
            if isinstance(exc, OSError):
                raise
            return None
    if last_error:
        raise last_error
    return None


def bounded_unlink(path, *, monotonic=time.monotonic, sleep=time.sleep, retry_seconds=RETRY_SECONDS) -> bool:
    """Delete only the named path, retrying transient Windows sharing violations."""
    path = Path(path)
    if not path.exists():
        return False
    start = monotonic()
    deadline = start + min(max(0.0, retry_seconds), RETRY_SECONDS)
    delay = 0.025
    while True:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            remaining = deadline - monotonic()
            if not _retryable(exc) or remaining <= 0:
                raise
            sleep(min(delay, remaining))
            delay = min(delay * 2, 0.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def rotate_verified_generations(store) -> None:
    """Retire only verified untagged generations beyond the store's keep count."""
    candidates = store.markers()
    untagged = [(generation, marker, meta) for generation, marker, meta in candidates if not meta.get("tag")]
    for _, marker, meta in untagged[store.keep:]:
        if meta.get("local_checkpoint_verified") is not True or Path(marker).parent.resolve() != store.directory.resolve():
            continue
        name = meta.get("file")
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("payload_"):
            continue
        payload = store.directory / name
        if not payload.exists() or payload.stat().st_size != meta.get("bytes") or _sha256(payload) != meta.get("sha256"):
            continue
        manifest = store.directory / "rotation.jsonl"
        with manifest.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical({"path": str(payload), "bytes": meta["bytes"], "sha256": meta["sha256"]}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        bounded_unlink(payload)
        bounded_unlink(marker)
