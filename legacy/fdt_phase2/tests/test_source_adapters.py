import sys
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.code_language_map import map_language
from fdt_rlm.data.source_adapters import likely_content_field, stream_local_code, stream_local_tar_texts


def test_likely_content_field():
    assert likely_content_field(["id", "text", "url"]) == "text"
    assert likely_content_field(["repo", "code", "path"]) == "code"


def test_language_mapping():
    assert map_language("Python", "x") == "python"
    assert map_language("", "app.tsx") == "javascript_typescript"
    assert map_language("C++", "x") == "c_cpp"


def test_stream_local_code(tmp_path):
    (tmp_path / "a.py").write_text("print('hi')\n", encoding="utf-8")
    docs = list(stream_local_code(tmp_path))
    assert len(docs) == 1
    assert docs[0].bucket == "code_python"


def test_stream_local_tar_texts_accepts_archives_with_txt_extension(tmp_path):
    archive_path = tmp_path / "openwebtext-part.txt"
    payload = b"A clean OpenWebText document."
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("doc.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    docs = list(stream_local_tar_texts(tmp_path, "openwebtext", "local_owt"))
    assert [doc.text for doc in docs] == [payload.decode()]
    assert docs[0].bucket == "openwebtext"
