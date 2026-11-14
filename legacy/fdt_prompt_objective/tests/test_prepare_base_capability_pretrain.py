from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "prepare_base_capability_pretrain.py"
    spec = importlib.util.spec_from_file_location("prepare_base_capability_pretrain_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pack_documents_preserves_order_and_masks_only_padding():
    module = load_module()
    ids, masks = module.pack_documents([[1, 2], [3, 4, 5]], seq_len=4, pad_token_id=0)
    assert ids.tolist() == [[1, 2, 3, 4], [5, 0, 0, 0]]
    assert masks.tolist() == [[1, 1, 1, 1], [1, 0, 0, 0]]


def test_document_hash_is_stable_and_order_sensitive():
    module = load_module()
    assert module.document_hash([1, 2, 3]) == module.document_hash([1, 2, 3])
    assert module.document_hash([1, 2, 3]) != module.document_hash([3, 2, 1])
