import torch
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.next_causal_lm import OptimizedCausalFuzzyAnchorLayer


def make_layer(backend):
    cfg = ModelConfig(vocab_size=32, dim=24, n_layers=1, n_heads=4, max_seq_len=12, num_anchors=10, top_k=3, dropout=0.0, routing_type="gaussian", routing_backend=backend)
    return OptimizedCausalFuzzyAnchorLayer(cfg)


def test_cdist_and_matmul_membership_are_equivalent():
    torch.manual_seed(4)
    matmul = make_layer("matmul")
    cdist = make_layer("cdist")
    cdist.load_state_dict(matmul.state_dict())
    q = F.normalize(torch.randn(2, 7, 24), dim=-1)
    keys = F.normalize(matmul.anchor_keys, dim=-1)
    logits_a, distance_a = matmul.membership_logits(q, keys)
    logits_b, distance_b = cdist.membership_logits(q, keys)
    assert torch.allclose(distance_a, distance_b, atol=2e-5, rtol=2e-5)
    assert torch.allclose(logits_a, logits_b, atol=2e-5, rtol=2e-5)


def test_cdist_and_matmul_full_routing_outputs_are_equivalent():
    torch.manual_seed(8)
    matmul = make_layer("matmul").eval()
    cdist = make_layer("cdist").eval()
    cdist.load_state_dict(matmul.state_dict())
    x = torch.randn(2, 7, 24)
    mask = torch.ones(2, 7, dtype=torch.long)
    output_a, stats_a = matmul(x, mask)
    output_b, stats_b = cdist(x, mask)
    assert torch.allclose(output_a, output_b, atol=3e-5, rtol=3e-5)
    assert torch.equal(stats_a.indices, stats_b.indices)
    assert torch.allclose(stats_a.membership, stats_b.membership, atol=3e-5, rtol=3e-5)
