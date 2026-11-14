"""Phase-2 tests: written before repairs; no optimizer updates."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_loss_accounting_matches_existing_scalar_and_gradient():
    from train_fdt_v4_1_retention import chunked_weighted_lm_loss
    class LM:
        def _language_logits(self, h): return h
    h = torch.tensor([[[0.,0.,0.]]*4, [[0.,4.,0.]]*4], requires_grad=True)
    labels = torch.tensor([[1,1,-100,-100],[1,1,1,2]])
    legacy = chunked_weighted_lm_loss(LM(),h,labels,labels.ne(-100),0,2,2.,2,False)
    stats = []
    measured = chunked_weighted_lm_loss(LM(),h,labels,labels.ne(-100),0,2,2.,2,False,accounting=stats)
    torch.testing.assert_close(legacy, measured, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(legacy,h,retain_graph=True)[0],torch.autograd.grad(measured,h)[0],rtol=0,atol=0)
    assert [r['supervised_tokens'] for r in stats] == [1,3]
    assert [r['loss_denominator'] for r in stats] == [1.,4.]
    assert sum(r['normalized_source_loss'] for r in stats) == pytest.approx(float(legacy))


def test_runtime_reports_legacy_limit_and_unsupported_cache():
    from fdt_rlm.runtime_contract import runtime_capabilities
    from fdt_rlm.config import ModelConfig
    cfg = ModelConfig(model_type='fdt_v4',max_seq_len=16384,legacy_position_transition_max_len=2048,legacy_position_scale=1.,rope_transition_alpha=.2,rope_transition_mode='output_blend')
    result = runtime_capabilities(cfg)
    assert result['full_forward_limit'] == 2048
    assert result['incremental_limit'] is None
    assert result['incremental_status'] == 'UNSUPPORTED'


def test_cache_configuration_inherits_and_records_explicit_override():
    from audit_fdt_v4_cache_integrity import apply_config_override
    from fdt_rlm.config import ModelConfig
    cfg=ModelConfig(inference_prefix_stable_group_size=64)
    assert apply_config_override(cfg, None)['override'] is None
    assert cfg.inference_prefix_stable_group_size == 64
    result=apply_config_override(cfg,0)
    assert result == {'checkpoint_value':64,'override':0,'effective_value':0}


def test_query_blind_registration_contains_all_records():
    from audit_astra_goldfree_exact import register_records
    class Tokenizer:
        def encode(self,s): return [ord(c) for c in s]
    records=[{'name':'a','value':'123'},{'name':'b','value':'124'},{'name':'a','value':'999'}]
    prompt, ids, meta, source_map=register_records(records,'Copy b.',Tokenizer(),torch.device('cpu'))
    assert int(meta['registered_key_mask'].sum()) == 3
    assert meta['registered_payload_lengths'].tolist() == [[3,3,3]]
    assert len(source_map)==3
    prompt2, ids2, meta2, _=register_records(records,'Copy missing.',Tokenizer(),torch.device('cpu'))
    for key in ['registered_key_positions','registered_payload_ids','registered_payload_lengths']:
        assert torch.equal(meta[key],meta2[key])


@pytest.mark.parametrize('length',[8192,11000,11334,11335,11336,12288,16383])
def test_recency_forward_backward_finite(length):
    from fdt_rlm.models.fdt_v3 import sparse_chunked_prefix_summaries
    m=torch.full((1,length,2),.5,requires_grad=True)
    i=torch.tensor([0,1]).reshape(1,1,2).expand(1,length,2)
    v=torch.ones(1,length,2,requires_grad=True)
    outputs=sparse_chunked_prefix_summaries(m,i,v,torch.ones(1,length),4,512,4.,64)
    assert all(torch.isfinite(x).all() for x in outputs)
    (outputs[0][:,-1].sum()+outputs[1][:,-1].sum()).backward()
    assert torch.isfinite(m.grad).all() and torch.isfinite(v.grad).all()
    torch.testing.assert_close(outputs[1],torch.ones_like(outputs[1]),rtol=0,atol=3e-4)


def test_source_score_function_shared_and_route_neutral():
    from fdt_rlm.lexical_pointer import exact_key_scores
    q=torch.tensor([[1.,0.]])
    k=torch.tensor([[[1.,0.],[0.,1.]]])
    assert exact_key_scores(q,k,torch.tensor(2.)).tolist()==[[2.,0.]]
