import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_full_scan_confidence_includes_unproposed_distractor():
    from fdt_rlm.lexical_pointer import SparseLexicalPointer,AnchorIndexedExactMemory
    pointer=SparseLexicalPointer(hidden_dim=2,pointer_dim=2,window=8)
    with torch.no_grad():
        pointer.q_proj.weight.copy_(torch.eye(2));pointer.logit_scale.zero_()
    source=torch.tensor([[3,10,11,12]])
    keys=torch.tensor([[[1.,0.],[.99,.1],[.98,.2],[0.,1.]]])
    memory=AnchorIndexedExactMemory.from_prompt(source,torch.ones(1,4,1,dtype=torch.long),torch.ones_like(source),key_vectors=keys)
    _,diag=pointer.mix_next_logits(torch.zeros(1,32),torch.tensor([[[1.,0.]]]),source,torch.ones_like(source),
        anchor_memory=memory,query_anchor_ids=torch.ones(1,1,dtype=torch.long),
        indexed_candidates=(torch.tensor([[0]]),torch.tensor([[True]])),full_scan_fallback=True,candidate_cap=1)
    assert diag['source_positions'][0][0]==0
    assert diag['pointer_confidence']<.6


def test_long_cache_routing_and_relative_state_identity():
    from fdt_rlm.config import ModelConfig
    from fdt_rlm.models import build_model
    from audit_astra_cache import audit_context
    cfg=ModelConfig(model_type='fdt_v4',vocab_size=97,pad_token_id=0,eos_token_id=2,dim=32,n_layers=1,n_heads=4,
        max_seq_len=16384,use_rope=True,num_anchors=16,top_k=4,router_dim=16,local_attention_window=64,
        anchor_layer_indices=[0],aggregation_impl='sparse_chunked_scan',anchor_scan_chunk_size=64,anchor_recency_bias=4.,
        anchor_recency_reference_len=512,inference_prefix_stable_group_size=64)
    torch.manual_seed(987);model=build_model(cfg).float().eval()
    assert torch.isfinite(model.blocks[0].anchor._decode_recency_scales).all()
    result=audit_context(model,16383)
    assert result['routing_identity']=='PASS'
    assert result['status']=='PASS'


def test_generation_rejects_effective_legacy_overrun():
    from fdt_rlm.config import ModelConfig
    from fdt_rlm.runtime_contract import validate_generation_budget
    cfg=ModelConfig(model_type='fdt_v4',max_seq_len=16384,legacy_position_transition_max_len=2048,
                    legacy_position_scale=1.,rope_transition_alpha=.2,rope_transition_mode='output_blend')
    with pytest.raises(ValueError,match='effective context 2048'):
        validate_generation_budget(cfg,2040,32)


def test_explicit_training_metric_is_not_cursor_rollout():
    from fdt_rlm.lexical_pointer import SparseLexicalPointer
    pointer=SparseLexicalPointer(hidden_dim=8,pointer_dim=4,window=8)
    ids=torch.tensor([[1,10,20,10,20]])
    result=pointer.training_loss(torch.randn(1,5,8),ids,torch.tensor([[-100,-100,-100,10,20]]),torch.ones_like(ids),
        copy_source_positions=torch.tensor([[-1,-1,-1,1,2]]),copy_target_mask=torch.tensor([[0,0,0,1,1]]),source_boundary=torch.tensor([3]))
    assert result.cursor_continuation_rate is None
    assert result.gold_continuation_rate is not None
    assert result.source_position_accuracy is not None


def test_low_confidence_mixed_token_cannot_latch_cursor():
    from fdt_rlm.lexical_pointer import LexicalPointerDecodeState
    state=LexicalPointerDecodeState(source_length=4,max_copy_tokens=8)
    state.commit(10,{'mode':'mixed','mix_gate':.5,'candidate_ids':[[10]],'source_positions':[[0]],
        'span_end_source':'explicit_payload','candidate_payload_ids':[[[10,20]]],
        'cursor_commit_eligible':False})
    assert state.copied_tokens==0
    assert state.payload_token_ids is None


def test_optional_token_loss_partition_invariance():
    from train_fdt_v4_1_retention import chunked_weighted_lm_loss
    class LM:
        def _language_logits(self,x):return x
    torch.manual_seed(9);h=torch.randn(2,4,3,requires_grad=True)
    labels=torch.tensor([[1,1,-100,-100],[1,1,1,2]]);mask=labels.ne(-100)
    actual=chunked_weighted_lm_loss(LM(),h,labels,mask,0,2,2.,2,False,reduction='weighted_sum')/5
    pieces=sum(chunked_weighted_lm_loss(LM(),h[i:i+1],labels[i:i+1],mask[i:i+1],0,2,2.,1,False,reduction='weighted_sum') for i in range(2))/5
    ce=torch.nn.functional.cross_entropy(h[:,:-1].reshape(-1,3),labels[:,1:].reshape(-1),reduction='none',ignore_index=-100).view(2,3)
    weights=mask[:,1:]*(1+labels[:,1:].eq(2))
    expected=(ce*weights).sum()/weights.sum()
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual,pieces)
    torch.testing.assert_close(torch.autograd.grad(actual,h,retain_graph=True)[0],torch.autograd.grad(expected,h)[0])


def test_uncalibrated_exact_is_blocked_at_product_entrypoint(monkeypatch,tmp_path):
    import generate_fdt_v4 as cli
    from fdt_rlm.config import ModelConfig
    class Model:
        exact_pointer=object()
        def to(self,**kwargs):return self
        def eval(self):return self
    class Tokenizer:
        def encode(self,*args,**kwargs):return [3,4,5]
    cfg=ModelConfig(model_type='fdt_v4',max_seq_len=128,exact_memory_mode='copy')
    monkeypatch.setattr(cli.evaluator,'load_checkpoint',lambda _: (Model(),cfg,{}))
    monkeypatch.setattr(cli,'load_tokenizer',lambda _:Tokenizer())
    span=tmp_path/'spans.json';span.write_text('[2,2,2]')
    monkeypatch.setattr(sys,'argv',['generate','--checkpoint','dummy.pt','--tokenizer','dummy','--prompt','x',
        '--max-new-tokens','4','--device','cpu','--exact-span-map',str(span)])
    with pytest.raises(RuntimeError,match='release gate has not passed'):
        cli.main()
