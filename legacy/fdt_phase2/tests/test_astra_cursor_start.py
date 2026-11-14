"""A token coincidence is never authority to enter deterministic copying."""
import pytest
import torch
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState


def diagnostic(mode='mixed', **extra):
    return {'mode':mode, 'mix_gate':1., 'gate':1., 'pointer_confidence':1.,
            'score_margin':10., 'source_positions':[[0]], 'candidate_ids':[[10]],
            'candidate_payload_ids':[[[10,20]]], 'span_end_source':'explicit_payload', **extra}


@pytest.mark.parametrize('flags',[{}, {'cursor_commit_eligible':True,'hard_copy_eligible':True}])
def test_high_confidence_mixed_coincidence_never_starts_cursor(flags):
    state=LexicalPointerDecodeState(source_length=4)
    state.commit(10,diagnostic(**flags))
    assert state.copied_tokens==0
    assert state.payload_token_ids is None
    assert state.source_token_position is None


def test_legacy_hard_copy_without_explicit_eligibility_fails_closed():
    state=LexicalPointerDecodeState(source_length=4)
    state.commit(10,diagnostic('hard_copy'))
    assert state.copied_tokens==0


def test_orphan_cursor_diagnostic_cannot_start_copying():
    state=LexicalPointerDecodeState(source_length=4)
    state.commit(10,diagnostic('cursor'))
    assert state.copied_tokens==0


@pytest.mark.parametrize('hard_copy',[False,True])
def test_prepare_commit_and_transport_require_explicit_copy_policy(hard_copy):
    class Pointer:
        def mix_next_logits(self,base,*args,**kwargs):
            logits=torch.full_like(base,-10.);logits[0,10]=0.
            return logits,diagnostic()
    state=LexicalPointerDecodeState(source_length=4)
    ids=torch.tensor([[1,10,20,3]])
    base=torch.zeros(1,32)
    logits,diag=state.prepare_logits(Pointer(),base,torch.zeros(1,4,2),ids,torch.ones_like(ids),
                                    min_gate=0.,hard_copy=hard_copy)
    assert diag['cursor_commit_eligible'] is hard_copy
    state.commit(int(logits.argmax(-1)),diag)
    assert state.copied_tokens==int(hard_copy)
    if hard_copy:
        logits,diag=state.prepare_logits(None,base,torch.zeros(1,4,2),ids,torch.ones_like(ids))
        assert diag['mode']=='cursor' and int(logits.argmax(-1))==20
        state.commit(20,diag)
        assert state.completed and state.copied_tokens==2
