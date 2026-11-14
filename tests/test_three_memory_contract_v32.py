import sys
from pathlib import Path
import pytest,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research.three_memory_contract_v32 import *

def fixture():
    p=PersistentFuzzySnapshot.prior_only(torch.eye(2),torch.tensor([[10.,20.],[30.,40.]]),'epoch32')
    c=ContextualFuzzyBank.empty(1,2,2,torch.tensor([.5,.5]),'doc','v1')
    e=ExactSource.from_input('doc','v1','pwd=R8j-9X Cafe\u0301'.encode())
    return p,c,e

def test_separate_prior_and_context_with_float64_distinct_anchor_values():
    p,c,e=fixture();mu=torch.tensor([[.7,.8]])
    assert mu.sum()>1 and c.long_N.dtype==torch.float64 and c.read(mu).count_nonzero()==0
    assert p.read(mu).count_nonzero()>0
    before=c.read(mu);a=c.write_after_read(torch.tensor([[1.,0.]]),torch.tensor([[1.,2.]]),torch.tensor([True]))
    b=a.write_after_read(torch.tensor([[0.,1.]]),torch.tensor([[9.,8.]]),torch.tensor([True]))
    assert before.count_nonzero()==0 and c.long_D.count_nonzero()==0
    assert torch.equal(b.long_N,torch.tensor([[[1.,2.],[9.,8.]]],dtype=torch.float64))
    assert not torch.equal(b.long_D,b.recent_D) and not torch.equal(b.read(torch.tensor([[1.,0.]])),b.read(torch.tensor([[0.,1.]])))
    assert ThreeMemoryBundle(p,b,(e,),'epoch32').persistent.certified_D.count_nonzero()==0

def test_context_validity_source_separation_and_gradient_reference():
    p,c,e=fixture();v=torch.tensor([[2.,3.]],requires_grad=True);mu=torch.tensor([[.6,.4]],requires_grad=True)
    invalid=c.write_after_read(mu,v,torch.tensor([False]));assert invalid.long_D.count_nonzero()==0
    b=c.write_after_read(mu,v,torch.tensor([True]));b.read(mu).sum().backward()
    assert v.grad is not None and mu.grad is not None
    padded=b.write_after_read(mu,v,torch.tensor([False]))
    assert torch.equal(padded.recent_N,b.recent_N) and torch.equal(padded.recent_D,b.recent_D)
    other=ExactSource.from_input('other','v1',b'other')
    with pytest.raises(ValueError):ThreeMemoryBundle(p,b,(other,),'epoch32')
    with pytest.raises(ValueError):ThreeMemoryBundle(p,b,(e,),'different_epoch')

def test_exact_transport_is_byte_preserving_not_normalization_or_identity_merge():
    p,c,e=fixture();selection=CopySelection('doc','v1',4,10,'cpu_mechanical_fixture')
    permit=issue_permit(e,selection);assert emit_copy(permit)==b'R8j-9X'
    decomposed=issue_permit(e,CopySelection('doc','v1',11,len(e.raw),'cpu_mechanical_fixture'))
    assert emit_copy(decomposed)=='Cafe\u0301'.encode() and emit_copy(decomposed)!='Café'.encode()
    with pytest.raises(ValueError):issue_permit(e,CopySelection('doc','v2',4,10,'cpu_mechanical_fixture'))
    with pytest.raises(UnicodeDecodeError):e.span(len(e.raw)-1,len(e.raw))

def test_whole_quote_reserved_for_future_explicit_semantic_commit_not_current_codec():
    p,c,e=fixture();full=CopySelection('doc','v1',0,len(e.raw),'semantic_commit')
    with pytest.raises(ValueError):issue_permit(e,full)
    with pytest.raises(ValueError):CopyPolicy(stage='codec_learning',allow_full_source=True)
    # This is a mechanical future-capability fixture, not current model inference.
    permit=issue_permit(e,full,CopyPolicy(stage='post_semantic_commit',allow_full_source=True))
    assert emit_copy(permit)==e.raw
    bundle=ThreeMemoryBundle(p,c,(e,),'epoch32');z=torch.zeros(1,2,4)
    with pytest.raises(ValueError):CodecCommit.seal(z,bundle,(permit,))
    pieces=tuple(issue_permit(e,CopySelection('doc','v1',a,b,'cpu_mechanical_fixture')) for a,b in ((0,4),(4,len(e.raw))))
    with pytest.raises(ValueError):CodecCommit.seal(z,bundle,pieces)
    assert CodecCommit.seal(z,bundle,(permit,),CopyPolicy(stage='post_semantic_commit',allow_full_source=True)).copy_stage=='post_semantic_commit'

def test_realizer_commit_boundary_no_live_store_or_writable_tensor_alias():
    p,c,e=fixture();bundle=ThreeMemoryBundle(p,c,(e,),'epoch32');z=torch.arange(8.).reshape(1,2,4)
    permit=issue_permit(e,CopySelection('doc','v1',4,10,'cpu_mechanical_fixture'))
    commit=CodecCommit.seal(z,bundle,(permit,));z.add_(100)
    a,permits=realizer_inputs(commit);a.zero_();b,_=realizer_inputs(commit)
    assert torch.equal(b,torch.arange(8.).reshape(1,2,4)) and emit_copy(permits[0])==b'R8j-9X'
    with pytest.raises(TypeError):realizer_inputs(bundle)
    assert not torch.cuda.is_initialized()

def test_current_stage_is_language_codec_not_denoising_training():
    assert_codec_training_only([('decoder.head.weight',torch.ones(1,requires_grad=True)),('core.refiner.0.weight',torch.ones(1))],0)
    with pytest.raises(ValueError):assert_codec_training_only([],1)
    with pytest.raises(ValueError):assert_codec_training_only([('core.rel_left',torch.ones(1,requires_grad=True))],0)
