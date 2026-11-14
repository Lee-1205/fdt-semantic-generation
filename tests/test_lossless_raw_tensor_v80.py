import copy,io
import pytest
import torch
from research.lossless_raw_tensor_v80 import pack,unpack,MARKER


def same_bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def test_exact_BF16_values_restore_FP32_and_negative_zero_bits():
    t=torch.tensor([0.,-0.,1.25,-18.,256.,float('inf')]);p=pack(t)
    assert p['carrier'].dtype==torch.bfloat16 and same_bits(t,unpack(p))
    scalar=torch.tensor(-0.,dtype=torch.float32);assert same_bits(scalar,unpack(pack(scalar)))


def test_nonexact_float32_values_are_never_truncated():
    t=torch.tensor([1.00001,1.25]);p=pack(t)
    assert torch.is_tensor(p) and same_bits(t,p) and p.dtype==torch.float32


def test_nested_torch_weights_only_roundtrip_preserves_types_bits_and_metadata():
    original={'logits':torch.randn(3,16,258).bfloat16().float(),'full32':torch.randn(4),'target':torch.arange(6),
        'meta':{'path':'unchanged.pt','SHA':'abc'},'tuple':(torch.ones(1),False),'list':['id',7]}
    buf=io.BytesIO();torch.save(pack(original),buf);buf.seek(0);restored=unpack(torch.load(buf,weights_only=True))
    assert all(same_bits(original[k],restored[k]) for k in ('logits','full32','target')) and restored['meta']==original['meta']
    assert type(restored['tuple']) is tuple and type(restored['list']) is list


def test_corrupt_carrier_is_rejected():
    p=pack(torch.ones(4));p['carrier'][0]=2
    with pytest.raises(ValueError,match='bits'):unpack(p)


def test_noncontiguous_and_exact_NaN_payload_roundtrip():
    t=torch.arange(12,dtype=torch.float32).reshape(3,4).T
    assert same_bits(t,unpack(pack(t)))
    nan=torch.tensor([2143289344],dtype=torch.int32).view(torch.float32)
    assert same_bits(nan,unpack(pack(nan)))


def test_live_gradient_or_marker_collision_is_rejected():
    with pytest.raises(ValueError):pack(torch.ones(2,requires_grad=True))
    with pytest.raises(ValueError):pack({MARKER:True})
