"""Optional storage-only BF16 carrier for EXACTLY representable FP32 values.

Not quantization of full-precision long logits. Float32 tensors with any
nonzero low16 bits stay float32. Unpack restores original float32 bits before
NLL/FSM or inference consumers. No old file is rewritten and no NN is called.
Storage sharing/strides are not preserved; this is for raw evidence, not models.
"""
import hashlib
import torch
MARKER='__fdt_lossless_float32_v1__'


def bit_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def pack(value):
    if torch.is_tensor(value):
        if value.requires_grad:raise ValueError('raw evidence must be detached')
        if value.dtype==torch.float32 and bool(((value.contiguous().view(torch.int32)&65535)==0).all()):
            carrier=value.to(torch.bfloat16)
            # Bitwise verification includes negative zero and NaN payloads.
            if not torch.equal(carrier.float().contiguous().view(torch.int32),value.contiguous().view(torch.int32)):return value
            return {MARKER:True,'shape':list(value.shape),'carrier':carrier,'float32_bits_SHA':bit_sha(value)}
        return value
    if isinstance(value,dict):
        if MARKER in value:raise ValueError('reserved raw storage marker collision')
        return {k:pack(v) for k,v in value.items()}
    if isinstance(value,list):return [pack(v) for v in value]
    if isinstance(value,tuple):return tuple(pack(v) for v in value)
    return value


def unpack(value):
    if isinstance(value,dict) and MARKER in value:
        if set(value)!={MARKER,'shape','carrier','float32_bits_SHA'} or value[MARKER] is not True:raise ValueError('raw carrier schema')
        t=value['carrier']
        if not torch.is_tensor(t) or t.dtype!=torch.bfloat16 or list(t.shape)!=value['shape']:raise ValueError('raw carrier dtype/shape')
        restored=t.float()
        if bit_sha(restored)!=value['float32_bits_SHA']:raise ValueError('raw carrier bits changed')
        return restored
    if isinstance(value,dict):return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):return [unpack(v) for v in value]
    if isinstance(value,tuple):return tuple(unpack(v) for v in value)
    return value
