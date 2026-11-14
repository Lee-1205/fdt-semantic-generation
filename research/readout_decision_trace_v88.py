"""Decoder-sufficient score trace, NOT lossless storage of all258 logits.

Stores the winning neural byte for every legal UTF8 state/remaining-cap case,
raw argmax and target log-probabilities. Targets affect score records only;
they cannot affect UTF8 choices. Existing full raw files are never rewritten.
"""
import torch

SCHEMA='FP32_readout_decision_trace_v88_not_full_logits'
RANGES=((0x80,0xBF),(0xA0,0xBF),(0x90,0xBF),(0x80,0x9F),(0x80,0x8F))

def masks(device):
    a=torch.arange(258,device=device);two=(a>=0xC2)&(a<=0xDF);three=(a>=0xE0)&(a<=0xEF);four=(a>=0xF0)&(a<=0xF4)
    return [((a<0x80)|(a==256)|(two&(slots>=2))|(three&(slots>=3))|(four&(slots>=4))) for slots in (1,2,3,4)]+[(a>=lo)&(a<=hi) for lo,hi in RANGES]

@torch.no_grad()
def pack(logits,labels,sample_patches):
    if logits.dtype!=torch.float32 or logits.ndim!=3 or logits.shape[1:]!=(16,258) or not torch.isfinite(logits).all():raise ValueError('finite genuine FP32 readout tensor')
    if labels.shape!=logits.shape[:2] or labels.dtype!=torch.long:raise ValueError('loss-side labels only')
    indices=torch.tensor(sample_patches,dtype=torch.long,device=logits.device)
    if len(set(sample_patches))!=len(sample_patches) or (indices<0).any() or (indices>=len(logits)).any():raise ValueError('bounded source-independent diagnostic patch indices')
    z=logits.reshape(-1,258)
    choices=torch.stack([z.masked_fill(~m,-torch.inf).argmax(-1).to(torch.int16) for m in masks(z.device)]).cpu()
    y=labels.to(z.device).flatten();gold=z.gather(-1,y[:,None]).squeeze(-1)-torch.logsumexp(z,-1)
    return dict(schema=SCHEMA,choices=choices.contiguous(),argmax=z.argmax(-1).to(torch.int16).cpu(),gold_log_probability=gold.cpu(),
        sample_patch_indices=indices.cpu(),sample_full_logits=logits[indices].cpu().clone(),patch_count=len(logits),full_logits_saved=False)

def decode(packet,max_symbols=None):
    if packet['schema']!=SCHEMA:raise ValueError('explicit decision-trace schema')
    choices=packet['choices'];n=packet['patch_count']*16;n=min(n,max_symbols) if max_symbols is not None else n
    if choices.dtype!=torch.int16 or choices.shape!=(9,packet['patch_count']*16):raise ValueError('all legal states required')
    a=choices.numpy();out=[];remaining=0;lo,hi=0x80,0xBF;EOS=False
    for i in range(n):
        state=4+RANGES.index((lo,hi)) if remaining else min(n-i,4)-1
        byte=int(a[state,i])
        if remaining:
            if not lo<=byte<=hi:raise ValueError('trace continuation legality')
        elif not (byte<128 or byte==256 or (0xC2<=byte<=0xDF and n-i>=2) or (0xE0<=byte<=0xEF and n-i>=3) or (0xF0<=byte<=0xF4 and n-i>=4)):raise ValueError('trace starter legality')
        out.append(byte)
        lo=0xA0 if byte==0xE0 else 0x90 if byte==0xF0 else 0x80
        hi=0x9F if byte==0xED else 0x8F if byte==0xF4 else 0xBF
        remaining=remaining-1 if remaining else 1 if 0xC2<=byte<=0xDF else 2 if 0xE0<=byte<=0xEF else 3 if 0xF0<=byte<=0xF4 else 0
        if byte==256:EOS=True;break
    if remaining:raise RuntimeError('cap within a codepoint')
    raw=bytes(v for v in out if v<256);out.extend([257]*(n-len(out)))
    return dict(ids=torch.tensor(out,dtype=torch.int16),text=raw.decode('utf-8'),EOS=EOS,UTF8=True,output_bytes=len(raw))
