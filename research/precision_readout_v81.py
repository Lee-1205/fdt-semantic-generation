"""Commit-only readout precision control, with identical parameter names/values.

No gold labels, positions, source text, length, or memory lookup enters here.
The caller chooses a fixed runtime mode; no per-answer correctness selection.
Autograd stays available for a future frozen-D/input-gradient path. Diagnostic
workers call this under no_grad and never optimize the original parameters.
"""
import torch
from research.three_memory_codec_v33 import DifferentiableCodecCommit,SealedCodecCommit
from research.three_memory_contract_v32 import PHASE

MODES=('bf16','last_fp32','full_fp32','prefix_fp32_last_bf16')


def from_commit(head,commit,mode):
    if type(commit) not in (DifferentiableCodecCommit,SealedCodecCommit) or commit.phase!=PHASE:raise TypeError('Commit-only language boundary')
    if mode not in MODES:raise ValueError('fixed precision mode')
    z=commit.reader_tensor().to(head.terminal.device)
    if len(z)!=1 or z.shape[1]<=2 or z.shape[-1]!=head.dimension:raise ValueError('one variable-length actual Commit')
    local=torch.cat((z[0,2:],z.new_zeros(1,head.dimension)),0)
    terminal=torch.zeros(len(local),device=z.device,dtype=torch.bool);terminal[-1]=True
    global_views=z[:,:2].expand(len(local),-1,-1)
    device=z.device.type;prefix_bf16=mode in ('bf16','last_fp32');last_bf16=mode in ('bf16','prefix_fp32_last_bf16')
    with torch.autocast(device,dtype=torch.bfloat16,enabled=prefix_bf16):
        # These are exactly the original head.forward feature operations.
        value=local+terminal[:,None]*head.terminal
        feature=torch.cat((value,global_views.flatten(1)),-1)
        norm=head.norm(feature.float());hidden=head.net[:2](norm)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=last_bf16):
        logits=head.net[2](hidden if last_bf16 else hidden.float()).reshape(len(local),head.chunk,258)
    return logits,dict(mode=mode,Commit_dtype=str(z.dtype),norm_dtype=str(norm.dtype),hidden_dtype=str(hidden.dtype),logit_dtype=str(logits.dtype),
        actual_last_projection_fp32=not last_bf16,weights_or_structure_changed=False)
