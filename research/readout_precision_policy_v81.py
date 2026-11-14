"""Deterministic corrected output arithmetic, with unchanged codec weights.

Only the neural Commit crosses this boundary. No per-row gold, textual repair,
Exact lookup, source prefix/length, or answer-dependent precision choice.
The whole readout (including prefix) uses FP32; input extraction is untouched.
"""
from research.precision_readout_v81 import from_commit

POLICY='fixed_full_FP32_Commit_readout_no_Gold_no_Exact_v81'

def verbalize(model,commit):
    return from_commit(model.readout.readout,commit,'full_fp32')[0]

def readout(head,commit):
    return from_commit(head,commit,'full_fp32')[0]
