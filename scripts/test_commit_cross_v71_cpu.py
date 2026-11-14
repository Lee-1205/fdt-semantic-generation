"""Exact patch subset API of the shared readout; two tiny fixture calls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout,PatchCommit
from research.three_memory_codec_v33 import DifferentiableCodecCommit

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71);head=OrderedFuzzyReadout(dimension=16,value_dimension=16,hidden=24).eval();z=torch.randn(1,10,16)
        full=head.from_commit(DifferentiableCodecCommit.from_semantic(z,(('x','v','sha'),),'fixture71'))
        indices=torch.tensor([0,1,3,7]);patch=PatchCommit(z[0,indices+2],z[:,:2].expand(4,-1,-1),torch.zeros(4,dtype=torch.bool))
        selected=head.decode_patch(patch);assert torch.allclose(full[indices],selected,atol=1e-6,rtol=1e-6)
        try:head.decode_patch(z);raise AssertionError('Commit boundary')
        except TypeError:pass
    assert not torch.cuda.is_initialized()
    report=dict(status='COMMIT_CROSS_V71_FIXTURE_PASS',tiny_fixture_forward_calls=2,actual_parent_forward_calls=0,source_encoder_calls=0,
        full_commit_body_subset_equals_PatchCommit=True,Commit_type_boundary=True,no_source_or_gold_input=True,CUDA_initialized=False)
    file=ROOT/'research/artifacts/commit_cross_20261112_v71_cpu_v2/root_CPU_contract.json';assert not file.exists();file.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
