"""Focused algebra, Commit boundary, enabled-scaler finite recovery checks."""
import copy,json,random,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout,fuzzy_role_reads,EOS_LOG_BIAS
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from ordered_fuzzy_engine_v55 import ReadoutEngine
import text_generation_support_v1 as support

class Budget:
    def __init__(self):
        self.calls={'encode':0,'readout':0,'denoise':0}
        self.parent_reference={'path':'fixture','sha256':'fixture'}
    def check(self):pass
    def charge(self,kind):self.calls[kind]+=1

def main():
    torch.set_num_threads(2);torch.manual_seed(55);random.seed(55);np.random.seed(55)
    mu=torch.rand(2,32,5);v=torch.randn(2,32,8);valid=torch.ones(2,32,dtype=torch.bool)
    valid[1,29:]=False
    r=fuzzy_role_reads(mu,v,valid)
    w=(mu*valid[...,None]).reshape(2,2,16,5);values=v.reshape(2,2,16,8)
    mass=w.sum(2);membership=1-(1-w).prod(2);rho=membership/membership.sum(-1,keepdim=True)
    summary=(rho[...,None,:]@((w.transpose(-1,-2)@values)/mass[...,None].clamp_min(1e-8))).squeeze(-2)*membership.amax(-1,keepdim=True)
    assert torch.allclose(r.sum(2),summary,atol=1e-6)
    rev=fuzzy_role_reads(mu.reshape(2,2,16,5).flip(2).reshape(2,32,5),values.flip(2).reshape(2,32,8),valid.reshape(2,2,16).flip(2).reshape(2,32))
    assert torch.allclose(rev,r.flip(2),atol=1e-6)
    assert torch.equal(r[1,1,13:],torch.zeros(3,8))
    model=OrderedFuzzyReadout(16,8,32);old=CommitPatchReadout(16,32)
    state=copy.deepcopy(old.state_dict());model.import_v54_head(state)
    z=torch.randn(1,4,16);commit=DifferentiableCodecCommit.from_semantic(z,(('source','v','hash'),),'epoch')
    bound=model.bind_commit(commit,r[:1])
    assert torch.equal(z,bound.reader_tensor())
    expected=old.from_commit(commit).detach();expected[:,:,256]-=EOS_LOG_BIAS
    assert torch.allclose(model.from_commit(bound),expected,atol=1e-6)
    with torch.no_grad():model.role_projection.weight.normal_()
    assert not torch.allclose(model.bind_commit(commit,r[:1]).reader_tensor(),model.bind_commit(commit,r[:1].flip(2)).reader_tensor())
    try:model.decode_patch({'raw':'answer'})
    except TypeError:pass
    else:raise AssertionError('raw bypass accepted')
    features={'local':torch.randn(9,16),'global':torch.randn(2,2,16),'doc_index':torch.tensor([0]*4+[1]*5),
        'terminal':torch.tensor([False]*3+[True]+[False]*4+[True]),'fuzzy_roles':torch.randn(9,16,8)}
    targets=torch.randint(0,256,(9,16));targets[[3,8],0]=256;orders=[[0],[1],list(range(2,9))]
    def create():return ReadoutEngine(OrderedFuzzyReadout(16,8,32),features,targets,orders,Budget(),'test','cpu')
    e=create();e.phase='readout_training';e.extract_cursor={'train':512,'dev':16};e.step()
    assert e.head.role_projection.weight.count_nonzero() and len(e.optimizer.state)==8
    saved=copy.deepcopy(e.export());e.step(True);e.step();final=e.export()
    other=create();other.restore(saved,False);other.step(True);other.step()
    assert support.state_sha(other.export())==support.state_sha(final)
    assert (other.attempts,other.updates,other.skips,other.cursor)==(3,2,1,9)
    assert other.scaler.is_enabled() and not torch.cuda.is_initialized()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'cache';path.write_bytes(b'original');tampered=copy.deepcopy(saved)
        tampered['cache_refs']={str(path):support.sha(path)};path.write_bytes(b'changed')
        try:create().restore(tampered)
        except ValueError:pass
        else:raise AssertionError('tampered cache accepted')
    report=dict(fuzzy_ND_conservation=True,joint_permutation_reorders_roles=True,padded_roles_zero=True,
        zero_init_Commit_exact=True,TRAIN_only_EOS_correction_exact=True,learned_role_order_sensitive=True,
        raw_boundary_rejected=True,projection_head_gradients=True,enabled_scaler_next_update_exact=True,
        AMP_skip_consumes_cursor=True,last_seven=True,phase_cache_RNG_recovery_exact=True,cache_tamper_rejected=True,
        CUDA_initialized=False)
    artifact=ROOT/'research/artifacts/ordered_fuzzy_20261111_v55_cpu';artifact.mkdir(exist_ok=True)
    (artifact/'root_CPU_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
