"""Commit-only boundary, finite overflow/cursor and exact learned-head recovery."""
import copy,io,json,random,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from commit_readout_engine_v54 import ReadoutEngine
import text_generation_support_v1 as support

class Budget:
    def __init__(self):self.calls={'encode':0,'readout':0};self.parent_reference={'path':'fixture','sha256':'fixture'}
    def check(self):pass
    def charge(self,kind):self.calls[kind]+=1

def main():
    torch.set_num_threads(2);torch.manual_seed(54);random.seed(54);np.random.seed(54)
    features=dict(local=torch.randn(9,8),global_views=torch.randn(2,2,8),doc_index=torch.tensor([0]*4+[1]*5),terminal=torch.tensor([False]*3+[True]+[False]*4+[True]))
    features['global']=features.pop('global_views');targets=torch.randint(0,256,(9,16));targets[[3,8],0]=256
    orders=[[0],[1],list(range(2,9))]
    def create():return ReadoutEngine(CommitPatchReadout(8,16),features,targets,orders,Budget(),'test','cpu')
    engine=create();engine.step();saved=copy.deepcopy(engine.export());engine.step(True);engine.step();final=engine.export()
    restored=create();restored.restore(saved,False);restored.step(True);restored.step()
    assert support.state_sha(restored.export())==support.state_sha(final)
    assert (restored.attempts,restored.updates,restored.skips,restored.cursor)==(3,2,1,9)
    commit=DifferentiableCodecCommit.from_semantic(torch.randn(1,4,8),(('source','v1','hash'),),'epoch')
    logits=engine.head.from_commit(commit);assert logits.shape==(3,16,258)
    try:engine.head.from_commit({'source':'bytes'})
    except TypeError:pass
    else:raise AssertionError('raw object accepted')
    assert engine.scaler.is_enabled() and not torch.cuda.is_initialized()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'cache';path.write_bytes(b'original')
        state=copy.deepcopy(saved);state['cache_refs']={str(path):support.sha(path)}
        path.write_bytes(b'changed')
        try:create().restore(state)
        except ValueError:pass
        else:raise AssertionError('tampered cache accepted')
    report=dict(enabled_scaler_next_update_exact=True,AMP_skip_consumes_cursor=True,last_seven=True,
        raw_object_rejected=True,Commit_shape_only=True,cache_tamper_rejected=True,CUDA_initialized=False)
    artifact=ROOT/'research/artifacts/commit_readout_20261111_v54_cpu'
    (artifact/'root_CPU_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
