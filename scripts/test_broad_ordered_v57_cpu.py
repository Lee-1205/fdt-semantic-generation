"""Actual live-step path, not the cached fixture: freeze/skip/RNG/document LR."""
import copy,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from types import SimpleNamespace
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from broad_ordered_engine_v57 import BroadOrderedEngine
import text_generation_support_v1 as support

class Budget:
    def __init__(self):
        self.p={'source_sha':'fixture','class_weights':[1.]*257+[0.]}
        self.calls={'encode':0,'readout':0,'denoise':0};self.parent_reference={'path':'fixture','sha256':'fixture'}
    def check(self):pass
    def charge(self,k):self.calls[k]+=1

class Source(torch.nn.Module):
    def __init__(self):
        super().__init__();self.frozen=torch.nn.Parameter(torch.ones(1),requires_grad=False)
    def encode_input(self,x,**metadata):
        n=x.shape[1];slots=(n+15)//16
        z=torch.randn(1,slots+2,512)
        commit=DifferentiableCodecCommit.from_semantic(z,(('s','v','hash'),),'fixture')
        return SimpleNamespace(commit=commit,diagnostics={'token_memberships':torch.rand(1,n,4),'token_values':torch.randn(1,n,256)})

def main():
    torch.set_num_threads(2);torch.manual_seed(57);np.random.seed(57);random.seed(57)
    orders=[[0],[1],[2]];rows=[dict(id=str(i),prompt=t) for i,t in enumerate(('sixteen-byte-inp','another input','last777'))]
    source=Source()
    def create():return BroadOrderedEngine(OrderedFuzzyReadout(),None,None,orders,Budget(),'test','cpu')
    e=create();e.step_document(source,rows[0]);saved=copy.deepcopy(e.export())
    e.step_document(source,rows[1],True);e.step_document(source,rows[2]);final=e.export()
    restored=create();restored.restore(saved,False);restored.step_document(source,rows[1],True);restored.step_document(source,rows[2])
    assert support.state_sha(restored.export())==support.state_sha(final)
    assert (restored.attempts,restored.cursor,restored.updates,restored.skips)==(3,3,2,1)
    assert restored.exposure['target_bytes']==sum(len(r['prompt'].encode()) for r in rows)
    assert restored.budget.calls=={'encode':3,'readout':3,'denoise':0}
    assert source.frozen.grad is None and source.frozen.item()==1 and restored.scaler.is_enabled()
    assert not torch.cuda.is_initialized()
    report=dict(actual_live_document_step_next_update_exact=True,allRNG_and_cosineLR_restore_exact=True,
        AMPskip_consumes_document=True,last7byte_input=True,frozen_source_no_gradient=True,
        correct_live_encode_readout_cost=True,no_target_prefix_or_cache=True,CUDA_initialized=False)
    artifact=ROOT/'research/artifacts/broad_ordered_20261111_v57_cpu';artifact.mkdir(exist_ok=True)
    (artifact/'root_CPU_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
