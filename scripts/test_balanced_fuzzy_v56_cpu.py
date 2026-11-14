"""Weighted training and Commit-topology stop have no gold inference features."""
import copy,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.codec_commit_stop_v56 import guard_codec_stop
from balanced_fuzzy_engine_v56 import ReadoutEngine
from run_balanced_fuzzy_20261111_v56 import add_early
import text_generation_support_v1 as support

class Budget:
    def __init__(self):
        self.p={'class_weights':[2. if 48<=i<=57 or 65<=i<=90 else 0. if i==257 else 1. for i in range(258)]}
        self.calls={'encode':0,'readout':0,'denoise':0};self.parent_reference={'path':'fixture','sha256':'fixture'}
    def check(self):pass
    def charge(self,k):self.calls[k]+=1

def main():
    torch.set_num_threads(2);torch.manual_seed(56);np.random.seed(56);random.seed(56)
    f=add_early(dict(local=torch.randn(9,16),global_views=torch.randn(2,2,16),
        doc_index=torch.tensor([0]*4+[1]*5),terminal=torch.tensor([False]*3+[True]+[False]*4+[True]),
        fuzzy_roles=torch.randn(9,16,8),offsets=[[0,4],[4,9]]))
    f['global']=f.pop('global_views');target=torch.randint(0,256,(9,16));target[3,0]=256;target[3,1:]=257
    orders=[[0],[1],list(range(2,9))]
    def create():return ReadoutEngine(OrderedFuzzyReadout(16,8,32),f,target,orders,Budget(),'test','cpu')
    e=create();e.step();saved=copy.deepcopy(e.export());e.step(True);e.step();final=e.export()
    other=create();other.restore(saved,False);other.step(True);other.step()
    assert support.state_sha(other.export())==support.state_sha(final)
    assert (other.cursor,other.attempts,other.updates,other.skips)==(9,3,2,1)
    assert torch.equal(f['early'],torch.tensor([True,True,False,False,True,True,False,False,False]))
    logits=torch.zeros(4,16,258);logits[:,:,256]=10
    guarded=guard_codec_stop(logits)
    assert not torch.isfinite(guarded[:2,:,256]).any() and torch.equal(logits[2:],guarded[2:])
    assert torch.equal(guarded[:,:,:256],logits[:,:,:256]) and (logits[:2,:,256]==10).all()
    wrong=copy.deepcopy(saved);wrong['loss_weights'][0]=99
    try:create().restore(wrong,False)
    except ValueError:pass
    else:raise AssertionError('wrong recovery objective accepted')
    assert e.scaler.is_enabled() and not torch.cuda.is_initialized()
    report=dict(weighted_PAD_zero=True,first32_Commit_position_only=True,weighted_enabled_scaler_next_update_exact=True,
        AMP_skip_and_last7_exact=True,changed_objective_restore_rejected=True,
        stop_does_not_change_bytes_or_force_EOS=True,raw_scores_preserved=True,CUDA_initialized=False)
    artifact=ROOT/'research/artifacts/balanced_fuzzy_20261111_v56_cpu';artifact.mkdir(exist_ok=True)
    (artifact/'root_CPU_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
