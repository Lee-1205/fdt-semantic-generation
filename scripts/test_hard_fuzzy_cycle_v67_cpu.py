"""Actual cycle engine populated Adam/scaler/RNG/skip/dynamic-length restore."""
import copy,json,random,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from hard_fuzzy_cycle_engine_v67 import CycleEngine
import text_generation_support_v1 as support

class Budget:
    data_sha='fixture'
    parent_reference={'path':'fixture','sha256':'fixture'}
    def __init__(self):self.calls={'encode':0,'readout':0,'denoise':0}
    def check(self):pass
    def charge(self,k):self.calls[k]+=1

def main():
    torch.set_num_threads(2);torch.manual_seed(6464);random.seed(6464);np.random.seed(6464)
    rows=[dict(id='a',prompt='A box.'),dict(id='b',prompt='한글 314159 sentence.'),dict(id='c',prompt='Longer varied sentence. '*4)]
    orders=[[0],[1],[2]]
    def create():
        c=replace(Config.smoke(),gradient_checkpointing=True);m=FuzzyCodecV64(c,input_capacity=256)
        m.enable_position_adapter();m.enable_long_capacity();m.enable_local_banks()
        return CycleEngine(FuzzyRoundTripV64(m,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=128)),rows,orders,Budget(),'fixture','cpu')
    engine=create()
    w=engine.symbol_weights;assert w[256]==1 and w[257]==0 and float(w[:256].min())>=.5 and float(w[:256].max())<=4.
    assert abs(engine.objective_report['normalization']-1)<1e-12
    z=torch.randn(1,4,258,requires_grad=True);y=torch.tensor([[65,97,49,257]])
    ce=torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='none');loss=(ce*w[y]).sum()/3
    gradient=torch.autograd.grad(loss,z)[0]
    expected=torch.softmax(z.detach(),-1);expected[0,0,65]-=1;expected[0,1,97]-=1;expected[0,2,49]-=1;expected[:,3]=0;expected*=w[y][...,None]/3
    assert torch.allclose(gradient,expected,atol=1e-7)
    from research.fuzzy_cycle_objective_v67 import cycle_losses
    fixture=torch.zeros(3,16,258,requires_grad=True);labels=torch.full((3,16),257,dtype=torch.long)
    labels.flatten()[:37]=65;labels.flatten()[37]=256
    with torch.no_grad():fixture[2,3,256]=12.;fixture[0,0,66]=10.
    total,parts=cycle_losses(fixture,labels,w)
    assert parts['hard_k']==1 and parts['tail_body_positions']==21 and parts['tail_false_EOS']==1
    g=torch.autograd.grad(parts['tail_false_EOS_UL'],fixture,retain_graph=True)[0]
    assert g[2,3,256]>0 and g[2,5].count_nonzero()==0 and g[0].count_nonzero()==0
    g=torch.autograd.grad(parts['hard_body_CE'],fixture)[0]
    assert g[2,3].abs().sum()>0 or g[0,0].abs().sum()>0
    engine.step();saved=copy.deepcopy(engine.export());engine.step(True);engine.step();expected=engine.export()
    restored=create();restored.restore(saved);restored.step(True);restored.step()
    assert support.state_sha(restored.export())==support.state_sha(expected)
    assert (engine.attempts,engine.updates,engine.skips,engine.cursor)==(3,2,1,3)
    assert engine.exposure['target_UTF8_bytes']==sum(len(r['prompt'].encode()) for r in rows)
    assert engine.scaler.is_enabled() and all(int(v['step'])==2 for v in engine.optimizer.state.values())
    assert engine.budget.calls==dict(encode=3,readout=3,denoise=0)
    engine.verify_frozen();assert not torch.cuda.is_initialized()
    report=dict(status='HARD_CYCLE_V67_CPU_PASS',populated_encoder_decoder_Adam_exact_restore=True,enabled_scaler_allRNG_exact_next_update=True,
        overflow_consumes_row_without_rewind=True,dynamic_UTF8_lengths_and_counts=True,live_encoder_no_cache=True,Exact_entries0=True,denoise0=True,CUDA_initialized=False)
    (ROOT/'research/artifacts/hard_fuzzy_cycle_20261112_v67_cpu_v2/root_engine_contract.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))

if __name__=='__main__':main()
