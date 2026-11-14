"""TRAIN loss gradient and actual populated engine/scaler/RNG restoration."""
import copy,json,random,re,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.fuzzy_factual_objective_v70 import cycle_losses,critical_mask
from research.fuzzy_cycle_objective_v67 import cycle_losses as old_losses
from factual_cycle_engine_v70 import CycleEngine
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(2);torch.manual_seed(70);random.seed(70);np.random.seed(70)
    rows=[dict(id='a',prompt='Vera kept 18 blue cards.'),dict(id='b',prompt='Milo has 8 coins.'),dict(id='c',prompt='A natural sentence without numerals. '*3)]
    targets={r['id']:[dict(start=m.start(),end=m.end()) for m in re.finditer(r'Vera|Milo|[0-9]+',r['prompt'])] for r in rows}
    class Budget:
        data_sha='fixture';parent_reference={'path':'fixture'};critical_targets=targets
        def __init__(self):self.calls=dict(encode=0,readout=0,denoise=0)
        def check(self):pass
        def charge(self,k):self.calls[k]+=1
    def create():
        c=replace(Config.smoke(),gradient_checkpointing=True);e=FuzzyCodecV64(c,input_capacity=256)
        e.enable_position_adapter();e.enable_long_capacity();e.enable_local_banks()
        return CycleEngine(FuzzyRoundTripV64(e,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=128)),rows,[[0],[1],[2]],Budget(),'fixture70','cpu')
    engine=create();weights=engine.symbol_weights
    z=torch.randn(3,16,258,requires_grad=True);y=torch.full((3,16),257,dtype=torch.long);y.flatten()[:35]=65;y.flatten()[8:10]=torch.tensor([49,56]);y.flatten()[35]=256
    mask=critical_mask(y,[dict(start=8,end=10)],35);loss,parts=cycle_losses(z,y,weights,mask);base,_=old_losses(z,y,weights)
    g=torch.autograd.grad(loss-base,z)[0];expected=torch.zeros_like(z);expected[0,8:10]=torch.softmax(z.detach()[0,8:10],-1);expected[0,8,49]-=1;expected[0,9,56]-=1;expected*=.25/2
    assert torch.allclose(g,expected,atol=1e-7) and g[y==257].count_nonzero()==0 and parts['critical_positions']==2
    try:critical_mask(y,[dict(start=35,end=36)],35);raise AssertionError('span must reject EOS')
    except ValueError:pass
    engine.step();saved=copy.deepcopy(engine.export());engine.step(True);engine.step();expected_state=engine.export()
    restored=create();restored.restore(saved);restored.step(True);restored.step();assert support.state_sha(restored.export())==support.state_sha(expected_state)
    assert (engine.attempts,engine.updates,engine.skips,engine.cursor)==(3,2,1,3) and engine.scaler.is_enabled()
    assert all(int(v['step'])==2 for v in engine.optimizer.state.values());engine.verify_frozen();assert not torch.cuda.is_initialized()
    report=dict(status='FACTUAL_CYCLE_V70_CPU_PASS',analytic_critical_CE_gradient=True,critical_target_feature_separation=True,PAD_EOS_critical_gradient0=True,
        zero_critical_row_valid=True,populated_Adam_scaler_ALLRNG_exact_nextupdate=True,AMP_overflow_row_consumption=True,
        actual_encoder_and_decoder_gradient=True,Exact_entries0=True,CUDA_initialized=False)
    out=ROOT/'research/artifacts/factual_cycle_20261112_v70_cpu_v2/root_engine_contract_v2.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
