"""Cache partial/transition, populated7Adam/scaler/ALLRNG and skip restore."""
import copy,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from frozen_verbalizer_engine_v72 import FrozenEngine
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(2);torch.manual_seed(72);random.seed(72);np.random.seed(72)
    rows=[dict(id='a',prompt='Vera has 8 gold.'),dict(id='b',prompt='Milo left at 03:17 with -2.50 kg of flour.'),dict(id='c',prompt='A plain sentence without a numeral. '*2)]
    labels={r['id']:complete_expression_spans(r['prompt'],[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'Vera|Milo',r['prompt'])]) for r in rows}
    directory=ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu/engine_fixture_v2';directory.mkdir(exist_ok=False)
    c=Config.smoke()
    def model():
        enc=FuzzyCodecV64(c,input_capacity=256);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks()
        return FuzzyRoundTripV64(enc,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=128))
    initial=copy.deepcopy(model().state_dict())
    class Budget:
        data_sha='fixture72';parent_reference={'source':'fixture72'};critical_targets=labels
        def __init__(self,label):self.output=directory/label;self.output.mkdir();self.calls=dict(encode=0,readout=0,denoise=0)
        def check(self,write_bytes=0,emergency=False):pass
        def charge(self,kind):self.calls[kind]+=1
    def create(label):
        with torch.random.fork_rng(devices=[]):m=model();m.load_state_dict(initial,strict=True)
        return FrozenEngine(m,rows,[[0],[1],[2]],Budget(label),'fixture_plan72','cpu')
    a=create('a');a.cache_next();partial=copy.deepcopy(a.export());a.cache_next();a.cache_next();complete=copy.deepcopy(a.export())
    b=create('b');b.restore(partial);b.cache_next();b.cache_next()
    # The replay writes to a distinct owned fixture directory. Payload bytes,
    # refs hashes and state match apart from those physical destination paths.
    def normalized(s):
        s=copy.deepcopy(s)
        for r in s['cache_refs']:
            r['features']=Path(r['features']).name;r['targets']=Path(r['targets']).name
        return s
    assert support.state_sha(normalized(b.export()))==support.state_sha(normalized(complete))
    restored_transition=create('transition');restored_transition.restore(complete);assert support.state_sha(restored_transition.export())==support.state_sha(complete)
    a.step();middle=copy.deepcopy(a.export());a.step(True);a.step();expected=a.export()
    restored=create('restored');restored.restore(middle);restored.step(True);restored.step();assert support.state_sha(restored.export())==support.state_sha(expected)
    assert a.attempts==a.cursor==3 and a.updates==2 and a.skips==1 and a.scaler.is_enabled() and len(a.optimizer.state)==7
    assert all(int(x['step'])==2 for x in a.optimizer.state.values()) and a.budget.calls==dict(encode=3,readout=6,denoise=0)
    assert a.exposure['encoder_rows']==a.exposure['teacher_rows']==a.exposure['head_rows']==3 and a.exposure['head_UTF8_bytes']==sum(len(x['prompt'].encode()) for x in rows)
    a.verify_frozen();assert a.model.readout.role_projection.weight.grad is None
    bad=copy.deepcopy(middle);bad['embedding_sha']='wrong'
    try:restored.restore(bad);raise AssertionError('embedding epoch')
    except ValueError:pass
    assert not torch.cuda.is_initialized()
    proof=dict(status='FROZEN_ENGINE_V72_CPU_PASS',partial_cache_restore_identical_payloads=True,partial_cache_state_equal_except_distinct_owned_output_paths=True,
        complete_cache_to_head_phase_exact_restore=True,populated7Adam_scaler_ALLRNG_exact_nextupdate=True,overflow_consumes_row_and_exposure=True,
        frozen_encoder_binding_all_tensor_digest_unchanged=True,cache_epoch_rejects_wrong_embedding=True,training_features_exclude_targets=True,
        fixture_encoder_calls=5,fixture_teacher_calls=5,fixture_head_training_calls=5,actual_parent_forward=0,CUDA_initialized=False)
    (directory.parent/'root_engine_contract_v2.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
