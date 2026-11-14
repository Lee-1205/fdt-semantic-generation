"""Focused numerical restoration, overflow consumption and long cached-decoder tests."""
import copy,io,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import numpy_state
from research.position_sensitive_codec_v46 import PositionSensitiveCodec
from research.trainable_long_codec_v50 import TrainableLongCodec
from research.text_diffusion_v1 import pack_prompts,pack_answers
from long_codec_engine_v50 import LongEngine
import text_generation_support_v1 as support

class Budget:
    data_sha='test';calls={}
    def __init__(self):self.calls={'encode':0,'decoder':0,'denoise':0}
    def check(self):pass
    def charge(self,kind):self.calls[kind]+=1

def main():
    torch.set_num_threads(2);torch.manual_seed(50);random.seed(50);np.random.seed(50)
    c=Config.smoke();old=PositionSensitiveCodec(c);old.enable_position_adapter()
    names=[n for n,p in old.named_parameters() if p.requires_grad]
    groups=[names[i::4] for i in range(4)]
    named=dict(old.named_parameters());optimizer=torch.optim.AdamW([{'params':[named[n] for n in g],'lr':1e-5} for g in groups])
    for p in old.parameters():
        if p.requires_grad:p.grad=torch.full_like(p,.001)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    source=dict(model=copy.deepcopy(old.state_dict()),optimizer=optimizer.state_dict(),optimizer_parameter_names=groups,
        scaler=torch.amp.GradScaler('cpu',init_scale=1024.).state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=[],
        python_rng=random.getstate(),numpy_rng=numpy_state(),plan_sha256='parent',attempts=1,updates=1,cursor=1,
        history=[],actual_calls={},evaluations={})
    rows=[dict(id=str(i),prompt=('A 2026 한글. '*20)[:100]) for i in range(9)]
    orders=[[0],[1],list(range(2,9))]
    def create():
        model=TrainableLongCodec(c,input_capacity=256);model.enable_position_adapter()
        model.load_state_dict(source['model'],strict=True);model.enable_long_capacity()
        engine=LongEngine(model,groups,rows,orders,Budget(),'test','cpu');engine.import_parent(source)
        assert torch.equal(model.decoder.positions.weight,old.decoder.positions.weight)
        return engine
    e=create();e.step();state=copy.deepcopy(e.export())
    assert e.scaler.is_enabled()
    e.step(force_overflow=True);after_skip=copy.deepcopy(e.export());e.step();final=e.export()
    assert (e.attempts,e.updates,e.skips,e.cursor)==(3,2,1,9)
    r=create();r.restore(state);r.step(force_overflow=True)
    assert support.state_sha(r.export())==support.state_sha(after_skip)
    r.step();assert support.state_sha(r.export())==support.state_sha(final)
    model=r.model.eval();text=rows[0]['prompt'];x=pack_prompts([text],len(text.encode()),'cpu')
    ep=model.encode_input(x,source_ids=('a',),source_versions=('a',),model_epoch='test')
    context=model.context(ep.commit);prefix=torch.randint(0,256,(1,160));prefix[:,0]=258
    full=model.decoder(prefix,context);cache=model.decoder.start_cache(context)
    cached=torch.cat([model.decoder.cached_step(prefix[:,i:i+1],cache) for i in range(160)],1)
    torch.testing.assert_close(cached,full,atol=3e-5,rtol=3e-5)
    assert not torch.cuda.is_initialized()
    # Independently re-derive every new row from its original split and UTF8 byte interval.
    artifact=ROOT/'research/artifacts/long_codec_20261111_v50'
    docs=json.loads((ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final/documents.json').read_text())
    for split in ('train','dev'):
        by_id={d['id']:d for d in docs[split]}
        for row in json.loads((artifact/(split+'.json')).read_text()):
            original=by_id[row['document_id']]['text'].encode()
            assert row['prompt'].encode()==original[row['source_byte_start']:row['source_byte_end']]
            assert row['actual_bytes']<=row['requested_bytes']<=16384
    report=dict(enabled_scaler_exact_recovery=True,overflow_consumes_row_and_attempt=True,last_seven_rows=True,
        full_RNG_Adam_next_update_exact=True,cached_decoder_over129_max_error=float((full-cached).abs().max()),
        old129positions_preserved=True,data_provenance_rows=448,CUDA_initialized=False)
    (artifact/'root_cpu_contract.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
