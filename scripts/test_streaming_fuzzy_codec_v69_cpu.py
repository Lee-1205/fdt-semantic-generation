"""Dense/stream Transformer, whole-bank N/D, Commit and readout parity."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from research.streaming_fuzzy_codec_v69 import StreamingFuzzyCodecV69
from research.fuzzy_roundtrip_v64 import FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.text_diffusion_v1 import pack_prompts
import text_generation_support_v1 as support

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(69);c=Config.smoke()
    encoder=StreamingFuzzyCodecV69(c,input_capacity=512,stream_capacity=4096,compute_chunk=32)
    encoder.enable_position_adapter();encoder.enable_long_capacity();encoder.enable_local_banks()
    with torch.no_grad():encoder.core.prior.normal_(0,.02);encoder.memory_codec.local_bank_gains.fill_(.002)
    model=FuzzyRoundTripV64(encoder,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=64)).eval().requires_grad_(False)
    digest=support.state_sha(model.state_dict());text='Calder gave Vera seven blue notebooks before closing the gate. '*6
    x=pack_prompts([text],len(text.encode())+5,'cpu');kwargs=dict(source_ids=('cpu69',),source_versions=('v1',),model_epoch='same69')
    with encoder.dense_reference():
        hd,pd,vd=encoder.token_features(x);ld,cd,ed=model(x,**kwargs)
    errors=[]
    for chunk in (17,48,96,128):
        encoder.compute_chunk=chunk;hs,ps,vs=encoder.token_features(x);ls,cs,es=model(x,**kwargs)
        assert torch.equal(vd,vs) and torch.allclose(hd,hs,atol=5e-6,rtol=5e-6) and torch.allclose(pd,ps,atol=5e-6,rtol=5e-6)
        for key in ('long_N','long_D','recent_N','recent_D'):
            assert torch.allclose(getattr(ed.memories.contextual,key),getattr(es.memories.contextual,key),atol=2e-5,rtol=2e-5),key
        assert torch.allclose(cd.reader_tensor(),cs.reader_tensor(),atol=2e-5,rtol=2e-5) and torch.allclose(ld,ls,atol=2e-5,rtol=2e-5)
        assert not es.memories.exact.valid.any() and not es.memories.exact.keys.count_nonzero() and not es.memories.exact.values.count_nonzero()
        errors.append(dict(chunk=chunk,max_H_abs=float((hd-hs).abs().max()),max_Commit_abs=float((cd.reader_tensor()-cs.reader_tensor()).abs().max())))
    # Capacity extrapolation in a tiny fixture; not large-model ability.
    long='Vera kept three brass clips and Calder kept five glass beads. '*20
    z=pack_prompts([long],len(long.encode()),'cpu');model(z,source_ids=('long69',),source_versions=('v1',),model_epoch='same69')
    assert encoder.stream_stats['actual_input_bytes']>512 and not encoder.stream_stats['per_chunk_bank_reset']
    assert support.state_sha(model.state_dict())==digest and not torch.cuda.is_initialized()
    report=dict(status='STREAM_FUZZY_V69_CPU_PASS',dense_stream_parity=errors,whole_document_bank_ND_parity=True,global_pool_parity=True,
        KV_cross_chunk_continuity=True,Exact_entries0=True,all_weights_unchanged=True,fixture_beyond_declared_training_capacity=True,
        CUDA_initialized=False,large_model_GPU_performance_NOT_measured=True)
    out=ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu';out.mkdir(exist_ok=True)
    (out/'root_CPU_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
