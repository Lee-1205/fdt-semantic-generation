"""Nonzero prior fixture, initial parity and isolated local-bank gradients."""
import json,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.trainable_long_codec_v50 import TrainableLongCodec
from research.local_three_bank_codec_v52 import LocalThreeBankCodec
from research.text_diffusion_v1 import pack_prompts,pack_answers

def main():
    torch.set_num_threads(2);torch.manual_seed(52);c=Config.smoke()
    p=TrainableLongCodec(c,input_capacity=256);p.enable_position_adapter();p.enable_long_capacity()
    with torch.no_grad():
        p.core.prior.normal_(0,.02)
        p.decoder.local_read_gain.fill_(.002)
    m=LocalThreeBankCodec(c,input_capacity=256);m.enable_position_adapter();m.enable_long_capacity()
    m.load_state_dict(p.state_dict(),strict=True);m.enable_local_banks()
    text='A 2026 한글 긴 문장. '*8
    x=pack_prompts([text],len(text.encode()),'cpu');y=pack_answers([text],len(text.encode())+1,'cpu')
    k=dict(source_ids=('one',),source_versions=('v1',),model_epoch='cpu')
    p.eval();m.eval();a=p.encode_input(x,**k);b=m.encode_input(x,**k)
    assert torch.equal(a.commit.reader_tensor(),b.commit.reader_tensor())
    assert torch.equal(p.teacher_logits(a.commit,y),m.teacher_logits(b.commit,y))
    m.train();loss,_=m.reconstruction_loss(x,y,**k);loss.backward()
    g=m.memory_codec.local_bank_gains.grad.clone();assert (g.abs()>0).all(),g
    with torch.no_grad():m.memory_codec.local_bank_gains.fill_(.002)
    m.eval();normal=m.encode_input(x,**k)
    withoutP=m.encode_input(x,**k,disabled_memories=('persistent',))
    withoutE=m.encode_input(x,**k,disabled_memories=('exact',))
    assert not torch.equal(normal.commit.reader_tensor()[:,2:],withoutP.commit.reader_tensor()[:,2:])
    assert not torch.equal(normal.commit.reader_tensor()[:,2:],withoutE.commit.reader_tensor()[:,2:])
    assert (withoutP.diagnostics['local_prior_norm']==0).all() and (withoutE.diagnostics['local_exact_norm']==0).all()
    try:m.encode_input(x,source_ids=('',),source_versions=('v1',),model_epoch='cpu')
    except ValueError:pass
    else:raise AssertionError('empty identity accepted')
    result=dict(zero_gain_exact_parent_parity=True,both_new_gain_gradients=g.tolist(),
        local_KP_and_KE_isolation=True,nonzero_prior_fixture=True,empty_source_rejected=True,CUDA_initialized=False)
    a=ROOT/'research/artifacts/local_three_bank_20261111_v52_cpu';a.mkdir(exist_ok=True)
    (a/'root_candidate_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
