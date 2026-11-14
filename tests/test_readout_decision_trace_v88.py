import pytest,torch
from research.readout_decision_trace_v88 import pack,decode
from research.commit_patch_readout_v54 import decode_fixed_logits

def compare(z,y):
    trace=pack(z,y,[0]);a=decode(trace);b=decode_fixed_logits(z,z.shape[0]*16)
    assert a['text']==b['text'] and a['ids'].tolist()==b['ids'] and a['EOS']==b['EOS'];return trace,a

def test_trace_reproduces_UTF8_states_and_raw_argmax_is_not_decoded_text():
    text='A€𐍈😀Z';raw=text.encode();z=torch.zeros(2,16,258);y=torch.full((2,16),257,dtype=torch.long);y.flatten()[:len(raw)]=torch.tensor(list(raw));y.flatten()[len(raw)]=256
    z.flatten(0,1).scatter_(1,y.flatten()[:,None],10.);z[:,:,0xC0]=20.;z[:,:,257]=30.
    t,a=compare(z,y);assert a['text']==text and a['EOS'] and int(t['argmax'][0])==257

def test_all_remaining_cap_variants_and_no_gold_effect_on_choices():
    torch.manual_seed(88)
    for patches in (1,2,5):
        z=torch.randn(patches,16,258);y=torch.zeros(patches,16,dtype=torch.long);t,a=compare(z,y);other=pack(z,torch.ones_like(y),[0])
        assert torch.equal(t['choices'],other['choices']) and not torch.equal(t['gold_log_probability'],other['gold_log_probability'])

def test_rejects_rounded_or_malformed_scores_and_false_lossless_claim():
    z=torch.zeros(1,16,258);y=torch.zeros(1,16,dtype=torch.long)
    with pytest.raises(ValueError):pack(z.bfloat16(),y,[0])
    with pytest.raises(ValueError):pack(z,y,[1])
    assert pack(z,y,[0])['full_logits_saved'] is False
