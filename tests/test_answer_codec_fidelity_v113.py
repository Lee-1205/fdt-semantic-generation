import sys
from pathlib import Path
import torch,pytest
from torch import nn
from research.answer_codec_fidelity_v113 import PairCodecEngine
from research.latent_identity_thought_v110 import LatentPolicy
from test_latent_identity_v110 import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from text_generation_support_v1 import state_sha


def setup():
    old,m,p=fixture()
    class Codec(nn.Module):
        def __init__(self):super().__init__();self.head=nn.Linear(16,16*258);self.calls=0
        def verbalize(self,commit):
            self.calls+=1;return self.head(commit.reader_tensor()[:,1:])[0].reshape(6,16,258)
    codec=Codec();u=[dict(ids=['a','b'],real_pair=True,key='pair')]
    e=PairCodecEngine(old.thought,codec,'SHA',['a','b'],times=[1.,1.],units=u,batch=2,policy=LatentPolicy(steps=4))
    return e,m,p


def test_CE_differentiates_to_thought_but_all_D_parameters_remain_frozen():
    e,m,p=setup();before=state_sha(e.codec.state_dict());n=torch.randn(1,7,16)
    loss,metrics,z=e.objective(m,p,torch.ones(1),n,lambda kind:None)
    g=torch.autograd.grad(e.objective.last_byte_CE,tuple(e.thought.parameters()),allow_unused=True)
    assert sum(float(x.square().sum()) for x in g if x is not None)>0
    assert all(not x.requires_grad and x.grad is None for x in e.codec.parameters())
    assert state_sha(e.codec.state_dict())==before and not metrics['EOS_auxiliary_used']


def test_actual_update_owns_theta_only_and_fidelity_READY_restore_exact():
    e,m,p=setup();before=state_sha(e.codec.state_dict());e.begin();e.unit(['a','b'],[m,m],[p,p]);ready=e.payload();readySHA=state_sha(ready)
    e.finish();want=state_sha(e.payload());e.restore(ready);e.finish()
    assert state_sha(e.payload())==want and state_sha(ready)==readySHA and state_sha(e.codec.state_dict())==before
    assert e.runtime['calls']==dict(denoise=10,readout=2)
    assert e.runtime['history'][0]['D_during_training'] is True
    with pytest.raises(ValueError):e.restore(dict(ready,language_losses_used=False))


def test_inference_signature_remains_target_free_and_loss_packet_eval_forbidden():
    e,m,p=setup();e.thought.eval();n=torch.randn(1,7,16)
    a=e.thought.sample(m,n,4).semantic;b=e.thought.sample(m,n,4).semantic
    assert torch.equal(a,b)
    with pytest.raises(TypeError):e.thought.sample(m,n,4,target=p)
    with pytest.raises(RuntimeError):e.objective(m,p,torch.ones(1),n,lambda kind:None)
