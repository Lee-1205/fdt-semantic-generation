import copy,sys
from pathlib import Path
import pytest,torch
from research.answer_pair_contrast_v112 import PairContrastEngine,contrast_error
from research.latent_identity_thought_v110 import LatentPolicy
from research.answer_meaning_thought_v111 import ConditionalAnswerObjective,answer_curriculum
from test_latent_identity_v110 import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from text_generation_support_v1 import state_sha


def setup():
    old,m,p=fixture();units=[dict(ids=['a','b'],real_pair=True,key='actual'),dict(ids=['c','d'],real_pair=False,key='singles')]
    e=PairContrastEngine(old.thought,old.codec,'SHA',['a','b','c','d'],times=[1.,1.,.5,.5],batch=4,
        units=units,policy=LatentPolicy(steps=4))
    q=dict(p,semantic_target=p['semantic_target']+.3)
    return e,m,p,q


def test_derived_contrast_doubles_antisymmetric_coefficient_and_gradients():
    a=torch.randn(2,5,requires_grad=True);b=torch.randn(2,5,requires_grad=True)
    y=torch.randn(2,5);w=torch.randn(2,5);ea=a-y;eb=b-w
    original=(ea.square().mean()+eb.square().mean())/2
    delta=(ea-eb).square().mean();mean=((ea+eb)/2).square().mean()
    assert torch.allclose(original,mean+.25*delta,atol=1e-6)
    enhanced=original+.25*delta
    assert torch.allclose(enhanced,mean+.5*delta,atol=1e-6)
    ga,gb=torch.autograd.grad(enhanced,(a,b))
    assert torch.allclose(ga,(ea+.5*(ea-eb))/a.numel(),atol=1e-6)
    assert torch.allclose(gb,(eb-.5*(ea-eb))/b.numel(),atol=1e-6)


def test_atomic_midgroup_and_READY_resume_scales_RNG_Adam_exact():
    e,m,p,q=setup();frozen=state_sha(e.codec.state_dict());e.begin();e.unit(['a','b'],[m,m],[p,q])
    mid=e.payload();mid_sha=state_sha(mid);assert mid['runtime']['cursor']==2
    e.unit(['c','d'],[m,m],[p,p]);ready=e.payload();e.finish();expected=state_sha(e.payload())
    e.restore(mid);e.unit(['c','d'],[m,m],[p,p]);e.finish()
    assert state_sha(e.payload())==expected and state_sha(mid)==mid_sha
    e.restore(ready);e.finish();assert state_sha(e.payload())==expected
    assert e.codec.calls==0 and state_sha(e.codec.state_dict())==frozen
    assert e.runtime['calls']==dict(denoise=20,readout=0)
    assert e.runtime['history'][0]['unit_RNG_draws']==3


def test_actualpair_same_noise_singles_independent_and_no_fake_contrast():
    e,m,p,q=setup();seen=[];old=e.objective
    def observe(m,p,t,n,charge):seen.append(n.clone());return old(m,p,t,n,charge)
    e.objective=observe;e.begin();e.unit(['a','b'],[m,m],[p,q]);e.unit(['c','d'],[m,m],[p,p])
    assert torch.equal(seen[0],seen[1]) and not torch.equal(seen[2],seen[3])
    assert all(x['pair_contrast_MSE']==0 for x in e.runtime['pending_losses'][2:])
    assert contrast_error(p['semantic_target'],q['semantic_target'],p,q)==0
    with pytest.raises(RuntimeError):e.micro('a',m,p)


def test_partial_failure_restores_gradients_consumes_whole_unit_and_never_replays():
    e,m,p,q=setup();e.begin();e.unit(['a','b'],[m,m],[p,q]);grads=state_sha(e._gradients());before=e.runtime['cursor']
    def stop(kind):raise OSError('injected call failure')
    with pytest.raises(OSError):e.unit(['c','d'],[m,m],[p,p],stop)
    assert e.runtime['phase']=='ABORTED' and e.runtime['cursor']==before+2 and state_sha(e._gradients())==grads
    with pytest.raises(ValueError):e.unit(['c','d'],[m,m],[p,p])


def test_AMPskip_consumes_rows_and_rejects_other_objective_resume():
    e,m,p,q=setup();e.begin();e.unit(['a','b'],[m,m],[p,q]);e.unit(['c','d'],[m,m],[p,p]);ready=e.payload()
    e.finish(force_overflow=True)
    assert e.runtime['AMP_skips']==1 and e.runtime['cursor']==4 and e.runtime['updates']==0
    with pytest.raises(ValueError):e.restore(dict(ready,pair_coefficient=1.))
    e.thought.eval()
    with pytest.raises(RuntimeError):e.objective(m,p,torch.ones(1),torch.randn(1,7,16),lambda kind:None)


def test_endpoint_and_auxiliary_math_unchanged_before_added_difference_loss():
    e,m,p,q=setup();noise=torch.randn(1,7,16);t=torch.ones(1)
    old=ConditionalAnswerObjective(e.thought,e.update,e.codec_SHA,e.latent_policy)
    want,metrics=old(m,p,t,noise);actual,checked,z=e.objective(m,p,t,noise,lambda kind:None)
    assert torch.allclose(want,actual,atol=1e-6)
    for key in ('runtime_body_MSE','runtime_tail_MSE','aux_VP'):assert abs(checked[key]-metrics[key])<1e-6


def test_actual_selected134_each16_epochs_pair_boundaries_and1216_draws():
    from test_answer_meaning_v111 import selected_rows
    rows=selected_rows();byid={r['id']:r for r in rows};orders=answer_curriculum(rows)
    flat=[r for ep in orders for r in ep];draws=0
    for j in range(0,len(flat),2):
        a,b=flat[j:j+2];same=byid[a['id']]['pair_key']==byid[b['id']]['pair_key']
        assert same==a['real_reverse_pair']==b['real_reverse_pair']
        if same:assert j//8==(j+1)//8 and a['t']==b['t']
        draws+=1 if same else 2
    assert draws==1216 and len(flat)==2144
