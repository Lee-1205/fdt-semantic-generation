import hashlib,io,json,sys
from pathlib import Path
import pytest
import torch
from research.answer_meaning_thought_v111 import answer_curriculum,AnswerMeaningEngine
from research.latent_identity_thought_v110 import LatentPolicy
from test_latent_identity_v110 import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from text_generation_support_v1 import state_sha

P=Path(__file__).resolve().parents[1]


def selected_rows():
    a=P/'research/artifacts/answer_meaning_objective_20261114_v1'
    ids={r['id'] for r in json.loads((a/'train_conditions.json').read_text())}
    allrows=json.loads((P/'research/artifacts/memory_thought_data_20261113_v95_cpu_v1/train_targets.json').read_text())
    return [r for r in allrows if r['id'] in ids]


def test_actual134_pair_singleton_curriculum_and_everytime_once():
    rows=selected_rows();order=answer_curriculum(rows);assert order==answer_curriculum(rows)
    ids={r['id'] for r in rows};byid={r['id']:r for r in rows};flat=[r for ep in order for r in ep]
    assert len(flat)==2144 and len(flat)//8==268
    assert sum(r['real_reverse_pair'] for r in order[0])==116
    for ep in order:
        assert len(ep)==134 and {r['id'] for r in ep}==ids
    for j in range(0,len(flat),2):
        a,b=flat[j:j+2]
        assert a['real_reverse_pair']==b['real_reverse_pair']
        if a['real_reverse_pair']:assert byid[a['id']]['pair_key']==byid[b['id']]['pair_key'] and a['t']==b['t']
    for i in ids:assert sorted(r['t'] for r in flat if r['id']==i)==sorted(1-k/16 for k in range(16))


def test_actual_manifest_maps_input_to_original_answer_NOT_prompt():
    a=P/'research/artifacts/answer_meaning_objective_20261114_v1'
    loss={r['id']:r for r in json.loads((a/'train_targets_LOSS_ONLY.json').read_text())}
    source=json.loads((a/'train_conditions.json').read_text());rows={r['id']:r for r in selected_rows()}
    assert len(loss)==134 and len({v['answer_SHA'] for v in loss.values()})==27
    for s in source:
        assert set(s)=={'id','input_condition'}
        r=rows[s['id']];target=loss[s['id']]
        assert target['answer_text_offline_only']==r['answer']
        assert target['answer_SHA']==hashlib.sha256(r['answer'].encode()).hexdigest()
        assert target['answer_meaning_loss_only']['path']!=s['input_condition']['path']


def test_answer_loss_gradient_decoder_forbidden_new_schema_exact_READY():
    prior,m,p=fixture();engine=AnswerMeaningEngine(prior.thought,prior.codec,'SHA',['a','b'],times=[1.,.5],batch=2,policy=LatentPolicy(steps=4))
    codec_before=state_sha(engine.codec.state_dict());theta_before=state_sha(engine.thought.state_dict())
    engine.begin();engine.micro('a',m,p);engine.micro('b',m,p)
    f=io.BytesIO();torch.save(engine.payload(),f);f.seek(0);saved=torch.load(f,weights_only=True);original=state_sha(saved)
    assert saved['schema']=='conditional_answer_meaning_fullstate_v111'
    assert any(g is not None and g.count_nonzero() for g in saved['gradients'].values())
    engine.finish();expected=state_sha(engine.payload());engine.restore(saved);engine.finish()
    assert state_sha(engine.payload())==expected and state_sha(saved)==original
    assert state_sha(engine.codec.state_dict())==codec_before and state_sha(engine.thought.state_dict())!=theta_before
    assert engine.codec.calls==0 and engine.runtime['calls']['readout']==0
    with pytest.raises(ValueError):engine.restore(dict(saved,schema='latent_identity_fullstate_v110'))


def test_target_swap_changes_loss_but_cannot_condition_free_sampling():
    prior,m,p=fixture();engine=AnswerMeaningEngine(prior.thought,prior.codec,'SHA',['a'],times=[1.],batch=1,policy=LatentPolicy(steps=4))
    noise=torch.randn(1,7,16,generator=torch.Generator().manual_seed(41));t=torch.ones(1)
    loss1,_=engine.objective(m,p,t,noise)
    alternate=dict(p,semantic_target=p['semantic_target']+3.,answer_SHA='other')
    loss2,_=engine.objective(m,alternate,t,noise)
    assert not torch.equal(loss1,loss2)
    engine.thought.eval();a=engine.thought.sample(m,noise,4).semantic;b=engine.thought.sample(m,noise,4).semantic
    assert torch.equal(a,b) and engine.codec.calls==0
    with pytest.raises(TypeError):engine.thought.sample(m,noise,4,target=p['semantic_target'])
