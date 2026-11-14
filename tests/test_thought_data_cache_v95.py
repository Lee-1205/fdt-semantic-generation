import copy,hashlib
from pathlib import Path
import pytest
import torch
from research.thought_data_v95 import build,answer_from_finite_worlds
from research.thought_cache_v95 import split_input,assemble_input
from research.memory_thought_diffusion_v94 import InputMeaning

ROOT=Path(__file__).resolve().parents[1]


def test_existing_English_finite_split_targets_and_pair_orders():
    data,orders=build(ROOT/'research/artifacts/text_surface_alignment_20261028_v5_cpu')
    train=data['train'];ids={r['id'] for r in train['features']}
    assert len(ids)==384 and len(orders)==4
    assert all(set(r)=={'id','prompt'} for d in data.values() for r in d['features'])
    for epoch in orders:
        flat=[i for pair in epoch for i in pair]
        assert len(flat)==384 and set(flat)==ids
        lookup={t['id']:t for t in train['targets']}
        assert all(lookup[a]['source_fact']==lookup[b]['source_fact'] and lookup[a]['reverse']!=lookup[b]['reverse'] for a,b in epoch)


def test_negation_does_not_become_false_strict_reverse_relation():
    t=dict(source_fact=dict(a='A',b='B',operator='not greater'),reverse=True)
    assert answer_from_finite_worlds(t)==('unknown','It is unknown.')


def test_cache_roundtrip_is_input_only_and_preserves_all_three_memories():
    m=InputMeaning(torch.randn(1,4,16),torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),
        torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'epoch',(('id','version','SHA'),))
    static,dynamic=split_input(m,'id','a'*64);rebuilt=assemble_input(static,dynamic)
    assert all(torch.equal(getattr(m,k),getattr(rebuilt,k)) for k in ('source','anchors','prior','contextual_long','contextual_recent','exact_keys','exact_values','exact_valid'))
    assert rebuilt.source.data_ptr()!=dynamic['source'].data_ptr()
    dynamic['answer']='leak'
    with pytest.raises(ValueError):assemble_input(static,dynamic)


def test_cache_rejects_answer_memory_epoch_or_nonNULL_Exact():
    m=InputMeaning(torch.randn(1,4,16),torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),
        torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'epoch',(('id','version','SHA'),))
    s,d=split_input(m,'id','a'*64);bad=copy.deepcopy(d);bad['model_epoch']='answer_epoch'
    with pytest.raises(ValueError):assemble_input(s,bad)
    s['exact_valid'].fill_(True)
    with pytest.raises(ValueError):assemble_input(s,d)
