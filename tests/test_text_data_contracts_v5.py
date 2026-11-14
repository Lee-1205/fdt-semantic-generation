"""Meaningful CPU data boundaries, not model capability or CUDA tests."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import text_surface_data_v5 as surface
from research import text_corpus_contract_v5 as corpus


def actual_source():
    root = ROOT/'research/artifacts/text_generation_20261028_v1'
    return {name: json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
            for name in ('train', 'dev_pairs', 'dev_wording')}


def document(text, split='train', key='train:1'):
    return {'source_id': key, 'text': text, 'group': split,
            'source_split': 'train' if split == 'train' else 'validation',
            'raw_text_sha256': corpus.sha(text),
            'normalized_text_sha256': corpus.sha(corpus.normalized(text))}


def test_surface_gold_separation_and_old_literal_group_boundaries():
    source = actual_source(); data = surface.build(source)
    assert {k: len(v) for k, v in data['features'].items()} == {
        'train': 2048, 'dev_pair': 64, 'dev_surface': 64, 'dev_position': 64}
    for rows in data['features'].values():
        assert all(set(row) == {'id', 'prompt'} for row in rows)
    old = {r['prompt'] for k in ('dev_pairs', 'dev_wording') for r in source[k]}
    assert not old & {r['prompt'] for r in data['features']['train']}
    mutated = copy.deepcopy(data)
    mutated['targets']['dev_pair'][0]['source_fact']['group'] = source['train'][0]['group']
    with pytest.raises(ValueError, match='group moved'):
        surface.validate(mutated, source)


def test_negation_scope_leaves_equality_undetermined_and_reversal_exact():
    for op in surface.OPS:
        f = {'operator': op}
        direct, reverse = surface.logic_label(f, False), surface.logic_label(f, True)
        if op.startswith('not '):
            assert direct == reverse == 'unknown'
        else:
            assert direct != reverse and {direct, reverse} == {'greater', 'less'}


def test_surface_utf8_offsets_and_neutral_shift_are_not_character_indices():
    source = actual_source(); data = surface.build(source)
    for split, rows in data['features'].items():
        for feature, t in zip(rows, data['targets'][split]):
            raw = feature['prompt'].encode('utf-8')
            assert [raw[n] - 65 for n in t['span_starts']] == t['head_labels'][:2]
    fact = next(f for f in surface.source_facts(source['train']) if f['language'] == 'ko')
    a, x = surface.render(fact, False, 0, 2, 16)
    b, y = surface.render(fact, False, 0, 2, 40)
    assert y['span_starts'] == [p+24 for p in x['span_starts']]
    assert x['head_labels'] == y['head_labels'] and x['answer'] == y['answer']
    bad = copy.deepcopy(data); bad['targets']['train'][0]['span_starts'][0] += 1
    with pytest.raises(ValueError, match='span mismatch'):
        surface.validate(bad, source)


def test_finite_surface_pair_order_resume_without_replay_or_extra_epoch():
    data = surface.build(actual_source())
    order = surface.finite_order(data['pairs']['train'], 2048)
    assert sorted(order) == list(range(2048))
    assert order == surface.finite_order(data['pairs']['train'], 2048)
    pairs = {frozenset((p['first'], p['second'])) for p in data['pairs']['train']}
    for start in range(0, len(order), 8):
        assert all(frozenset(order[i:i+2]) in pairs for i in range(start, start+8, 2))
    state = json.loads(json.dumps({'order': order, 'cursor': 1016, 'attempt': 127}))
    assert state['order'][state['cursor']:state['cursor']+8] == order[1016:1024]
    assert order[2048:2056] == []
    with pytest.raises(ValueError):
        surface.finite_order(data['pairs']['train'][:-1], 2048)


def test_utf8_document_windows_cover_targets_once_and_only_eof_has_eos():
    doc = document('A short preface. '*5 + '민지는 작은 배를 접었다. '*30 + 'The story ended.')
    pairs = corpus.windows(doc)
    features = {'train': [f for f, _ in pairs], 'dev': []}
    targets = {'train': [t for _, t in pairs], 'dev': []}
    assert corpus.validate_rows({'train': [doc], 'dev': []}, features, targets)
    ids = [i for _, t in pairs for i in t['target_ids']]
    assert ids.count(corpus.EOS) == 1 and pairs[-1][1]['source_eof']
    assert all(corpus.EOS not in t['target_ids'] for _, t in pairs[:-1])
    expected = doc['text'].encode('utf-8')[pairs[0][1]['target_start']:]
    assert bytes(i for i in ids if i < 256) == expected
    assert any(len(t['target']) != len(t['target'].encode('utf-8')) for _, t in pairs)


def test_no_target_lookahead_in_context_or_prefix_padding_loss():
    doc = document('This document has ordinary context. '*25)
    pairs = corpus.windows(doc); packed = corpus.pack_targets([t for _, t in pairs])
    for (_, target), p in zip(pairs, packed):
        active = len(target['target_ids'])
        assert p['causal_prefix_ids'][:active] == [corpus.BOS] + target['target_ids'][:-1]
        assert all(x == -100 for x in p['labels'][active:])
        assert all(x == corpus.PAD for x in p['causal_prefix_ids'][active:])
        assert not any(p['loss_mask'][active:])
    bad = copy.deepcopy(pairs[-1][1]); bad['source_eof'] = False
    with pytest.raises(ValueError, match='EOS'):
        corpus.pack_targets([bad])


def test_document_integrity_split_and_cross_split_near_duplicate_guard():
    text = 'A child carried a blue basket through a quiet green garden. '*30
    a = document(text)
    b = document(text+' Again.', 'dev', 'validation:2')
    found = corpus.cross_split_near_duplicates({'train': [a], 'dev': [b]})
    assert len(found) == 1 and found[0]['jaccard'] >= .8
    bad = copy.deepcopy(a); bad['group'] = 'dev'
    with pytest.raises(ValueError, match='split changed'):
        corpus.prepare_documents({'train': [bad], 'dev': []}, set())
    bad = copy.deepcopy(a); bad['text'] += ' tampered'
    with pytest.raises(ValueError, match='hash changed'):
        corpus.prepare_documents({'train': [bad], 'dev': []}, set())


def test_false_eos_and_target_overlap_rejected_by_independent_source_rebuild():
    doc = document('A calm day in a small village. '*30)
    pairs = corpus.windows(doc)
    fs = {'train': [f for f, _ in pairs], 'dev': []}
    ts = {'train': [t for _, t in pairs], 'dev': []}
    bad = copy.deepcopy(ts); bad['train'][0]['source_eof'] = True
    with pytest.raises(ValueError, match='artificial EOS'):
        corpus.validate_rows({'train': [doc], 'dev': []}, fs, bad)
    bad = copy.deepcopy(ts); bad['train'][1]['target_start'] -= 1
    with pytest.raises(ValueError, match='overlap/gap'):
        corpus.validate_rows({'train': [doc], 'dev': []}, fs, bad)


def test_overwrite_rejected_before_reading_any_missing_sources(tmp_path):
    out = tmp_path/'immutable'; out.mkdir()
    with pytest.raises(FileExistsError):
        surface.write_new(tmp_path/'missing', out)
    with pytest.raises(FileExistsError):
        corpus.write_new(tmp_path/'missing', out, tmp_path/'missing')


def test_builders_import_no_model_framework_or_final_holdout_registry():
    # Data construction itself cannot initialize CUDA or call a model.
    import ast
    for module in (surface, corpus):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imported += [a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names]
        assert not any(name and name.startswith(('torch', 'research.text_diffusion', 'fdt_local')) for name in imported)
        assert not any(name and 'e2' in name.lower() for name in imported)
