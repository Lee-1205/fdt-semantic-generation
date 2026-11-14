"""Prospective CPU-only wording/position intervention data, never an inference parser.

Features contain only IDs and raw prompts. All roles, byte spans, facts, answers,
surface families, reverse-pair IDs and splits are offline targets/metadata.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

SOURCE_FILES = {
    'train.json': '2b2275811c673da932a2b640c309ba464c5773bf025ccc33a4afe9916f8644ee',
    'dev_pairs.json': '5511afb0832685fe7a0248dd4c150c553fe58d65aa628e00a446d4e7bb95d6c5',
    'dev_wording.json': 'cf6e2e91f4f314b64ddc596ce024205989343563b5adfecaef111845c9519b52',
}
OPS = ('greater', 'less', 'not greater', 'not less')
FAMILIES = {
    'en': (
        ('What is the size relation of ', ' to ', '?'),
        ('For ', ' versus ', ', report their size relation.'),
        ('Compare the size of ', ' relative to ', '.'),
        ('Is ', ' larger, smaller, or undetermined relative to ', '?'),
        ('State how ', ' relates in size to ', '.'),
        ('Taking ', ' first, describe its size against ', '.'),
    ),
    'ko': (
        ('크기를 판단할 대상은 ', ' 그리고 ', '입니다.'),
        ('첫 대상 ', ', 둘째 대상 ', '의 크기 관계를 답하세요.'),
        ('', '의 크기를 ', '에 견주어 설명하세요.'),
        ('질문의 순서는 ', ', ', '입니다. 크거나 작거나 불명인가요?'),
        ('', '를 기준이 아닌 첫 대상으로 두고 ', '와의 크기를 말하세요.'),
        ('답할 두 대상은 차례로 ', ', ', '입니다. 크기 관계는 무엇인가요?'),
    ),
}


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def source_facts(rows):
    """Collapse old template/reversal views into one *annotated* fact, offline."""
    facts = {}
    for row in rows:
        op, lang = row['operator'], row['language']
        phrases = {'greater': 'larger than', 'less': 'smaller than',
                   'not greater': 'not larger than', 'not less': 'not smaller than'}
        ko = {'greater': '크다', 'less': '작다', 'not greater': '크지 않다', 'not less': '작지 않다'}
        pattern = (rf'([A-H]) is {phrases[op]} ([A-H])\.' if lang == 'en'
                   else rf'([A-H])는 ([A-H])보다 {ko[op]}\.')
        matches = list(re.finditer(pattern, row['prompt']))
        if len(matches) != 1:
            raise ValueError('one source premise required')
        a, b = matches[0].groups()
        key = '|'.join((a, b, op, lang))
        value = {'a': a, 'b': b, 'operator': op, 'language': lang,
                 'group': row['group'], 'fact': matches[0].group(), 'key': key}
        if key in facts and facts[key] != value:
            raise ValueError('inconsistent source fact')
        facts[key] = value
    return sorted(facts.values(), key=lambda f: digest(f['key']))


def select_facts(facts, per_stratum):
    pools = defaultdict(list)
    for fact in facts:
        pools[(fact['language'], fact['operator'])].append(fact)
    if set(pools) != set(itertools.product(('en', 'ko'), OPS)):
        raise ValueError('missing source stratum')
    if min(map(len, pools.values())) < per_stratum:
        raise ValueError('insufficient source facts')
    return [fact for key in sorted(pools) for fact in pools[key][:per_stratum]]


def logic_label(fact, reverse):
    """Independent finite-order entailment, preserving equality under negation."""
    outcomes = set()
    for a, b in itertools.product(range(3), repeat=2):
        constraints = (a > b, a < b, not a > b, not a < b)
        if constraints[OPS.index(fact['operator'])]:
            q, r = (b, a) if reverse else (a, b)
            outcomes.add('greater' if q > r else 'less' if q < r else 'equal')
    return next(iter(outcomes)) if len(outcomes) == 1 else 'unknown'


def render(fact, reverse, family, layout, padding):
    lang = fact['language']; q, r = (fact['b'], fact['a']) if reverse else (fact['a'], fact['b'])
    left, middle, right = FAMILIES[lang][family]
    query = left + q + middle + r + right
    offsets = (len(left.encode('utf-8')), len((left + q + middle).encode('utf-8')))
    # Neutral text contains candidate-letter distractors; it states no comparisons.
    note = 'Note C H. ' if lang == 'en' else '메모 C H. '
    neutral = ('x' * padding + ' ') if padding else ''
    if layout in (0, 2):
        prefix = (note + neutral if layout == 2 else '') + fact['fact'] + ' '
        prompt = prefix + query
    elif layout in (1, 3):
        prefix = note + neutral if layout == 3 else ''
        prompt = prefix + query + ' ' + fact['fact']
    else:
        raise ValueError('unknown layout')
    starts = [len(prefix.encode('utf-8')) + n for n in offsets]
    if len(prompt.encode('utf-8')) > 256:
        raise ValueError('new context byte capacity exceeded')
    label = logic_label(fact, reverse)
    if lang == 'en':
        answer = 'It is unknown.' if label == 'unknown' else f'{q} is {"larger" if label == "greater" else "smaller"} than {r}.'
    else:
        answer = '알 수 없습니다.' if label == 'unknown' else f'{q}는 {r}보다 {"큽니다" if label == "greater" else "작습니다"}.'
    key = '|'.join((fact['key'], str(family), str(layout), str(padding)))
    feature = {'id': digest(prompt), 'prompt': prompt}
    target = {'answer': answer, 'span_starts': starts, 'span_ends': [n + 1 for n in starts],
              'head_labels': [ord(q)-65, ord(r)-65, ('greater', 'less', 'unknown').index(label), ('en', 'ko').index(lang)],
              'label': label, 'language': lang, 'source_fact': fact, 'reverse': reverse,
              'family': family, 'layout': layout, 'padding_bytes': padding, 'pair_key': digest(key)}
    return feature, target


def build(source):
    train_facts = select_facts(source_facts(source['train']), 8)
    pair_facts = select_facts(source_facts(source['dev_pairs']), 1)
    # Matched controls intentionally share eight training facts, never literal prompts.
    controls = select_facts(train_facts, 1)
    specs = {
        'train': (train_facts, range(4), ((0, 0), (1, 0), (2, 16), (3, 32))),
        'dev_pair': (pair_facts, range(4), ((0, 0),)),
        'dev_surface': (controls, (4, 5), ((0, 0), (1, 0))),
        'dev_position': (controls, (0, 1), ((2, 40), (3, 48))),
    }
    features, targets, pairs = {}, {}, {}
    old_observed = {r['prompt'] for k in ('dev_pairs', 'dev_wording') for r in source[k]}
    for split, (facts, families, positions) in specs.items():
        features[split], targets[split], pairs[split] = [], [], []
        for fact, family, (layout, padding) in itertools.product(facts, families, positions):
            first = len(features[split])
            for reverse in (False, True):
                f, t = render(fact, reverse, family, layout, padding)
                if split == 'train' and f['prompt'] in old_observed:
                    raise ValueError('old observed development literal in new train')
                features[split].append(f); targets[split].append(t)
            pairs[split].append({'first': first, 'second': first + 1,
                                 'pair_key': targets[split][-1]['pair_key']})
    data = {'features': features, 'targets': targets, 'pairs': pairs}
    validate(data, source)
    return data


def validate(data, source):
    train_groups = {r['group'] for r in source['train']}
    pair_groups = {r['group'] for r in source['dev_pairs']}
    if train_groups & pair_groups:
        raise ValueError('inherited pair/language split overlap')
    literal_sets = {}
    for split, features in data['features'].items():
        literal_sets[split] = {f['prompt'] for f in features}
        if len(literal_sets[split]) != len(features):
            raise ValueError('duplicate prompt within split')
        for f, t in zip(features, data['targets'][split]):
            if set(f) != {'id', 'prompt'} or f['id'] != digest(f['prompt']):
                raise ValueError('feature contract')
            raw = f['prompt'].encode('utf-8'); labels = t['head_labels']
            if any(raw[n:n+1] != bytes([65 + labels[i]]) for i, n in enumerate(t['span_starts'])):
                raise ValueError('independent byte span mismatch')
            expected = 'unknown' if t['source_fact']['operator'].startswith('not ') else t['source_fact']['operator']
            if t['reverse'] and expected != 'unknown':
                expected = {'less': 'greater', 'greater': 'less'}[expected]
            if expected != t['label'] or logic_label(t['source_fact'], t['reverse']) != expected:
                raise ValueError('independent entailment disagrees')
            if len(raw) > 256 or len(t['answer'].encode('utf-8')) + 1 > 40:
                raise ValueError('byte capacity')
            expected_group = pair_groups if split == 'dev_pair' else train_groups
            if t['source_fact']['group'] not in expected_group:
                raise ValueError('inherited group moved across split')
        for pair in data['pairs'][split]:
            a, b = [data['targets'][split][pair[k]] for k in ('first', 'second')]
            if a['source_fact'] != b['source_fact'] or a['head_labels'][:2] != b['head_labels'][:2][::-1]:
                raise ValueError('reverse pair premise/roles changed')
    for left, right in itertools.combinations(literal_sets, 2):
        if literal_sets[left] & literal_sets[right]:
            raise ValueError('literal cross-split contamination')
    if literal_sets['train'] & {r['prompt'] for k in ('dev_pairs', 'dev_wording') for r in source[k]}:
        raise ValueError('old observed development contamination')
    return True


def finite_order(pairs, row_count, epochs=1, seed=53, batch_size=8):
    if epochs < 1 or batch_size % 2 or row_count % batch_size:
        raise ValueError('finite full paired batches required')
    flat = [p[k] for p in pairs for k in ('first', 'second')]
    if sorted(flat) != list(range(row_count)):
        raise ValueError('exact partition into reverse pairs required')
    rng = random.Random(seed); order = []
    for _ in range(epochs):
        groups = [(p['first'], p['second']) for p in pairs]; rng.shuffle(groups)
        order.extend(i for pair in groups for i in pair)
    return order


def write_new(root, directory):
    root, directory = Path(root), Path(directory)
    if directory.exists():
        raise FileExistsError('immutable artifact already exists')
    src = root/'research/artifacts/text_generation_20261028_v1'
    for name, expected in SOURCE_FILES.items():
        if file_sha(src/name) != expected:
            raise ValueError('changed source '+name)
    source = {Path(n).stem: json.loads((src/n).read_text(encoding='utf-8')) for n in SOURCE_FILES}
    data = build(source)
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in data.items():
        with (directory/(name+'.json')).open('x', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
    order = finite_order(data['pairs']['train'], len(data['features']['train']))
    (directory/'one_epoch_order.json').write_text(json.dumps(order)+'\n', encoding='utf-8')
    stats = {}
    for split, features in data['features'].items():
        ts = data['targets'][split]
        stats[split] = {'rows': len(features), 'reverse_pairs': len(data['pairs'][split]),
            'unique_fact_keys': len({t['source_fact']['key'] for t in ts}),
            'pair_language_groups': len({t['source_fact']['group'] for t in ts}),
            'support': dict(Counter(t['language']+'|'+t['label'] for t in ts)),
            'prompt_utf8_bytes': sum(len(f['prompt'].encode('utf-8')) for f in features),
            'target_utf8_bytes': sum(len(t['answer'].encode('utf-8')) for t in ts),
            'max_prompt_bytes': max(len(f['prompt'].encode('utf-8')) for f in features),
            'subject_byte_position_range': [min(t['span_starts'][0] for t in ts), max(t['span_starts'][0] for t in ts)],
            'families': sorted({t['family'] for t in ts})}
    manifest = {'schema': 'fdt_surface_position_cpu_v5', 'status': 'CPU_DATA_ONLY',
        'builder_sha256': file_sha(__file__), 'source_files': SOURCE_FILES,
        'feature_keys': ['id', 'prompt'], 'statistics': stats,
        'same_name_alphabet': 'A-H', 'new_independent_graphs_claimed': False,
        'train_families': [0, 1, 2, 3], 'held_surface_families': [4, 5],
        'matched_controls_share_training_facts': True, 'old_dev_literal_train_overlap': 0,
        'limitations': ['Authored comparison/negation only; new prompts are not broad language learning.',
            'Surface/position controls intentionally share base facts and names with training.',
            'Pair control inherits old pair+language partition; old dev has been observed repeatedly.',
            'Family/literal distinction is an authored intervention, not a universal linguistic taxonomy.'],
        'actual_model_calls': 0, 'optimizer_updates': 0, 'E2_final_accessed': False,
        'files': {p.name: file_sha(p) for p in sorted(directory.glob('*.json'))}}
    manifest['self_excluding_payload_sha256'] = digest(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    (directory/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest
