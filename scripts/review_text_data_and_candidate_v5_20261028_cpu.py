"""Independent stored-row reconstruction and actual-parent CPU import, no forwards."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest(directory):
    value = read(directory/'manifest.json')
    for name, expected in value['files'].items():
        assert file_sha(directory/name) == expected, name
    clean = {k: v for k, v in value.items() if k != 'self_excluding_payload_sha256'}
    assert sha(json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')) == value['self_excluding_payload_sha256']
    return value


def main():
    out = ROOT/'research/artifacts/text_language_candidate_20261028_v5_cpu'
    if out.exists():
        raise FileExistsError('new root review output must not overwrite')
    base = ROOT/'research/artifacts'
    sd = base/'text_surface_alignment_20261028_v5_cpu'
    cd = base/'text_story_continuation_20261028_v5_cpu'
    sm, cm = manifest(sd), manifest(cd)
    assert sm['builder_sha256'] == file_sha(ROOT/'research/text_surface_data_v5.py')
    assert cm['builder_sha256'] == file_sha(ROOT/'research/text_corpus_contract_v5.py')
    features, targets, pairs = (read(sd/(n+'.json')) for n in ('features', 'targets', 'pairs'))
    old = ROOT/'research/artifacts/text_generation_20261028_v1'
    oldsource = {n: read(old/(n+'.json')) for n in ('train', 'dev_pairs', 'dev_wording')}
    groups = {n: {r['group'] for r in oldsource[n]} for n in ('train', 'dev_pairs')}
    olddev = {r['prompt'] for n in ('dev_pairs', 'dev_wording') for r in oldsource[n]}
    rebuilt = {}
    for split, fs in features.items():
        ts = targets[split]; support = Counter()
        for f, t in zip(fs, ts):
            assert set(f) == {'id', 'prompt'} and sha(f['prompt'].encode('utf-8')) == f['id']
            raw = f['prompt'].encode('utf-8'); fact = t['source_fact']
            assert f['prompt'].count(fact['fact']) == 1
            group = ''.join(sorted((fact['a'], fact['b'])))+'|'+fact['language']
            assert fact['group'] == group and group in groups['dev_pairs' if split == 'dev_pair' else 'train']
            start = raw.index(fact['fact'].encode('utf-8')); end = start+len(fact['fact'].encode('utf-8'))
            q, r = (fact['b'], fact['a']) if t['reverse'] else (fact['a'], fact['b'])
            assert [raw[n:n+1].decode('ascii') for n in t['span_starts']] == [q, r]
            assert all(not start <= n < end for n in t['span_starts'])
            assert t['span_ends'] == [n+1 for n in t['span_starts']]
            cases = []
            for a, b in itertools.product((0, 1, 2), repeat=2):
                op = fact['operator']
                condition = {'greater': a>b, 'less': a<b, 'not greater': a<=b, 'not less': a>=b}[op]
                if condition:
                    x, y = (b, a) if t['reverse'] else (a, b)
                    cases.append(0 if x>y else 1 if x<y else 2)
            relation = cases[0] if len(set(cases)) == 1 else 2
            assert t['head_labels'] == [ord(q)-65, ord(r)-65, relation, int(fact['language']=='ko')]
            expected = ('It is unknown.' if relation==2 else f'{q} is {("larger", "smaller")[relation]} than {r}.') if fact['language']=='en' else (
                '알 수 없습니다.' if relation==2 else f'{q}는 {r}보다 {("큽니다", "작습니다")[relation]}.')
            assert t['answer'] == expected
            if split == 'train':
                assert f['prompt'] not in olddev
            support[t['language']+'|'+t['label']] += 1
        for pair in pairs[split]:
            a, b = [ts[pair[k]] for k in ('first', 'second')]
            assert a['source_fact'] == b['source_fact'] and a['head_labels'][:2] == b['head_labels'][:2][::-1]
            assert a['family'] == b['family'] and a['layout'] == b['layout']
        rebuilt[split] = {'rows': len(fs), 'support': dict(support), 'pairs': len(pairs[split])}
        assert len(fs) == sm['statistics'][split]['rows'] and dict(support) == sm['statistics'][split]['support']
    literals = {k: {f['prompt'] for f in fs} for k, fs in features.items()}
    assert all(not literals[a] & literals[b] for a, b in itertools.combinations(literals, 2))
    order = read(sd/'one_epoch_order.json')
    assert sorted(order) == list(range(2048))
    pairlookup = {frozenset((p['first'], p['second'])) for p in pairs['train']}
    for i in range(0, 2048, 8):
        assert all(frozenset(order[n:n+2]) in pairlookup for n in range(i, i+8, 2))
    resumed = json.loads(json.dumps({'order': order, 'cursor': 1016}))
    assert resumed['order'][resumed['cursor']:resumed['cursor']+8] == order[1016:1024]
    docs, fs, ts = (read(cd/(n+'.json')) for n in ('documents', 'features', 'targets'))
    olddocs = ROOT.parent/'outputs/astra_document_data_candidate_20260926'
    corpus_review = {}; normsets = {}
    for split, ds in docs.items():
        source = {r['source_id']: r for r in (json.loads(s) for s in (olddocs/(split+'_records.jsonl')).read_text(encoding='utf-8').splitlines())}
        bydoc = defaultdict(list)
        for f, t in zip(fs[split], ts[split]):
            assert set(f) == {'id', 'prompt'} and f['id'] == t['id']
            bydoc[t['document_id']].append((f, t))
        physical = context = supervised = eos = initial = 0; normsets[split] = set()
        for d in ds:
            assert d == source[d['source_id']] and d['group'] == split
            raw = d['text'].encode('utf-8'); physical += len(raw)
            assert sha(raw) == d['raw_text_sha256']
            normalized = ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', d['text']).casefold()))
            assert sha(normalized.encode('utf-8')) == d['normalized_text_sha256']
            normsets[split].add(sha(normalized.encode('utf-8')))
            rows = bydoc[d['source_id']]; cursor = rows[0][1]['target_start']; initial += cursor
            assert cursor <= 64
            for f, t in rows:
                a, b, c = t['context_start'], t['target_start'], t['target_end']
                assert b == cursor and f['prompt'].encode('utf-8') == raw[a:b] and t['target'].encode('utf-8') == raw[b:c]
                assert 0 < b-a <= 256 and 0 < c-b <= 128
                assert t['source_eof'] == (c==len(raw))
                assert t['target_ids'] == list(raw[b:c])+([256] if c==len(raw) else [])
                context += b-a; supervised += c-b; eos += c==len(raw); cursor = c
            assert cursor == len(raw)
        assert eos == len(ds) and physical == initial+supervised
        expected = cm['statistics'][split]
        assert (physical, context, supervised, eos) == (expected['physical_document_utf8_bytes'], expected['repeated_context_utf8_bytes'], expected['supervised_target_utf8_bytes'], expected['source_eof_labels'])
        corpus_review[split] = {'documents': len(ds), 'rows': len(fs[split]), 'physical_bytes': physical,
            'repeated_context_bytes': context, 'target_bytes': supervised, 'real_EOF': eos}
    assert not normsets['train'] & normsets['dev']
    # Independent complete cross-split exact shingle comparison, no source expansion.
    def grams(text):
        words = re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())
        return {tuple(words[i:i+8]) for i in range(len(words)-7)}
    tg = [grams(d['text']) for d in docs['train']]
    dg = [grams(d['text']) for d in docs['dev']]
    assert all(len(a & b)/len(a | b) < .8 for a in tg for b in dg)
    # Actual parent is loaded and copied, but is never forwarded on CPU or CUDA.
    import torch
    torch.set_num_threads(2)
    import text_generation_support_v1 as support
    from fdt_local.model_masked import Config
    from research.text_language_continuation_v5 import LanguageContinuation
    parent = ROOT/'runs/text_span_alignment_20261028_v4/realizer/checkpoints/state_1024.pt'
    parent_sha = file_sha(parent)
    assert parent_sha == '5abf9351ddd12509e3622783dc95373f0ac1aeb1fef1a1feb5bb241c44a608df'
    payload = torch.load(parent, map_location='cpu', weights_only=True)
    before = support.state_sha(payload['model'])
    candidate = LanguageContinuation(Config(**payload['core']))
    mapping = candidate.warmstart(payload['model']['generator'], payload['model']['span_generator'])
    for k, value in candidate.vp.state_dict().items():
        if k != 'positions.weight':
            assert torch.equal(value, payload['model']['generator'][k])
    for k, value in candidate.decoder.state_dict().items():
        if k != 'positions.weight':
            assert torch.equal(value, payload['model']['span_generator']['decoder.'+k])
    assert before == support.state_sha(payload['model']) and not torch.cuda.is_initialized()
    spec = candidate.specification()
    assert spec['trainable_language_parameters'] > 0 and spec['trainable_language_blocks'] == [7, 8]
    assert support.core_sha(ROOT) == support.CORE_SHA
    out.mkdir(parents=True, exist_ok=False)
    files = {str(p.relative_to(ROOT)): file_sha(p) for p in (
        ROOT/'research/text_language_continuation_v5.py', ROOT/'tests/test_text_data_contracts_v5.py',
        ROOT/'tests/test_text_language_candidate_v5.py', Path(__file__))}
    result = {'status': 'CPU_DATA_AND_ACTUAL_PARENT_IMPORT_PASS', 'surface_rows': rebuilt,
        'corpus_rows': corpus_review, 'cross_split_8word_Jaccard_ge_0_8_pairs': 0,
        'surface_manifest_FILE_sha256': file_sha(sd/'manifest.json'),
        'corpus_manifest_FILE_sha256': file_sha(cd/'manifest.json'),
        'candidate_specification': spec, 'warmstart': mapping, 'parent_sha256': parent_sha,
        'parent_model_digest_before_after': before, 'core_sha256': support.CORE_SHA,
        'focused_data_CPU_tests': 10, 'focused_tiny_candidate_CPU_tests': 5,
        'test_scope': 'CPU tiny gradient/causality/model-Adam-nextstep only, not a complete worker/full phase/CUDA recovery contract.',
        'actual_parent_model_forward_calls': 0, 'CUDA_initialized': torch.cuda.is_initialized(),
        'new_GPU_training': False, 'new_optimizer_trial': False, 'actual_worker_or_OS_owner': False,
        'run_plan_or_seal': False, 'actual_generation_performance_improvement_claimed': False,
        'E2_final_accessed': False, 'new_code_files': files}
    (out/'root_cpu_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('new_code_files', 'candidate_specification')}, ensure_ascii=True))
    print(json.dumps({k: v for k, v in spec.items() if k != 'trainable_names'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
