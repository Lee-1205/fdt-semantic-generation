"""Local TinyStories byte-continuation contract. CPU stdlib only, no model imports.

No downloads here; source evidence is fetched separately and bound by hashes.
Only explicit old train/dev files are opened. Sealed test and E2 are not read.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
SOURCE_SHA = {'train_records.jsonl': 'f4a6d017f21d4682bd33b640cfe79e5653ab987dcf10a523b707eaed8de5a865',
              'dev_records.jsonl': 'c7b1c6eba100e14e99295c297f1b726d1ae2a05a96ccab8ddcf181b7d1775153'}
EOS, PAD, BOS = 256, 257, 258
CONTEXT_BYTES, TARGET_BYTES, INITIAL_PREFIX_BYTES = 256, 128, 64


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def boundaries(text):
    positions = [0]
    for character in text:
        positions.append(positions[-1] + len(character.encode('utf-8')))
    return positions


def windows(document):
    """Nonoverlapping supervised bytes, context overlap, EOS only at source EOF."""
    raw = document['text'].encode('utf-8'); edges = boundaries(document['text'])
    if len(raw) <= INITIAL_PREFIX_BYTES:
        raise ValueError('document shorter than required initial prefix')
    start = max(p for p in edges if p <= INITIAL_PREFIX_BYTES)
    result = []
    while start < len(raw):
        end = max(p for p in edges if p <= start + TARGET_BYTES)
        if end <= start:
            raise ValueError('target cannot advance')
        context_start = min(p for p in edges if p >= max(0, start - CONTEXT_BYTES))
        prompt, target = raw[context_start:start], raw[start:end]
        complete = end == len(raw)
        row_id = sha(document['source_id']+'|'+str(start)+'|'+str(end))
        result.append(({'id': row_id, 'prompt': prompt.decode('utf-8')},
            {'id': row_id, 'document_id': document['source_id'], 'target': target.decode('utf-8'),
             'context_start': context_start, 'target_start': start, 'target_end': end,
             'source_eof': complete, 'target_ids': list(target) + ([EOS] if complete else []),
             'split': document['group'], 'raw_document_sha256': sha(raw)}))
        start = end
    return result


def pack_targets(targets):
    """Gold training/score only; causal prefix BOS then previous *target* byte."""
    width = TARGET_BYTES + 1
    out = []
    for target in targets:
        ids = target['target_ids']
        if not ids or len(ids) > width or ids.count(EOS) != int(target['source_eof']):
            raise ValueError('EOS/target capacity contract')
        labels = ids + [-100] * (width - len(ids))
        prefix = [BOS] + ids[:-1] + [PAD] * (width - len(ids))
        out.append({'labels': labels, 'causal_prefix_ids': prefix,
                    'loss_mask': [v != -100 for v in labels]})
    return out


def cross_split_near_duplicates(docs, threshold=.8):
    """Exact 8-word-shingle Jaccard candidate index, no approximate MinHash."""
    train = {}; index = defaultdict(set); found = []
    for doc in docs['train']:
        words = normalized(doc['text']).split()
        grams = {tuple(words[i:i+8]) for i in range(max(0, len(words)-7))}
        train[doc['source_id']] = grams
        for gram in grams:
            index[gram].add(doc['source_id'])
    for doc in docs['dev']:
        words = normalized(doc['text']).split()
        grams = {tuple(words[i:i+8]) for i in range(max(0, len(words)-7))}
        candidates = set().union(*(index[g] for g in grams)) if grams else set()
        for key in sorted(candidates):
            other = train[key]; union = len(grams | other)
            score = len(grams & other)/union if union else 1.
            if score >= threshold:
                found.append({'train': key, 'dev': doc['source_id'], 'jaccard': score})
    return found


def prepare_documents(source, blocked):
    documents, excluded, all_seen = {}, [], set()
    for split in ('train', 'dev'):
        allowed = []
        for row in source[split]:
            if row['group'] != split or row['source_split'] != ('train' if split == 'train' else 'validation'):
                raise ValueError('original split changed')
            raw_hash, norm_hash = sha(row['text']), sha(normalized(row['text']))
            if raw_hash != row['raw_text_sha256'] or norm_hash != row['normalized_text_sha256']:
                raise ValueError('source text hash changed')
            if norm_hash in all_seen:
                raise ValueError('normalized duplicate across source documents')
            all_seen.add(norm_hash)
            reason = None
            if row['source_id'] in blocked:
                reason = 'inherited_old_crop_16_token_overlap_proxy'
            elif '\ufffd' in row['text'] or any(x in row['text'] for x in ('\u00e2\u20ac', '\u00c3\u00a2', '\u00c2\u00a0')):
                reason = 'conservative_mojibake_or_replacement_marker'
            if reason:
                excluded.append({'source_id': row['source_id'], 'split': split, 'reason': reason})
                continue
            allowed.append(row)
        # Selection is before model evaluation; old split assignment is immutable.
        documents[split] = sorted(allowed, key=lambda r: r['normalized_text_sha256'])[:(512 if split == 'train' else 64)]
        if len(documents[split]) != (512 if split == 'train' else 64):
            raise ValueError('insufficient eligible documents')
    if cross_split_near_duplicates(documents):
        raise ValueError('near duplicate across selected document splits')
    return documents, excluded


def validate_rows(documents, features, targets):
    for split in ('train', 'dev'):
        by_doc = defaultdict(list)
        if len(features[split]) != len(targets[split]):
            raise ValueError('feature/target row count')
        for feature, target in zip(features[split], targets[split]):
            if set(feature) != {'id', 'prompt'} or feature['id'] != target['id']:
                raise ValueError('gold exposed in features or identity mismatch')
            by_doc[target['document_id']].append((feature, target))
        for doc in documents[split]:
            raw = doc['text'].encode('utf-8'); rows = by_doc[doc['source_id']]
            first = max(p for p in boundaries(doc['text']) if p <= INITIAL_PREFIX_BYTES)
            cursor = first
            for feature, target in rows:
                if target['split'] != split or target['target_start'] != cursor:
                    raise ValueError('split moved or supervised bytes overlap/gap')
                a, b, c = target['context_start'], target['target_start'], target['target_end']
                if feature['prompt'].encode('utf-8') != raw[a:b] or target['target'].encode('utf-8') != raw[b:c]:
                    raise ValueError('context or target differs from source')
                if target['source_eof'] != (c == len(raw)) or len(raw[a:b]) > CONTEXT_BYTES or c-b > TARGET_BYTES:
                    raise ValueError('artificial EOS/capacity violation')
                expected_ids = list(raw[b:c]) + ([EOS] if c == len(raw) else [])
                if target['target_ids'] != expected_ids:
                    raise ValueError('byte labels differ')
                cursor = c
            if cursor != len(raw) or sum(t['source_eof'] for _, t in rows) != 1:
                raise ValueError('document coverage or real EOF count')
        pack_targets(targets[split])
    return True


def write_new(workspace, directory, evidence):
    workspace, directory, evidence = Path(workspace), Path(directory), Path(evidence)
    if directory.exists():
        raise FileExistsError('immutable corpus artifact already exists')
    src = workspace/'outputs/astra_document_data_candidate_20260926'
    for filename, expected in SOURCE_SHA.items():
        if file_sha(src/filename) != expected:
            raise ValueError('changed local source '+filename)
    notice = json.loads((evidence/'rights_review.json').read_text(encoding='utf-8'))
    for filename, expected in notice['files'].items():
        if file_sha(evidence/filename) != expected:
            raise ValueError('changed source/license evidence')
    if notice['dataset_revision'] != REVISION or notice['status'] != 'LOCAL_COMPUTATIONAL_USE_REVIEWED':
        raise ValueError('rights scope unreviewed')
    source = {split: [json.loads(line) for line in (src/(split+'_records.jsonl')).read_text(encoding='utf-8').splitlines()]
              for split in ('train', 'dev')}
    old_overlap = json.loads((src/'old_crop_overlap.json').read_text(encoding='utf-8'))
    # Existing metadata proxy only; does not read sealed test, E2, or historical payloads.
    blocked = {r['source_id'] for r in old_overlap['matches'] if r['group'] in ('train', 'dev')}
    documents, excluded = prepare_documents(source, blocked)
    pages = json.loads((src/'fetch_manifest.json').read_text(encoding='utf-8'))['pages']
    bound = json.loads((src/'revision_binding_audit.json').read_text(encoding='utf-8'))
    if bound['status'] != 'PASS' or bound['revision'] != REVISION:
        raise ValueError('local source revision unbound')
    page_evidence = {p['sha256']: p for p in bound['pages']}
    needed = {r['source_page_sha256'] for rs in documents.values() for r in rs}
    origin_rows = {}
    source_hashes = {}
    for page in pages:
        if page['sha256'] not in needed:
            continue
        path = src/'source'/Path(page['path']).name
        if file_sha(path) != page['sha256'] or page['sha256'] not in page_evidence:
            raise ValueError('source response identity')
        response = json.loads(path.read_text(encoding='utf-8'))
        for row in response['rows']:
            if row.get('truncated_cells'):
                raise ValueError('truncated source response')
            origin_rows[page['split']+':'+str(row['row_idx'])] = row['row']['text']
        source_hashes[str(path)] = page['sha256']
    for rs in documents.values():
        for row in rs:
            if origin_rows.get(row['source_id']) != row['text']:
                raise ValueError('stored text not exact source response')
    features, targets = {}, {}
    for split, docs in documents.items():
        rows = [row for doc in docs for row in windows(doc)]
        features[split] = [f for f, _ in rows]; targets[split] = [t for _, t in rows]
    validate_rows(documents, features, targets)
    stats = {split: {'source_documents_available': len(source[split]), 'selected_documents': len(docs),
        'physical_document_utf8_bytes': sum(len(d['text'].encode('utf-8')) for d in docs),
        'continuation_rows': len(features[split]),
        'repeated_context_utf8_bytes': sum(len(f['prompt'].encode('utf-8')) for f in features[split]),
        'supervised_target_utf8_bytes': sum(len(t['target'].encode('utf-8')) for t in targets[split]),
        'initial_unsupervised_prefix_bytes': sum(max(p for p in boundaries(d['text']) if p <= INITIAL_PREFIX_BYTES) for d in docs),
        'source_eof_labels': sum(t['source_eof'] for t in targets[split]),
        'supervised_byte_plus_eos_symbols': sum(len(t['target_ids']) for t in targets[split]),
        'artificial_boundary_eos_labels': 0} for split, docs in documents.items()}
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in (('documents', documents), ('features', features), ('targets', targets), ('exclusions', excluded)):
        (directory/(name+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest = {'schema': 'fdt_local_story_continuation_cpu_v5', 'status': 'CPU_DATA_ONLY',
        'builder_sha256': file_sha(__file__), 'dataset': 'roneneldan/TinyStories', 'source_revision': REVISION,
        'source_files': {str(src/n): h for n, h in SOURCE_SHA.items()}, 'source_pages': source_hashes,
        'metadata_bindings': {n: file_sha(src/n) for n in ('candidate_manifest.json', 'fetch_manifest.json', 'revision_binding_audit.json', 'old_crop_overlap.json')},
        'rights_evidence_path': str(evidence), 'rights_review_file_sha256': file_sha(evidence/'rights_review.json'),
        'license': 'CDLA-Sharing-1.0', 'local_computational_use_reviewed': True,
        'statistics': stats, 'exclusion_counts': dict(Counter(r['split']+'|'+r['reason'] for r in excluded)),
        'byte_contract': {'context_capacity': CONTEXT_BYTES, 'target_byte_capacity': TARGET_BYTES,
            'packed_target_capacity_with_eos': TARGET_BYTES+1, 'EOS': EOS, 'PAD': PAD, 'BOS': BOS,
            'document_initial_prefix_bytes': INITIAL_PREFIX_BYTES,
            'target_byte_overlap_within_document': 0, 'context_overlap_allowed': True},
        'old_split_preserved': True, 'sealed_test_records_or_shard_opened': False,
        'source_page_scope': 'Old validation API response pages also contain unselected historical test rows; only selected dev IDs are compared, exported, or used. No test records/shard or model evaluation accessed.',
        'E2_final_accessed': False,
        'factual_QA_rows': 0, 'project_report_training_rows': 0, 'new_downloaded_corpus_bytes': 0,
        'limits': ['English synthetic short stories, not diverse factual or bilingual general language corpus.',
            'Document groups and existing held split preserved; old development documents were already observed.',
            'Whole historical model pretraining overlap is UNKNOWN; source selection is not uniform sampling.',
            'Mojibake marker rejection is conservative, not a full text-quality certification.',
            'Byte symbols and repeated contexts are not BPE LM tokens or independent documents.',
            'Exact continuation is not the unique valid free-generation answer; no factual QA claim.'],
        'actual_model_calls': 0, 'optimizer_updates': 0, 'executed_epochs': 0,
        'files': {p.name: file_sha(p) for p in sorted(directory.glob('*.json'))}}
    manifest['self_excluding_payload_sha256'] = sha(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    (directory/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest
