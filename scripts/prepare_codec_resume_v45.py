"""Build one finite balanced V45 schedule from the existing OWT TRAIN split."""
import hashlib
import importlib.util
import json
import random
import unicodedata
from pathlib import Path

import codec_resume_v45_contract as c

BUILDER = c.ROOT / 'research/artifacts/owt_fuzzy_reconstruction_20261028_v23/executed_builder_snapshot.py'


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))


def key(text):
    return unicodedata.normalize('NFKC', ' '.join(text.split())).casefold()


def main():
    if c.RUN.exists() or c.ARTIFACT.exists():
        raise FileExistsError('V45 output or inputs already exist')
    original = c.read(c.PARENT/'plan.json')
    previous = c.read(c.PARENT/'root_result.json')
    if not previous['status'].startswith('V44_COMPLETE_') or not previous['all_owned_processes_gone']:
        raise ValueError('V44 terminal review required')
    corpus = Path(original['artifact'])
    focus = Path(original['focused_artifact'])
    for path in (corpus/'features.json', corpus/'documents.json', focus/'features.json'):
        if c.sha256(path) != original['files'][str(path)]:
            raise ValueError('original corpus pin mismatch')
    spec = importlib.util.spec_from_file_location('historical_sentence_rules', BUILDER)
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    docs = c.read(corpus/'documents.json')
    original_rows = c.read(corpus/'features.json')
    dev_docs = {d['normalized_sha256'] for d in docs['dev']}
    if dev_docs.intersection(d['normalized_sha256'] for d in docs['train']):
        raise ValueError('document split overlap')
    excluded = {key(f['prompt']) for f in original_rows['train'] + original_rows['dev']
                + c.read(focus/'features.json')}
    prior_features = c.read(Path(original['training_artifact'])/'features.json')[:8192]
    excluded.update(key(row['prompt']) for row in prior_features)
    for doc in docs['dev']:
        excluded.update(key(sentence) for sentence in rules.split_spans(doc['text']))
    rng = random.Random(c.SEED)
    train_docs = list(docs['train'])
    rng.shuffle(train_docs)
    pools = []
    for doc in train_docs:
        pool = []
        for span, sentence in enumerate(rules.split_spans(doc['text'])):
            if not rules.acceptable_new_span(sentence):
                continue
            normalized = key(sentence)
            if normalized in excluded:
                continue
            excluded.add(normalized)
            identity = hashlib.sha256(f"v45:{doc['id']}:{span}:{sentence}".encode()).hexdigest()
            pool.append(dict(id=identity, prompt=sentence, document_id=doc['id'], span=span))
        rng.shuffle(pool)
        if pool:
            pools.append(pool[:4])
    fresh = []
    for i in range(4):
        fresh.extend(pool[i] for pool in pools if i < len(pool))
    fresh = fresh[:c.FRESH]
    if len(fresh) != c.FRESH:
        raise ValueError('insufficient new TRAIN sentence spans')
    # The previous builder emitted four consecutive spans per TRAIN document.
    if len(original_rows['train']) != 4*len(docs['train']) or len(docs['train']) != c.ORIGINAL:
        raise ValueError('unexpected source grouping')
    old_replay = [original_rows['train'][4*i+rng.randrange(4)] for i in range(c.ORIGINAL)]
    replay = rng.sample(prior_features, c.PRIOR) + old_replay
    features = [{k:row[k] for k in ('id','prompt')} for row in fresh + replay]
    if len({row['id'] for row in features}) != len(features):
        raise ValueError('duplicate row identity')
    targets = [dict(id=row['id'], target=row['prompt'],
                    target_ids=list(row['prompt'].encode('utf-8'))+[256]) for row in features]
    fresh_order = list(range(c.FRESH))
    prior_order = list(range(c.FRESH, c.FRESH+c.PRIOR))
    old_order = list(range(c.FRESH+c.PRIOR, len(features)))
    for order in (fresh_order, prior_order, old_order):
        rng.shuffle(order)
    orders = []
    for offset in range(0, c.FRESH, 2):
        batch = fresh_order[offset:offset+2]+prior_order[offset:offset+2]+old_order[2*offset:2*offset+4]
        rng.shuffle(batch)
        orders.append(batch)
    c.ARTIFACT.mkdir()
    save(c.ARTIFACT/'features.json', features)
    save(c.ARTIFACT/'targets.json', targets)
    save(c.ARTIFACT/'orders.json', orders)
    save(c.ARTIFACT/'fresh_provenance.json',
         [{k:row[k] for k in ('id','document_id','span')} for row in fresh])
    save(c.ARTIFACT/'manifest.json', dict(
        policy=c.POLICY, source_corpus=str(corpus), sentence_rule_SHA256=c.sha256(BUILDER),
        selected_rows=len(features), fresh_distinct_documents=len({row['document_id'] for row in fresh}),
        replay_document_count=len(docs['train']), target_UTF8_bytes=sum(len(f['prompt'].encode()) for f in features),
        evaluation_rows_added_to_training=0, new_download_bytes=0,
        limitation='New spans from the same TRAIN documents, with heuristic sentence boundaries.'))
    files = {name:digest for name,digest in original['files'].items() if Path(name).suffix != '.pt'}
    for path in [c.SOURCE, c.PARENT/'plan.json', c.PARENT/'root_result.json', BUILDER,
                 *c.ARTIFACT.iterdir(), Path(__file__).resolve(),
                 *(c.ROOT/'scripts'/name for name in (
                     'codec_resume_v45_contract.py','three_memory_codec_engine_v45.py',
                     'run_three_memory_codec_20261111_v45.py','supervise_three_memory_codec_20261111_v45.py',
                     'build_codec_resume_v45.py'))]:
        files[str(path)] = c.sha256(path)
    plan = dict(original, schema=c.SCHEMA, authorized=False, output=str(c.RUN), source=str(c.SOURCE),
                source_sha=c.SOURCE_SHA, source_plan_sha=c.SOURCE_PLAN, source_algorithm=c.SOURCE_ALGORITHM,
                source_plan_path=str(c.PARENT/'plan.json'), source_global_attempts=c.GLOBAL_ATTEMPTS,
                source_global_updates=c.GLOBAL_UPDATES, limits=c.LIMITS, cost_contract=c.COST,
                storage_contract=c.STORAGE, continuation_policy=c.POLICY, epochs=1,
                order_seed_range=[c.SEED], learning_epoch_number=16, training_artifact=str(c.ARTIFACT),
                training_exposure=c.read(c.ARTIFACT/'manifest.json'),
                purpose='V45 same-input codec learning on unused TRAIN spans plus replay',
                allocation_start=None, cutoff=None, deadline=None, files=files,
                source_baseline_reference=dict(run=str(c.PARENT), same_function=True,
                    root_result_sha256=c.sha256(c.PARENT/'root_result.json'), repeat_initial_GPU_baseline=False))
    # The historical disk/time policies are replaced by the V45 contract.
    for key_name in ('time_cost_contract', 'start_storage_contract', 'resource_policy'):
        plan.pop(key_name, None)
    save(c.ARTIFACT/'plan_unsealed.json', plan)
    print(json.dumps(plan['training_exposure'], ensure_ascii=False))


if __name__ == '__main__':
    main()
