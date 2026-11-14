"""Bounded JSON-only planning arithmetic. No torch/model/optimizer/network/RNG.

Reads the existing TRAIN/observed DEV contract only; never scans E2/registries.
Writes a new immutable small evidence directory, not a runnable training plan.
"""
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
DATA = ROOT / 'research/artifacts/memory_thought_data_20261113_v95_cpu_v1'
CONTRACT = ROOT / 'research/artifacts/answer_meaning_objective_20261114_v1'
OUT = WORK / 'outputs/FDT_ANSWER_MEANING_FEASIBILITY_20261114_evidence'
PINS = {}


def read(path):
    raw = path.read_bytes()
    if len(raw) > 8 * 1024**2:
        raise ValueError('bounded JSON inputs only')
    PINS[str(path)] = dict(SHA=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return json.loads(raw.decode('utf-8-sig'))


def finite_answer(t):
    f = t['source_fact']
    outcomes = set()
    for a, b in itertools.product(range(3), repeat=2):
        ok = {'greater': a > b, 'less': a < b,
              'not greater': a <= b, 'not less': a >= b}[f['operator']]
        if ok:
            q, r = (b, a) if t['reverse'] else (a, b)
            outcomes.add('greater' if q > r else 'less' if q < r else 'equal')
    label = next(iter(outcomes)) if len(outcomes) == 1 else 'unknown'
    q, r = (f['b'], f['a']) if t['reverse'] else (f['a'], f['b'])
    answer = 'It is unknown.' if label == 'unknown' else f'{q} is {"larger" if label == "greater" else "smaller"} than {r}.'
    if label != t['label'] or answer != t['answer']:
        raise ValueError('finite-world reference mismatch')


def describe(rows, feature_map):
    counts = collections.Counter(r['answer'] for r in rows)
    pairs = collections.defaultdict(list)
    for r in rows:
        pairs[r['pair_key']].append(r)
    classes = collections.Counter(r['label'] for r in rows)
    lengths = [len(feature_map[r['id']]['prompt'].encode()) for r in rows]
    return dict(
        rows=len(rows), distinct_facts=len({r['source_fact']['key'] for r in rows}),
        pair_language_groups=len({r['source_fact']['group'] for r in rows}),
        distinct_answers=len(counts), answer_frequencies=dict(counts), labels=dict(classes),
        family_counts=dict(collections.Counter(r['family'] for r in rows)),
        layout_counts=dict(collections.Counter(r['layout'] for r in rows)),
        operators=dict(collections.Counter(r['source_fact']['operator'] for r in rows)),
        reverse_counts=dict(collections.Counter(r['reverse'] for r in rows)),
        reverse_pair_groups=len(pairs), complete_reverse_pairs=sum(len(x) == 2 for x in pairs.values()),
        unpaired_rows=sum(len(x) == 1 for x in pairs.values()),
        positive_complete_reverse_pairs=sum(len(x) == 2 and all(r['label'] != 'unknown' for r in x) for x in pairs.values()),
        unknown_constant_correct=classes['unknown'], unknown_constant_accuracy=classes['unknown']/len(rows),
        row_answer_entropy_bits=-sum((n/len(rows))*math.log2(n/len(rows)) for n in counts.values()),
        input_bytes=sum(lengths), input_byte_min=min(lengths), input_byte_max=max(lengths),
        target_answer_bytes=sum(len(r['answer'].encode()) for r in rows),
        target_answer_lengths=dict(collections.Counter(len(r['answer'].encode()) for r in rows)),
    )


def main():
    if OUT.exists():
        raise FileExistsError('immutable evidence already exists')
    manifest = read(DATA / 'manifest.json')
    train = read(DATA / 'train_targets.json')
    train_features = read(DATA / 'train_features.json')
    conditions = read(CONTRACT / 'train_conditions.json')
    loss_targets = read(CONTRACT / 'train_targets_LOSS_ONLY.json')
    if len(train) != 384 or len(conditions) != 134 or len(loss_targets) != 134:
        raise ValueError('fixed planning scope')
    fmap = {f['id']: f for f in train_features}
    byid = {r['id']: r for r in train}
    selected_ids = [r['id'] for r in conditions]
    selected = [byid[i] for i in selected_ids]
    if len(set(selected_ids)) != 134:
        raise ValueError('unique selected TRAIN')
    for a, b in zip(conditions, loss_targets):
        if a['id'] != b['id'] or byid[a['id']]['answer'] != b['answer_text_offline_only']:
            raise ValueError('input/answer alignment')
    before, after = describe(train, fmap), describe(selected, fmap)
    dev = {}
    answer_set = {r['answer'] for r in selected}
    seen_fact_set = {r['source_fact']['key'] for r in selected}
    seen_group_set = {r['source_fact']['group'] for r in selected}
    all_rows = list(train)
    for split in ('dev_pair', 'dev_surface', 'dev_position'):
        rows = read(DATA / f'{split}_targets.json')
        features = {f['id']: f for f in read(DATA / f'{split}_features.json')}
        all_rows.extend(rows)
        dev[split] = describe(rows, features)
        dev[split].update(
            target_answer_seen_in_selected_TRAIN=sum(r['answer'] in answer_set for r in rows),
            fact_seen_in_selected_TRAIN=sum(r['source_fact']['key'] in seen_fact_set for r in rows),
            pair_language_group_seen_in_selected_TRAIN=sum(r['source_fact']['group'] in seen_group_set for r in rows),
            literal_selected_TRAIN_overlap=len(set(features) & set(selected_ids)),
        )
    for row in all_rows:
        finite_answer(row)
    old = read(ROOT / 'runs/thought_cache_20261113_v95/target_index_loss_only.json')
    ceiling = [dict(answer=r['expected'], actual=r['output'],
                    expected_bytes=len(r['expected'].encode()), actual_bytes=len(r['output'].encode()),
                    trailing_bytes=len(r['output'].encode())-len(r['expected'].encode()),
                    prefix=r['output'].startswith(r['expected']), exact=r['expected']==r['output']) for r in old]
    proposed = []
    # Counts only. Not an authorized/sealed plan or runtime estimate guarantee.
    for epochs in (8, 16, 32):
        rows = 134 * epochs
        thought_train = rows * 17  # target-free16 + one noisy-target auxiliary.
        thought_eval = 112 * 2 * 16
        dec_eval = 112 * 2
        total = thought_train + thought_eval + dec_eval
        proposed.append(dict(
            epochs=epochs, TRAIN_row_exposures=rows, optimizer_attempts=math.ceil(rows/8),
            answer_target_bytes=after['target_answer_bytes']*epochs,
            train_thought_forward_calls=thought_train, baseline_final_thought_calls=thought_eval,
            baseline_final_D_calls=dec_eval, train_encoder_calls=0, train_D_calls=0,
            total_forward_calls=total,
            V110_total_seconds_scaled_by_call_count=204.47765636444092*total/12920,
            timing_note='Rough planning ratio only; backward, output length, checkpointing and environment costs differ.',
        ))
    result = dict(
        status='SAVED_JSON_STATISTICS_AND_PROSPECTIVE_ARITHMETIC_ONLY',
        objective='conditional answer meaning, not source identity',
        TRAIN_before_screen=before, TRAIN_selected=after, observed_DEV=dev,
        finite_reference_labels_reconstructed=len(all_rows),
        existing_V95_answer_ceiling=ceiling, prospective_costs=proposed,
        model_forward_calls=0, optimizer_steps=0, tensor_loads=0,
        E2_final_or_registry_access=False, new_data_or_external_downloads=False,
        input_pins=PINS,
        code_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    raw = (json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode()
    if len(raw) > 1024**2 or time.monotonic()-START > 15:
        raise RuntimeError('bounded arithmetic exceeded')
    OUT.mkdir()
    (OUT / 'saved_analysis.json').write_bytes(raw)
    print(json.dumps(dict(
        selected=after, observed_DEV={k: {field: v[field] for field in ('rows','target_answer_seen_in_selected_TRAIN','fact_seen_in_selected_TRAIN','pair_language_group_seen_in_selected_TRAIN','literal_selected_TRAIN_overlap')} for k,v in dev.items()},
        prospective_costs=proposed, seconds=time.monotonic()-START,
        evidence_SHA=hashlib.sha256(raw).hexdigest()), ensure_ascii=False))


if __name__ == '__main__':
    main()
