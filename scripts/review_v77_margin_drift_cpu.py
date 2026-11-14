"""Read saved logits and final biases only; never execute a neural forward."""
import collections, gc, hashlib, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    run = ROOT / 'runs/update_balance_20261113_v77'
    assert io.load_json(run / 'root_saved_review.json')['status'] == 'ROOT_V77_SAVED_REVIEW_PASS'
    plan = io.load_json(run / 'plan.json')
    versions = dict(V75=Path(plan['baseline_run']), V76=ROOT / 'runs/joint_quantity_20261113_v76', V77=run)
    biases, active_counts, pins = {}, {}, {}
    final_key = 'readout.readout.net.2.bias'
    for name, folder in versions.items():
        path = folder / 'model/runtime.pt'
        env = torch.load(path, map_location='cpu', weights_only=True)
        biases[name] = env['active_weights'][final_key].float().reshape(16, 258)
        active_counts[name] = sum(t.numel() for t in env['active_weights'].values())
        pins[str(path)] = sha(path)
        assert len(env['active_weights']) == 7 and active_counts[name] == 2908192
        del env
    rows = io.load_json(Path(plan['old_eval_data']) / 'long_features.json')[:2]
    labels = io.load_json(Path(plan['old_eval_data']) / 'long_targets.json')[:2]
    summary, cards = {}, []
    for row_index, (row, target) in enumerate(zip(rows, labels)):
        source = row['prompt'].encode()
        role = ['ordinary'] * len(source)
        for event in target['events']:
            start, end = event['byte_start'], event['byte_end']
            match = PATTERN.fullmatch(source[start:end].decode())
            assert match
            for key, value in match.groupdict().items():
                if value is not None:
                    a, b = match.span(key)
                    role[start+a:start+b] = [key] * (b-a)
        packets, logits, prediction, margins = {}, {}, {}, {}
        for name, folder in versions.items():
            path = folder / f'raw_long_{row_index:02d}.pt'
            record = io.load_json(folder / 'long_reconstructions.json')[row_index]
            pins[str(path)] = sha(path)
            assert pins[str(path)] == record['raw_sha256']
            packet = torch.load(path, map_location='cpu', weights_only=True)
            y = packet['targets'].flatten()[:len(source)]
            assert torch.equal(y, torch.tensor(list(source))) and packet['Exact_entries'] == 0
            z = packet['logits'].float().reshape(-1, 258)[:len(source)]
            del packet
            gold = z.gather(1, y[:, None]).flatten()
            competitor = z.clone(); competitor.scatter_(1, y[:, None], -torch.inf)
            margins[name] = gold - competitor.max(1).values
            logits[name], prediction[name] = z, z.argmax(-1)
            confusion, by_role, plus_slots = collections.Counter(), collections.Counter(), collections.Counter()
            for position in torch.where(prediction[name] != y)[0].tolist():
                a, b = source[position], int(prediction[name][position])
                confusion[f'{a}->{b}'] += 1
                by_role[role[position]] += 1
                if b == 43: plus_slots[str(position % 16)] += 1
            digits = (y >= 48) & (y <= 57)
            summary[f'{name}_row{row_index}'] = dict(
                body_raw_errors=int((prediction[name] != y).sum()),
                wrong_plus=int(((prediction[name] == 43) & (y != 43)).sum()),
                digit_to_plus=int(((prediction[name] == 43) & digits).sum()),
                error_fields=dict(by_role), confusions=dict(confusion), wrong_plus_patch_roles=dict(plus_slots),
                digit_min_margin=float(margins[name][digits].min()),
                digit_margin_under_1=int((margins[name][digits] < 1).sum()),
                digit_positions=int(digits.sum()))
        union = sorted(set().union(*(set(torch.where(prediction[k] != y)[0].tolist()) for k in versions)))
        for position in union:
            gold = source[position]
            current = int(prediction['V77'][position])
            competitor = current if current != gold else int(logits['V77'][position].masked_fill(torch.arange(258) == gold, -torch.inf).argmax())
            total_gap_delta = float((logits['V77'][position, competitor] - logits['V77'][position, gold]) - (logits['V75'][position, competitor] - logits['V75'][position, gold]))
            bias_gap_delta = float((biases['V77'][position % 16, competitor] - biases['V77'][position % 16, gold]) - (biases['V75'][position % 16, competitor] - biases['V75'][position % 16, gold]))
            cards.append(dict(row=row_index, byte=position, patch_role=position % 16, source_field=role[position], gold=gold,
                predicted={k:int(prediction[k][position]) for k in versions},
                gold_margin={k:float(margins[k][position]) for k in versions},
                V77_competitor=competitor, V75_to_V77_competitor_gap_delta=total_gap_delta,
                final_bias_gap_delta=bias_gap_delta, other_path_plus_rounding_gap_delta=total_gap_delta-bias_gap_delta,
                source_context=source[max(0,position-48):position+49].decode(errors='replace')))
        del logits, prediction, margins, competitor
        gc.collect()
    newly_wrong = [c for c in cards if c['predicted']['V75'] == c['gold'] != c['predicted']['V77']]
    plus_new = [c for c in newly_wrong if c['predicted']['V77'] == 43]
    assert len(plus_new) > 0
    decomposition = dict(newly_wrong_positions=len(newly_wrong), newly_wrong_plus=len(plus_new),
        median_total_competitor_gap_delta=statistics.median(c['V75_to_V77_competitor_gap_delta'] for c in plus_new),
        median_final_bias_gap_delta=statistics.median(c['final_bias_gap_delta'] for c in plus_new),
        all_plus_new_require_other_path_change=all(c['V75_to_V77_competitor_gap_delta'] > abs(c['final_bias_gap_delta']) for c in plus_new))
    metadata = dict(plan_trainable_declared=plan['actual_trainable_parameters'],
        actual_active_parameters=active_counts['V77'], actual_active_tensors=7,
        inherited_estimate_fields={k:v for k,v in plan.items() if k.startswith('output_estimate_including')},
        authoritative_output_estimate=sum(plan['output_estimate_detail'].values()),
        hard_output_cap=plan['max_output_bytes'],
        interpretation='Stale descriptive fields inherited by dict(old); actual optimizer ownership and wired disk/call caps independently verified. Sealed plan preserved; future plans must not inherit unchecked descriptive fields.')
    assert metadata['plan_trainable_declared'] != metadata['actual_active_parameters']
    assert not torch.cuda.is_initialized()
    report = dict(status='ROOT_V77_SAVED_MARGIN_DRIFT_PASS', summaries=summary,
        active_parameters=active_counts, final_bias_decomposition=decomposition, plan_metadata_discrepancy=metadata,
        raw_and_state_SHA=pins, cards=cards, extra_neural_forwards=0, optimizer_steps=0, CUDA_initialized=False,
        limitations='Saved BF16-rounded logits and final-layer bias arithmetic, not parameter-gradient or causal mediation attribution. Same observed two long sources; lexical slot preservation is not an abstract semantic certificate. No logits/output edited or held facts used for TRAIN.')
    io.atomic_json(run / 'root_saved_margin_drift.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cards','raw_and_state_SHA')}))


if __name__ == '__main__': main()
