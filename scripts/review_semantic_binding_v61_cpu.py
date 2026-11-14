"""Independent saved-state arithmetic; no encoder/binder/decoder forward calls."""
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support


def predicted(raw, rows, hierarchical=False):
    relation = raw['relation_logits'].float().argmax(-1)
    if hierarchical:
        # A prospective conditional decision, NOT a replacement for old metrics.
        polarity = torch.where(relation == 1, (raw['negative_logits'] <= 0).long(), 1)
        quantity = torch.where(relation == 2, raw['quantity_logits'].argmax(-1) + 1, 0)
    else:
        polarity = raw['polarity_probabilities'].argmax(-1)
        quantity = raw['quantity_probabilities'].argmax(-1)
    result = []
    for i, row in enumerate(rows):
        b = row['prompt'].encode('utf8')
        roles = [None if a < 0 else b[a:z].decode('utf8') for a, z in raw['spans'][i].tolist()]
        result.append(dict(relation=int(relation[i]), polarity=int(polarity[i]),
                           quantity=int(quantity[i]), roles=dict(zip(('primary', 'recipient', 'theme'), roles))))
    return result


def scores(predictions, targets, raw):
    result = {}
    for kind in range(3):
        ids = [i for i, t in enumerate(targets) if t['meaning']['relation'] == kind]
        truth = [targets[i]['meaning'] for i in ids]
        got = [predictions[i] for i in ids]
        present = [(p['roles'][key] == t['roles'][key]) for p, t in zip(got, truth)
                   for key in t['roles'] if t['roles'][key] is not None]
        result[str(kind)] = dict(rows=len(ids), exact_facts=sum(p == t for p, t in zip(got, truth)),
            relation_correct=sum(p['relation'] == kind for p in got), present_role_correct=sum(present),
            present_role_total=len(present), polarity_correct=sum(p['polarity'] == t['polarity'] for p, t in zip(got, truth)),
            quantity_correct=sum(p['quantity'] == t['quantity'] for p, t in zip(got, truth)))
        if kind == 1:
            neg = (raw['negative_logits'][ids] <= 0).long()
            result[str(kind)]['conditional_polarity_head_correct'] = int((neg == torch.tensor([t['polarity'] for t in truth])).sum())
        if kind == 2:
            q = raw['quantity_logits'][ids].argmax(-1) + 1
            result[str(kind)]['conditional_quantity_head_correct'] = int((q == torch.tensor([t['quantity'] for t in truth])).sum())
    return result


def validate_probabilities(raw):
    p = raw['relation_logits'].float().softmax(-1)
    negative = p[:, 1] * raw['negative_logits'].float().sigmoid()
    expected_p = torch.stack((negative, 1 - negative), -1)
    expected_q = torch.cat((1 - p[:, 2, None], p[:, 2, None] * raw['quantity_logits'].float().softmax(-1)), -1)
    assert torch.allclose(p, raw['relation_probabilities'], atol=1e-6, rtol=1e-6)
    assert torch.allclose(expected_p, raw['polarity_probabilities'], atol=1e-6, rtol=1e-6)
    assert torch.allclose(expected_q, raw['quantity_probabilities'], atol=1e-6, rtol=1e-6)
    # Independently enumerate every ordered start/end pair, including NULL.
    n = raw['start_logits'].shape[-1] - 1
    pair = raw['start_logits'][..., :n, None] + raw['end_logits'][..., None, :n]
    pair = pair.masked_fill(torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1), -torch.inf)
    logz=torch.logaddexp(pair.flatten(-2).logsumexp(-1),raw['start_logits'][...,-1]+raw['end_logits'][...,-1])
    assert torch.allclose(logz,raw['joint_logZ'],atol=3e-5,rtol=1e-6)
    probabilities=(pair-logz[...,None,None]).exp()
    occupied=probabilities.cumsum(-2).flip(-1).cumsum(-1).flip(-1).diagonal(dim1=-2,dim2=-1)
    assert torch.allclose(occupied*raw['valid'][:,None],raw['role_degree'][:,:3],atol=2e-5,rtol=2e-5)
    saved = raw['spans']
    best_score = pair.flatten(-2).max(-1).values
    null_score = raw['start_logits'][..., -1] + raw['end_logits'][..., -1]
    null = null_score >= best_score
    assert torch.equal(saved[..., 0] < 0, null)
    present = ~null
    chosen = pair[present, saved[..., 0][present], saved[..., 1][present] - 1]
    assert torch.equal(chosen, best_score[present])


def main():
    torch.set_num_threads(4)
    run = ROOT / 'runs/semantic_binding_20261111_v61'
    plan = io.load_json(run / 'plan.json')
    terminal = io.load_json(run / 'final_status.json')
    native = io.load_json(run / 'root_terminal_native.json')
    assert terminal['status'] == 'COMPLETE' and terminal['exit']['all_owned_processes_gone']
    assert native['all_owned_gone'] and native['native_python_count'] == 0
    states = {}
    checkpoint_manifest = []
    for directory, steps in [('seed', [0]), ('initial', [0]), ('model', [1, 512, 1024])]:
        for step in steps:
            path = run / directory / 'checkpoints' / f'state_{step:04d}.pt'
            meta = io.load_json(path.with_suffix('.json'))
            assert support.sha(path) == meta['sha256'] and path.stat().st_size == meta['bytes']
            state = torch.load(path, weights_only=True, map_location='cpu')
            assert support.state_sha(state) == meta['algorithm_sha256']
            assert state['schema'] == 'fdt_semantic_binding_v61_fullstate'
            assert state['graph_contract'] == 'coupled_spans_fourrole_memory_readout'
            assert state['attempts'] == state['updates'] == step and state['skips'] == 0
            assert all(int(v['step']) == step for v in state['optimizer']['state'].values())
            assert state['cursor'] == sum(map(len, state['orders'][:step]))
            assert [h['rows'] for h in state['history']] == state['orders'][:step]
            assert not state['gradients']
            states[(directory, step)] = state
            checkpoint_manifest.append(dict(directory=directory, step=step, sha256=meta['sha256'], algorithm_sha256=meta['algorithm_sha256']))
    initial, final = states[('initial', 0)], states[('model', 1024)]
    assert not initial['optimizer']['state']
    assert final['cursor'] == 16384 and final['exposure'] == dict(row_selections=16384, cached_source_UTF8_bytes=1198324)
    assert final['scaler']['scale'] == 1024 and final['scaler']['_growth_tracker'] == 1024
    assert final['calls'] == dict(encode=256, bind=1084, denoise=0)
    rng_digests = {key: dict(initial=support.state_sha({key:initial[key]}), final=support.state_sha({key:final[key]}))
                   for key in ('torch_rng','cuda_rng','python_rng','numpy_rng')}
    warm = torch.load(plan['binder_source'], weights_only=True, map_location='cpu')
    for key, value in warm['binder'].items():
        assert torch.equal(initial['binder'][key], value), key
    new_keys=set(initial['binder'])-set(warm['binder'])
    assert len(new_keys)==12 and all(k.startswith(('sequence.','sequence_out.','entity_inside.')) for k in new_keys)
    assert all(v.count_nonzero()==0 for k,v in initial['binder'].items() if k.startswith(('sequence_out.','entity_inside.')))
    assert torch.equal(initial['binder']['persistent_prior'], final['binder']['persistent_prior'])
    cache_manifest = {name: support.sha(name) for name in final['cache_refs']}
    assert cache_manifest == final['cache_refs']
    for key, sha_key in [('source', 'source_sha'), ('head_source', 'head_sha'), ('binder_source', 'binder_sha')]:
        assert support.sha(plan[key]) == plan[sha_key]
    data = Path(plan['data'])
    targets, rows = io.load_json(data / 'dev_targets.json'), io.load_json(data / 'dev_features.json')
    groups = io.load_json(data / 'dev_groups.json')
    features = torch.load(plan['reused_dev_features'], weights_only=True, map_location='cpu')
    raw = torch.load(run / 'raw_final.pt', weights_only=True, map_location='cpu')
    baseline = torch.load(run / 'raw_baseline.pt', weights_only=True, map_location='cpu')
    validate_probabilities(raw)
    validate_probabilities(baseline)
    pfinal, pbase = predicted(raw, rows), predicted(baseline, rows)
    saved = io.load_json(run / 'final_predictions.json')
    assert pfinal == [r['predicted'] for r in saved]
    gold = [t['meaning'] for t in targets]
    assert sum(a == b for a, b in zip(pfinal, gold)) == 29
    # Four-role N/D independently rebuilt from actual saved neural degrees.
    weights = raw['role_degree'].double()[..., None] * features['mu'].double()[:, None] * features['valid'][:, None, :, None]
    expected_n = torch.einsum('brta,btv->brav', weights, raw['projected_values'].double())
    expected_d = weights.sum(2)[..., None]
    assert torch.allclose(raw['role_N'], expected_n, atol=1e-8)
    assert torch.allclose(raw['role_D'], expected_d, atol=1e-10)
    controls = {}
    for bank in ('persistent', 'contextual', 'exact'):
        other = torch.load(run / ('raw_read_off_' + bank + '.pt'), weights_only=True, map_location='cpu')
        delta = {k: float((other[k] - raw[k][:64]).abs().max()) for k in ('start_logits', 'end_logits', 'relation_logits', 'negative_logits', 'quantity_logits')}
        assert delta['start_logits'] == delta['end_logits'] == 0
        assert any(delta[k] > 0 for k in ('relation_logits', 'negative_logits', 'quantity_logits'))
        controls[bank] = dict(logit_deltas=delta, bound_memory_value_delta=float((other['role_values'] - raw['role_values'][:64]).abs().max()))
    # Compare paraphrase and same-group counterfact using pre-factor role state,
    # separately from the vector which also includes predicted head probabilities.
    ranking = {}
    for key in ('role_values', 'semantic_vector'):
        z = torch.nn.functional.normalize(raw[key].flatten(1).float(), dim=-1)
        result = collections.defaultdict(list)
        for group in groups:
            a, b, c, d = group
            kind = str(targets[a]['meaning']['relation'])
            same = float(((z[a] * z[b]).sum() + (z[a] * z[c]).sum()) / 2)
            changed = float((z[a] * z[d]).sum())
            result[kind].append((same, changed))
        ranking[key] = {kind: dict(groups=len(v), same_closer=sum(s > d for s, d in v), mean_same=sum(s for s, _ in v)/len(v),
            mean_counterfact=sum(d for _, d in v)/len(v)) for kind, v in result.items()}
    id_to_index = {row['id']: i for i, row in enumerate(rows)}
    swaps = io.load_json(run / 'event_interchange.json')
    swap_summary = {}
    for axis in ('polarity', 'quantity'):
        cases = [s for s in swaps if s['axis'] == axis]
        eligible, eligible_ok = 0, 0
        donor_ok, source_kept, relation_ok = 0, 0, 0
        for case in cases:
            a, b = id_to_index[case['source_id']], id_to_index[case['donor_id']]
            assert case['source_fact'] == gold[a] and case['donor_fact'] == gold[b]
            factor = case['factor_logits']
            relation_probs = torch.tensor(factor['relation_logits']).softmax(-1)[0]
            negative_prob = relation_probs[1] * torch.tensor(factor['negative_logits']).sigmoid()[0]
            quantity_probs = torch.cat((1 - relation_probs[2:3], relation_probs[2] * torch.tensor(factor['quantity_logits']).softmax(-1)[0]))
            assert case['predicted_relation'] == int(relation_probs.argmax())
            assert case['predicted_polarity'] == int(torch.stack((negative_prob, 1 - negative_prob)).argmax())
            assert case['predicted_quantity'] == int(quantity_probs.argmax())
            predicted_axis = case['predicted_' + axis]
            donor_ok += predicted_axis == gold[b][axis]
            source_kept += predicted_axis == gold[a][axis]
            relation_ok += case['predicted_relation'] == gold[b]['relation']
            normal_both = pfinal[a][axis] == gold[a][axis] and pfinal[b][axis] == gold[b][axis]
            eligible += normal_both
            eligible_ok += normal_both and predicted_axis == gold[b][axis]
        swap_summary[axis] = dict(cases=len(cases), donor_axis_correct=donor_ok, source_axis_kept=source_kept,
            relation_correct=relation_ok, normal_both_axis_correct=eligible, donor_correct_in_eligible=eligible_ok)
    examples = []
    for kind in range(3):
        ids = [i for i, t in enumerate(targets) if t['meaning']['relation'] == kind]
        selected = [i for i in ids if pfinal[i] == gold[i]][:2]
        selected += [i for i in ids if pfinal[i] != gold[i]][:6]
        for i in selected:
            examples.append(dict(id=rows[i]['id'], source=rows[i]['prompt'], predicted=pfinal[i], gold=gold[i], exact=pfinal[i] == gold[i]))
    training = {key: sum(h['losses'][key] for h in final['history'][-256:])/256 for key in final['history'][0]['losses']}
    report = dict(status='INDEPENDENT_V61_SAVED_REVIEW_PASS', checkpoints=checkpoint_manifest,
        all5_states_SHA_safeload_digest=True, fresh_Adam_all_steps_scaler_ALLRNG_cursor_orders_verified=True,
        warm_all_old_tensors_exact_new_residual_and_inside_zero=True, prior_source_codec_old_binder_SHA_unchanged=True,
        cache_and_raw_SHA=cache_manifest, fourrole_ND_reconstructed=True, exhaustive_ordered_span_scores_verified=True,
        factor_probability_arithmetic_verified=True,joint_partition_and_occupancy_verified=True,stored_RNG_digests=rng_digests,calls=final['calls'], exposure=final['exposure'],
        baseline_per_relation=scores(pbase, targets, baseline), final_per_relation=scores(pfinal, targets, raw),
        prospective_conditional_decoding=scores(predicted(raw, rows, True), targets, raw),
        last_epoch_losses=training, final_role_start_gain=final['binder']['role_start_gain'].tolist(),
        final_role_end_gain=final['binder']['role_end_gain'].tolist(), read_channel_controls=controls,
        paraphrase_counterfact_cosine_ranking=ranking, event_interchanges=swap_summary, examples=examples,
        C_free_bytes=shutil.disk_usage(ROOT.anchor).free, additional_model_forwards=0, CUDA_initialized=torch.cuda.is_initialized(),
        interpretation='Memory-only readout is active; restricted held fact accuracy remains low. Three-bank read-channel effects do not prove unique semantic content. Conditional decoding is saved-logit arithmetic, not a new model trial. No language realization or meaning denoising.')
    assert not report['CUDA_initialized']
    io.atomic_json(run / 'root_saved_review.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cache_and_raw_SHA','examples','checkpoints')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
