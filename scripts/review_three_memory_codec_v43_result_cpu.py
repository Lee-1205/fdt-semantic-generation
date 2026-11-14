"""ROOT saved-only review: parent reference + ONE updated fullstate, all80 reading.

Manual comments were authored after ROOT read every original/reconstruction pair.
No model construction/forward, no CUDA initialization/replay, no dataset expansion.
"""
import json
import math
import random
import statistics
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import text_generation_support_v1 as support
from codec_initial_reference_v42_gzip import load_initial_reference
from review_three_memory_codec_v33_result_cpu import raw_rows, logits_review
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count

def prefix_alignment(tensor, rows, indices):
    """Describe saved values; equality across inference batching is no invariant."""
    result = []
    for raw, index in zip(rows, indices):
        gold = tensor['target_ids'][index]
        prediction = tensor['normal'][index].argmax(-1)
        count = int((gold != 257).sum())
        generated = raw['generated']['token_ids']
        teacher_first = next((j for j in range(count) if int(prediction[j]) != int(gold[j])), None)
        bos_first = next((j for j in range(count) if generated[j] != int(gold[j])), None)
        result.append({'id': raw['id'], 'teacher_first_error': teacher_first,
                       'BOS_first_error': bos_first, 'same_first_error': teacher_first == bos_first})
    return result

PARTIAL = {7,17,20,27,31,36,40,59}
NONEXACT_WHOLE = {29,45}
COMMENTS = {0: 'Mr_Ogunjobi/Panthers team wish remains lost. Prior f-loop/no-EOS is gone; this alone is not semantic recovery.', 1: 'Crops/farm/Tillsonburg/Ontario propositions lost.', 3: 'Probable-vs-still-listed contrast lost.', 5: 'RE24/no-park-adjustment/SB-CS alternatives lost.', 7: 'April2015 instruction to place assembled hive boxes and frames survives; new locations corrupted. Partial only.', 9: 'Gameplay referent corrupted to gemit players; not a faithful proposition.', 17: 'Story on WallStreet survives; over-a-century duration corrupted to cutery. Partial only.', 19: 'Bride/alcohol-poisoning/dehydration-treatment relation lost.', 20: 'Setback for22-year-old survives; pre-season return/Euro2016 context lost. Partial only.', 21: 'Phenomenal-consciousness predicate lost.', 27: 'Bears fell to bottom of ranking survives; lowest scores across board lost. Partial only.', 29: 'Financial complication beyond most tea-garden labourers remains readable; labourers becomes misspelling laborerus. ROOT whole-proposition judgement with spelling error, never byte exact.', 30: 'Baby-shower gift-to-friend proposition lost.', 31: 'RobertOliver/photos/survival survives;1899 remains1989, producing1989-1902. Partial only; date error retained.', 32: 'Radio-etiquette/experience relation corrupted.', 34: 'Remanufactured-model offer and woman-did-not-like-it negation lost.', 36: 'Seeing grin on his face survives; when-he-said-it clause remains side at. Partial only.', 37: 'Nothing-more-to-it and means-that-much-to-you relation lost.', 40: 'Kaepernick negation fragment recovers; More importantly Wilson wins remains corrupted. Partial only.', 43: 'DavidCameron/big-society echoes proposition lost.', 45: 'Defines/linker options/compiler options list retained; linkers option inflection differs. Whole proposition, not byte exact.', 46: 'NFL remainsNBA and noble judgement lost; entity substitution not faithful.', 52: 'Portuguese150-companies/art-collecting propositions lost; new8/1000 numbers not source facts.', 55: 'OPI MyVampireIsBuff product identity lost.', 59: 'Works on all systems sold worldwide survives; regional-account/online-feature requirement lost. Partial only.'}
HELD = ['Personal culpability/American victims/trickery/deceit lost.', 'Fans/light-up flowers/anticipation/standing lost.', 'Disney/donut-sundae/surprised-unknown-to-speaker relation lost.', 'Ransom payments and restore-from-backups relation lost.', 'Deathstroke/page-panel/SladeWilson/Moscow relation lost.', 'Syria/Israel/stable-ceasefire/Assad-rule relation lost.', 'Failed sobriety-test/station-lobby/DUI charge lost.', 'Yanick stall/leaving without paying for taken goods lost.', 'No-explosion and causal abridgment-of-speech proposition lost.', 'Fabbri/Chiarelli/Bowman/three-month radar relation lost.', 'Second academic reviewing anonymous article copy lost.', 'Wubbels/lawyer/inaction/primary concern relation lost.', 'Gunmen holding vessels from six named countries lost; isolated Malaysia not a preserved proposition.', 'Jamaica/two-man bobsled/WinterOlympics/Sochi lost.', 'Janissa/preventing assassination using wall lost.', 'get_input_string/temp string/tokenize_string2/pointer-array relation lost.']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def recursive_count(state, key):
    total = 0
    while key in state:
        total += state[key]
        state = state.get('ancestry', {})
    return total


def main():
    torch.set_num_threads(2)
    run = ROOT/'runs/three_memory_codec_20261030_v43'
    plan, result, end = [read(run/n) for n in ('plan.json', 'result.json', 'supervisor/final_status.json')]
    terminal = read(run/'code_snapshot/root_terminal_native_before_analysis.json')
    live = read(run/'code_snapshot/root_initial_durable_and_training_observation.json')
    assert terminal['native'] == [] and terminal['all_owned_proxy_gone_before_analysis']
    assert end['status'] == 'COMPLETE' and end['exit_code'] == 0 and end['exit']['all_owned_processes_gone']
    assert end['launches'] == 1 and end['actual']['identity_verified']
    assert live['progress']['phase'] == 'two_epoch_broad_codec_training' and live['progress']['updates'] > 0
    assert any(x['ProcessId'] == 24896 and x['ParentProcessId'] == 18328 and x['CommandLine'] for x in live['native'])
    assert support.sha(run/'plan.json') == end['plan_sha256']
    assert support.sha(plan['source']) == plan['source_sha'] and support.core_sha(ROOT) == support.CORE_SHA
    assert all(support.sha(path) == digest for path, digest in plan['files'].items())

    parent = torch.load(plan['source'], weights_only=True, map_location='cpu')
    initial_path = run/'import/checkpoints/state_0000_reference_gzip.pt'
    initial = load_initial_reference(initial_path, plan['source'], plan['source_sha'], plan['source_algorithm'])
    assert support.state_sha(parent) == plan['source_algorithm']
    fields = ('model', 'optimizer', 'optimizer_parameter_names', 'scaler', 'frozen_sha',
              'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')
    imported_digest = support.state_sha({k: parent[k] for k in fields})
    assert imported_digest == support.state_sha({k: initial[k] for k in fields})
    assert imported_digest == initial['continuation']['imported_component_sha256']
    assert len(initial['optimizer']['state']) == 91
    assert initial['allocation_calls'] == {'encode': 0, 'denoise': 0, 'decoder': 0}
    assert not initial['evaluations']
    assert initial['ancestry']['source_schema'] == parent['schema']
    assert initial['ancestry']['continuation'] == parent['continuation']
    for key in ('history', 'orders', 'exposure', 'evaluations', 'ancestry', 'actual_calls'):
        assert support.state_sha(initial['ancestry'][key]) == support.state_sha(parent[key])
    scaler = dict(parent['scaler'])
    adam_steps = {i: int(v['step']) for i, v in parent['optimizer']['state'].items()}
    source_calls = dict(parent['actual_calls'])
    terminal_lrs = [g['lr'] for g in parent['optimizer']['param_groups']]
    base_lrs = [g['base_lr'] for g in parent['optimizer']['param_groups']]
    assert initial['continuation']['source_terminal_group_lrs'] == terminal_lrs
    del parent
    trained = {n for group in initial['optimizer_parameter_names'] for n in group}
    checkpoints = []
    # V43 intentionally has NO midpoint and at most ONE updated self-contained fullstate.
    updated_paths = list((run/'model/checkpoints').glob('state_*.pt'))
    assert updated_paths == [run/'model/checkpoints/state_8192.pt']
    assert not (run/'emergency').exists()
    for local, path in ((0, initial_path), (8192, updated_paths[0])):
        state = initial if local == 0 else torch.load(path, weights_only=True, map_location='cpu')
        marker = read(path.with_suffix('.json'))
        digest, algorithm = support.sha(path), support.state_sha(state)
        assert digest == marker['sha256'] and algorithm == marker['algorithm_sha256']
        assert state['schema'] == 'fdt_three_memory_language_codec_v43_fullstate'
        assert state['attempts'] == local and state['updates'] == local-state['skips']
        assert state['cursor'] == 8*local and not state['gradients']
        for key in ('continuation', 'migration', 'ancestry'):
            assert state[key] == initial[key]
        assert [h['rows'] for h in state['history']] == state['orders'][:local]
        assert len(state['history']) == local
        assert state['frozen_sha'] == initial['frozen_sha']
        assert all(torch.equal(value, initial['model'][name]) for name, value in state['model'].items() if name not in trained)
        assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for item in state['optimizer']['state'].values()
                   for v in item.values() if isinstance(v, torch.Tensor) and v.is_floating_point())
        assert all(int(v['step']) == adam_steps[i]+state['updates'] for i, v in state['optimizer']['state'].items())
        scale, tracker = scaler['scale'], scaler['_growth_tracker']
        skips = []
        for index, row in enumerate(state['history']):
            assert row['attempt'] == index+1 and row['denoise_calls'] == row['copy_bytes'] == 0
            assert row['mask_probability'] == .5
            assert abs(row['loss']-.5*(row['clean_CE']+row['corrupted_prefix_CE'])) < 1e-5
            if row['AMP_skip']:
                scale *= scaler['backoff_factor']; tracker = 0; skips.append(index+1)
            else:
                tracker += 1
                if tracker == scaler['growth_interval']:
                    scale *= scaler['growth_factor']; tracker = 0
            assert row['scale'] == scale
        assert state['scaler'] == dict(scaler, scale=scale, _growth_tracker=tracker)
        assert [g['lr'] for g in state['optimizer']['param_groups']] == terminal_lrs
        assert [g['base_lr'] for g in state['optimizer']['param_groups']] == base_lrs
        assert all(row['optimizer_group_lrs'] == terminal_lrs for row in state['history'])
        for label, digest_value in state['evaluations'].items():
            path_value = run/(label if label.endswith(('.pt', '.json')) else label+'.json')
            assert support.sha(path_value) == digest_value
        checkpoints.append({'attempt': local, 'updates': state['updates'], 'AMP_skips': skips,
            'bytes': path.stat().st_size, 'sha256': digest, 'algorithm_sha256': algorithm,
            'stage': state['worker_stage'], 'scaler': state['scaler'], 'standalone': local != 0})
    assert checkpoints[-1]['sha256'] == result['latest']['sha256']
    assert checkpoints[-1]['algorithm_sha256'] == result['latest']['algorithm_sha256']
    assert recursive_count(state, 'attempts') == result['global_attempts'] == 161764
    assert recursive_count(state, 'updates') == result['global_updates'] == 161704
    expected_orders = []
    for seed in (42001, 42002):
        order = list(range(32768)); random.Random(seed).shuffle(order)
        expected_orders.extend(order[i:i+8] for i in range(0, len(order), 8))
    assert state['orders'] == expected_orders
    assert state['exposure'] == result['exposure']
    assert state['exposure']['target_UTF8_bytes'] == 5852956
    assert state['exposure']['selected_rows'] == state['exposure']['processed_rows'] == 65536
    for key in ('masked_prefix_bytes', 'eligible_prefix_bytes', 'decoder_supervised_symbol_occurrences'):
        assert state['exposure'][key] == sum(row[key] for row in state['history'])
    assert state['allocation_calls'] == result['actual_calls']
    assert state['actual_calls'] == result['cumulative_calls']
    online = [{'attempts': [start+1, start+1024], **{key: statistics.mean(
        row[key] for row in state['history'][start:start+1024])
        for key in ('loss', 'clean_CE', 'corrupted_prefix_CE')}} for start in range(0, 8192, 1024)]
    external = state['external_closed_assessment_calls']
    assert external == plan['external_closed_assessment_calls'] and external['total'] == 1936
    del state, initial

    focused_features = read(Path(plan['focused_artifact'])/'features.json')
    focused_targets = read(Path(plan['focused_artifact'])/'targets.json')
    features = read(Path(plan['artifact'])/'features.json')['dev']
    raw_train, raw_held = [read(run/(name+'_reconstruction.json')) for name in ('focused_final', 'held_final')]
    train, train_summary = raw_rows(raw_train, focused_features)
    held, held_summary = raw_rows(raw_held, [features[i] for i in plan['generation_indices']], HELD)
    assert {x['index'] for x in train if not x['exact']} == set(COMMENTS)
    for row in train:
        index = row['index']
        whole = row['exact'] or index in NONEXACT_WHOLE
        row.update(ROOT_reading='Exact original span; all source details and punctuation identical.'
            if row['exact'] else COMMENTS[index], whole_source_span_faithful=whole,
            confirmed_partial_proposition=index in PARTIAL,
            meaning_grade='exact_source_span' if row['exact'] else 'whole_proposition_typo_or_inflection_changed'
            if index in NONEXACT_WHOLE else 'partial_proposition' if index in PARTIAL else 'not_faithful')
    assert train_summary['exact'] == 39 and held_summary['exact'] == 0
    assert sum(row['whole_source_span_faithful'] for row in train) == 41
    assert len(PARTIAL) == 8
    tensor, focus_metrics = logits_review(run/'focused_final_logits.pt', focused_targets, result['focused_final'])
    held_tensor = torch.load(run/'held_final_logits.pt', weights_only=True, map_location='cpu')
    alignment = {'focused64': prefix_alignment(tensor, raw_train, range(64)),
                 'held16': prefix_alignment(held_tensor, raw_held, plan['generation_indices'])}
    write(run/'root_teacher_BOS_first_error_mismatch.json', {'alignment': alignment,
        'mismatch_count': sum(not row['same_first_error'] for rows in alignment.values() for row in rows),
        'AR_logits_stored': False, 'sole_cause_proven': False, 'additional_model_calls': 0})
    memory = read(run/'root_three_memory_readouts_review.json')
    assert memory['status'] == 'V43_SAVED_THREE_MEMORY_READ_PATHS_VERIFIED'
    assert memory['plan_sha256'] == end['plan_sha256']
    calls = {'encode': 8192+8+16+32+4+6, 'denoise': 0,
             'decoder': 16384+24+96+6+stop_count(raw_train)+stop_count(raw_held)}
    assert calls == result['actual_calls'] and sum(calls.values()) == 26626
    assert result['cumulative_calls'] == {key: source_calls[key]+calls[key] for key in calls}
    old = read(ROOT/'runs/three_memory_codec_20261030_v42/root_reconstruction_similarity.json')
    gains = [i for i, row in enumerate(train) if row['exact'] and not old['focused64'][i]['exact']]
    losses = [i for i, row in enumerate(train) if not row['exact'] and old['focused64'][i]['exact']]
    assert not torch.cuda.is_initialized()
    out = {'status': 'V43_COMPLETE_THIRTEENTH_FOURTEENTH_LOW_RATE_EPOCHS_ROOT_REVIEWED',
        'plan_sha256': end['plan_sha256'], 'fullstates': checkpoints,
        'initial_checkpoint_parent_required': True, 'updated_checkpoint_self_contained': True,
        'one_updated_fullstate_no_midpoint_verified': True,
        'source_all91_model_Adam_scaler_ALLstoredRNG_exact': True,
        'source_component_sha256': imported_digest, 'all_model_Adam_finite': True,
        'frozen_original_reasoning_language_prefix_basis_unchanged': True,
        'constant_source_terminal_group_LRs_verified_every_attempt': True,
        'source_terminal_group_lrs': terminal_lrs, 'source_base_lrs_unchanged': base_lrs,
        'new_attempts': 8192, 'new_updates': 8189, 'AMP_skips': checkpoints[-1]['AMP_skips'],
        'global_attempts': result['global_attempts'], 'global_updates': result['global_updates'],
        'actual_calls': calls, 'total_new_calls': 26626, 'cumulative_calls': result['cumulative_calls'],
        'external_closed_assessment_calls': external, 'exposure': result['exposure'],
        'online_loss_blocks': online, 'focused64_reconstruction': train_summary,
        'held16_reconstruction': held_summary, 'focused64_whole_proposition_faithful': 41,
        'focused64_nonexact_whole_proposition_faithful': 2, 'focused64_partial_propositions': 8,
        'held_whole_faithful': 0, 'held_confirmed_partial_propositions': 0, 'ROOT_read_all80': True,
        'focused64_exact_gains_vs_V42': gains, 'focused64_exact_losses_vs_V42': losses,
        'same_first_teacher_BOS_error_position_all80': all(
            row['same_first_error'] for rows in alignment.values() for row in rows),
        'teacher_BOS_first_error_mismatch_count': sum(
            not row['same_first_error'] for rows in alignment.values() for row in rows),
        'old_prefix_equality_assumption_removed_without_model_rerun': True,
        'saved_teacher_BOS_prefix_alignment': alignment,
        'saved_logit_metrics': {'focused_final': focus_metrics,
                               'held_final': memory['held256_saved_logit_metrics']},
        'memory_readout_review_sha256': support.sha(run/'root_three_memory_readouts_review.json'),
        'held16_normal_NLL': memory['normal_same16_NLL'],
        'held16_memory_ablations': memory['each_memory_disabled_same16'],
        'all_owned_proxy_gone_before_analysis': True, 'ROOT_worker_live_observed': True,
        'additional_model_construction_forward': 0, 'CUDA_replay_performed': False,
        'denoise_calls': 0, 'literal_copy_bytes': 0, 'E2_final_accessed': False,
        'conclusion': 'At preserved terminal group LRs, TRAIN-reference exact31->39 and whole propositions33->41;8exact gains/0losses. All80UTF8EOS recovers; prior f-loop vanishes but row0 meaning still lost. Held teacher byte/EOS80.2258->80.3163%, while NLL.745030->.778913 worsens and held whole fidelity remains0/16. Preserve both improvement and regression; no independent LR-causality, perfect codec, or global FDT failure claim.'}
    write(run/'root_result.json', out)
    write(run/'root_reconstruction_similarity.json', {'focused64': train, 'held16': held,
        'summary': {'focused64': train_summary, 'held16': held_summary}})
    lines = ['# V43 ROOT reads all80 SAME-input source/reconstructions', '', out['conclusion']]
    for label, rows in (('TRAIN_REFERENCE', train), ('HELD', held)):
        for row in rows:
            lines += ['', f"## {label} {row['index']}", f"Source: {row['source']}",
                      f"Reconstruction: {row['reconstruction']}", f"ROOT: {row['ROOT_reading']}"]
    with (run/'root_all80_reading.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps({key: out[key] for key in ('status', 'fullstates', 'focused64_reconstruction',
        'held16_reconstruction', 'focused64_exact_gains_vs_V42', 'focused64_exact_losses_vs_V42',
        'online_loss_blocks', 'total_new_calls')}))


if __name__ == '__main__':
    main()
