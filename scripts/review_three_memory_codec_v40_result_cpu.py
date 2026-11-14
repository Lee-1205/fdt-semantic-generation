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
from codec_initial_reference_v39 import load_initial_reference
from review_three_memory_codec_v33_result_cpu import raw_rows, logits_review
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count
from review_three_memory_codec_v39_result_cpu import prefix_alignment

PARTIAL = {7, 15, 17, 20, 22, 24, 41, 45, 49, 50, 61}
NONEXACT_WHOLE = {57}
COMMENTS = {
    1: 'Crops/farm/Tillsonburg/Ontario propositions are lost.',
    3: 'Probable remains, but still-listed assertion and contrast are lost.',
    5: 'RE24 recalculation, no park adjustment and SB/CS alternative are lost.',
    7: 'Placement of assembled hive boxes/new locations remains recognizable; frames becomes farmers and the coordination is corrupted. Partial only.',
    9: 'Probably-here-for lead remains but gameplay itself is replaced with gemist player.',
    11: 'Sealed container/take-with-you-anywhere relation is lost.',
    13: 'Drag-that-out lead remains; them/kicking-and-screaming participants and manner are corrupted.',
    15: 'April Ryan/20-year veteran remains; press corps, does-not-have-an-agenda negation and attribution are lost.',
    17: 'Wall Street/story remains; duration over a century is corrupted to over a cutenter. Partial only.',
    19: 'Treating his bride for alcohol poisoning/dehydration is lost.',
    20: 'Setback for22-year-old/returned remains; pre-season, summer and Euro2016 causal/timing details are lost.',
    21: 'What-it-is-like/how-it-feels lead remains, but phenomenally conscious predicate is lost.',
    22: 'Incidents of anti-Semitic behaviour proposition survives; Baroness Royall confirmation/report attribution is corrupted. Partial only.',
    24: 'Workers self-management/Ceramic Workers Union participation survives; fundamental-role predicate is corrupted. Partial only.',
    27: 'Bears/bottom-of-rankings/lowest-scores-across-board proposition is lost.',
    28: 'Camp remains but too-early/jump-the-gun relation is lost.',
    30: 'Thoughtful baby-shower gift-to-a-friend proposition is lost.',
    32: 'Radio etiquette/experience relation is corrupted.',
    33: 'Crimes in2008/age17 becomes2017/7 and other corruption; factual identity/time loss.',
    34: 'Remanufactured-model offer and woman-did-not-like-it negation are lost.',
    40: 'Kaepernick negation and Wilson wins contrast are corrupted.',
    41: 'McManus did-not-say remains; cause of shooting is lost. Partial only.',
    43: 'David Cameron/big society/faint-echo relation is lost.',
    45: 'Defines/linker/compiler list remains recognizable, but linker/compiler options are corrupted. Partial only.',
    47: 'Belichick remains, but praise of team and liberal smiles are not recoverable confidently from corrupted wording.',
    49: 'Psaki reply to Fox News reporter survives; James Rosen surname changes to Reports. Partial only, no exact identity fidelity.',
    50: 'Confidence/good-intentioned people remains; bringing it to reality predicate is corrupted. Partial only.',
    52: 'Portuguese150-companies/art-collecting propositions are lost.',
    55: 'Base-of lead remains, but OPI My Vampire is Buff product identity is lost.',
    57: 'Turning herself in and stating that her marriage to Danil was fake both survive. Commas move; byte-exact reconstruction fails, but the whole original proposition is preserved on ROOT reading.',
    58: 'Built-in Bluetooth survives; routing everything through a speaker is lost.',
    59: 'Worldwide systems/regional account/online-feature requirement is lost.',
    61: 'Playing in big markets survives; making big bets is corrupted to making bebits. Partial only.',
}
HELD = [
    'Personal culpability/American victims/trickery/deceit are lost.',
    'Fans waving light-up flower accessories/anticipation/standing are lost.',
    'Disney donut sundae/unknown-to-speaker/surprised question are lost.',
    'Few ransom payments/restoring systems from backups are lost.',
    'Deathstroke/page-panel/assault team/Slade Wilson/Moscow are lost.',
    'Syria/Israel/ceasefire/Assad-rule relations are lost.',
    'Failed sobriety test/station lobby/DUI charge are lost.',
    'Yanick stall/leaving without paying for taken goods is lost.',
    'No-explosion predicate and causal speech-restriction proposition are lost. V39 recognizable no-explosion core is no longer recognizable.',
    'Fabbri/Chiarelli/Bowman/three-month radar relation are lost.',
    'Second academic reviewing anonymous copy of article is lost.',
    'Wubbels/lawyer/inaction/primary concern relation is lost.',
    'Gunmen holding vessels from six named countries are lost.',
    'Jamaica/two-man bobsled/Winter Olympics/Sochi are lost.',
    'Janissa preventing assassination using her wall is lost.',
    'get_input_string temporary string/tokenize_string2 pointer-array relation is lost.',
]


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
    run = ROOT/'runs/three_memory_codec_20261029_v40'
    plan, result, end = [read(run/n) for n in ('plan.json', 'result.json', 'supervisor/final_status.json')]
    terminal = read(run/'code_snapshot/root_terminal_native_observation.json')
    live = read(run/'code_snapshot/root_actual_training_observation.json')
    assert terminal['Native'] == [] and terminal['AllFourOwnedPIDsAbsent']
    assert end['status'] == 'COMPLETE' and end['exit_code'] == 0 and end['exit']['all_owned_processes_gone']
    assert end['launches'] == 1 and end['actual']['identity_verified']
    assert live['ActualTraining'] and live['Progress']['updates'] > 0
    assert any(x['ProcessId'] == 32204 and x['ParentProcessId'] == 20960 and x['CommandLine'] for x in live['Native'])
    assert support.sha(run/'plan.json') == end['plan_sha256']
    assert support.sha(plan['source']) == plan['source_sha'] and support.core_sha(ROOT) == support.CORE_SHA
    assert all(support.sha(path) == digest for path, digest in plan['files'].items())

    parent = torch.load(plan['source'], weights_only=True, map_location='cpu')
    initial_path = run/'import/checkpoints/state_0000_reference.pt'
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
    del parent
    trained = {n for group in initial['optimizer_parameter_names'] for n in group}
    checkpoints = []
    # V40 intentionally has NO midpoint and at most ONE updated self-contained fullstate.
    updated_paths = list((run/'model/checkpoints').glob('state_*.pt'))
    assert updated_paths == [run/'model/checkpoints/state_8192.pt']
    assert not (run/'emergency').exists()
    for local, path in ((0, initial_path), (8192, updated_paths[0])):
        state = initial if local == 0 else torch.load(path, weights_only=True, map_location='cpu')
        marker = read(path.with_suffix('.json'))
        digest, algorithm = support.sha(path), support.state_sha(state)
        assert digest == marker['sha256'] and algorithm == marker['algorithm_sha256']
        assert state['schema'] == 'fdt_three_memory_language_codec_v40_fullstate'
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
        if local:
            factor = .1+.9*(1+math.cos(math.pi*(local-1)/8192))/2
            assert all(g['lr'] == g['base_lr']*factor for g in state['optimizer']['param_groups'])
        for label, digest_value in state['evaluations'].items():
            path_value = run/(label if label.endswith(('.pt', '.json')) else label+'.json')
            assert support.sha(path_value) == digest_value
        checkpoints.append({'attempt': local, 'updates': state['updates'], 'AMP_skips': skips,
            'bytes': path.stat().st_size, 'sha256': digest, 'algorithm_sha256': algorithm,
            'stage': state['worker_stage'], 'scaler': state['scaler'], 'standalone': local != 0})
    assert checkpoints[-1]['sha256'] == result['latest']['sha256']
    assert checkpoints[-1]['algorithm_sha256'] == result['latest']['algorithm_sha256']
    assert recursive_count(state, 'attempts') == result['global_attempts'] == 137188
    assert recursive_count(state, 'updates') == result['global_updates'] == 137138
    expected_orders = []
    for seed in (39001, 39002):
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
            meaning_grade='exact_source_span' if row['exact'] else 'whole_proposition_punctuation_changed'
            if index in NONEXACT_WHOLE else 'partial_proposition' if index in PARTIAL else 'not_faithful')
    assert train_summary['exact'] == 31 and held_summary['exact'] == 0
    assert sum(row['whole_source_span_faithful'] for row in train) == 32
    assert len(PARTIAL) == 11
    tensor, focus_metrics = logits_review(run/'focused_final_logits.pt', focused_targets, result['focused_final'])
    held_tensor = torch.load(run/'held_final_logits.pt', weights_only=True, map_location='cpu')
    alignment = {'focused64': prefix_alignment(tensor, raw_train, range(64)),
                 'held16': prefix_alignment(held_tensor, raw_held, plan['generation_indices'])}
    memory = read(run/'root_three_memory_readouts_review.json')
    assert memory['status'] == 'V40_SAVED_THREE_MEMORY_READ_PATHS_VERIFIED'
    assert memory['plan_sha256'] == end['plan_sha256']
    calls = {'encode': 8192+8+16+32+4+6, 'denoise': 0,
             'decoder': 16384+24+96+6+stop_count(raw_train)+stop_count(raw_held)}
    assert calls == result['actual_calls'] and sum(calls.values()) == 26613
    assert result['cumulative_calls'] == {key: source_calls[key]+calls[key] for key in calls}
    old = read(ROOT/'runs/three_memory_codec_20261029_v39/root_reconstruction_similarity.json')
    gains = [i for i, row in enumerate(train) if row['exact'] and not old['focused64'][i]['exact']]
    losses = [i for i, row in enumerate(train) if not row['exact'] and old['focused64'][i]['exact']]
    assert not torch.cuda.is_initialized()
    out = {'status': 'V40_COMPLETE_SEVENTH_EIGHTH_BROAD_EPOCHS_ROOT_REVIEWED',
        'plan_sha256': end['plan_sha256'], 'fullstates': checkpoints,
        'initial_checkpoint_parent_required': True, 'updated_checkpoint_self_contained': True,
        'one_updated_fullstate_no_midpoint_verified': True,
        'source_all91_model_Adam_scaler_ALLstoredRNG_exact': True,
        'source_component_sha256': imported_digest, 'all_model_Adam_finite': True,
        'frozen_original_reasoning_language_prefix_basis_unchanged': True,
        'new_finite_cosine_without_Adam_reset_verified': True,
        'new_attempts': 8192, 'new_updates': 8189, 'AMP_skips': checkpoints[-1]['AMP_skips'],
        'global_attempts': result['global_attempts'], 'global_updates': result['global_updates'],
        'actual_calls': calls, 'total_new_calls': 26613, 'cumulative_calls': result['cumulative_calls'],
        'external_closed_assessment_calls': external, 'exposure': result['exposure'],
        'online_loss_blocks': online, 'focused64_reconstruction': train_summary,
        'held16_reconstruction': held_summary, 'focused64_whole_proposition_faithful': 32,
        'focused64_nonexact_whole_proposition_faithful': 1, 'focused64_partial_propositions': 11,
        'held_whole_faithful': 0, 'held_confirmed_partial_propositions': 0, 'ROOT_read_all80': True,
        'focused64_exact_gains_vs_V39': gains, 'focused64_exact_losses_vs_V39': losses,
        'same_first_teacher_BOS_error_position_all80': True,
        'saved_teacher_BOS_prefix_alignment': alignment,
        'saved_logit_metrics': {'focused_final': focus_metrics,
                               'held_final': memory['held256_saved_logit_metrics']},
        'memory_readout_review_sha256': support.sha(run/'root_three_memory_readouts_review.json'),
        'held16_normal_NLL': memory['normal_same16_NLL'],
        'held16_memory_ablations': memory['each_memory_disabled_same16'],
        'all_owned_proxy_gone_before_analysis': True, 'ROOT_worker_live_observed': True,
        'additional_model_construction_forward': 0, 'CUDA_replay_performed': False,
        'denoise_calls': 0, 'literal_copy_bytes': 0, 'E2_final_accessed': False,
        'conclusion': 'Observed held teacher byte accuracy78.3083->79.0796%; focused64exact28->31 '
                      'with32whole propositions including one punctuation-only change; held whole fidelity remains0/16. '
                      'Recovered dates1948/1899, automaton and question mark coexist with exact regressions, '
                      'and the prior held no-explosion partial core regressed. Saved similarity is not meaning accuracy. '
                      'No global failure, perfect codec, complete semantics or independent diffusion claim.'}
    write(run/'root_result.json', out)
    write(run/'root_reconstruction_similarity.json', {'focused64': train, 'held16': held,
        'summary': {'focused64': train_summary, 'held16': held_summary}})
    lines = ['# V40 ROOT reads all80 SAME-input source/reconstructions', '', out['conclusion']]
    for label, rows in (('TRAIN_REFERENCE', train), ('HELD', held)):
        for row in rows:
            lines += ['', f"## {label} {row['index']}", f"Source: {row['source']}",
                      f"Reconstruction: {row['reconstruction']}", f"ROOT: {row['ROOT_reading']}"]
    with (run/'root_all80_reading.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps({key: out[key] for key in ('status', 'fullstates', 'focused64_reconstruction',
        'held16_reconstruction', 'focused64_exact_gains_vs_V39', 'focused64_exact_losses_vs_V39',
        'online_loss_blocks', 'total_new_calls')}))


if __name__ == '__main__':
    main()
