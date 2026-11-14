"""ONE bounded three-memory input-meaning-input codec adaptation, no denoising."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import ctypes
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.three_memory_codec_v33 import ThreeMemoryCodec
from research.text_diffusion_v1 import pack_answers, decode_bytes, PAD
from three_memory_codec_engine_v33 import CodecEngine, CONTRACT, OWNERSHIP
from owt_fuzzy_engine_v17_base import AMP_STOP_POLICY
from owt_resource_budget_v29 import RecoveryBudget, RESOURCE_POLICY
from run_owt_fuzzy_reconstruction_20261028_v12 import finite_orders
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA = 'fdt_three_memory_language_codec_v33'
START_STORAGE = {'protected_free_bytes': 20*1024**3, 'max_owned_output_bytes': 6*1024**3,
    'temporary_checkpoint_bytes': 1024**3, 'additional_headroom_bytes': 8*1024**3,
    'minimum_start_bytes': 35*1024**3, 'previous_unlaunched_prospective_start_bytes': 40*1024**3,
    'historical_closed_run_floors_changed': False}
LIMITS = {'attempts': 2048, 'batch': 8, 'wall': 900, 'reserve': 120, 'max_calls': 12000,
    'max_output_bytes': 6*1024**3, 'start_free_bytes': 35*1024**3, 'min_gpu_free_mib': 12288}
COST = {'training': 2048*3, 'focused_initial_final_teacher': 2*8*4,
    'focused_initial_final_BOS_max': 2*16*130, 'held_final_teacher': 32*4,
    'held_final_BOS_max': 4*130, 'held16_three_memory_ablations': 3*2*2, 'total_max': 11028}
SOURCE_SHA = '1d02d0d52c44683a582e70273e3e951e84384ecb406700470af620e21e8c252f'
SOURCE_PLAN = '6e0be6c55d882a3ab2a00fa764b03a11333bda10d4cbfa50c694cf4b137f2375'
SOURCE_ALGORITHM = '66316e12eeeb9ae3d6c3515f5983deca0363f9d54f2166c1bc6210a0c604bf19'
FOCUSED_MANIFEST = 'bd887866d4dc46357931a84934e5fb6a1fee1e77b11979fd4a3910490663e7f8'
CORPUS_MANIFEST = '936351c98e8e8f5e614814fff84f2a7903ee07bed3a2cc46a18b6f820c0290f8'
CPU_MANIFEST = '6928250f7eca1aa94b9a6ecf76d2681a73c15cb5b6b51f1cb6a65ec1adf51111'


def validate(path, output, worker=None):
    p = io.load_json(path)
    if p.get('schema') != SCHEMA or type(p.get('authorized')) is not bool or p.get('limits') != LIMITS:
        raise ValueError('new three-memory actual schema/caps')
    if p.get('start_storage_contract') != START_STORAGE or LIMITS['start_free_bytes'] != sum(START_STORAGE[k] for k in (
        'protected_free_bytes', 'max_owned_output_bytes', 'temporary_checkpoint_bytes', 'additional_headroom_bytes')):
        raise ValueError('explicit NEW prelaunch storage calculation')
    if Path(p['root']).resolve() != ROOT or Path(p['output']).resolve() != Path(output).resolve():
        raise ValueError('root/output identity')
    if p.get('codec_contract') != CONTRACT or p.get('ownership_policy') != OWNERSHIP or p.get('resource_policy') != RESOURCE_POLICY or p.get('amp_stop_policy') != AMP_STOP_POLICY:
        raise ValueError('actual codec/ownership/resource policy')
    if p.get('epochs') != 256 or p.get('order_seed_range') != [8331, 8586] or p.get('cost_contract') != COST:
        raise ValueError('new finite order/cost contract')
    if p.get('reasoning_diffusion_training') is not False or p.get('E2_final_accessed') is not False or p.get('automatic_source_copy') is not False:
        raise ValueError('language codec only/no whole-copy/holdout')
    if (p.get('source_sha'), p.get('source_plan_sha'), p.get('source_algorithm')) != (SOURCE_SHA, SOURCE_PLAN, SOURCE_ALGORITHM):
        raise ValueError('V31 source identity')
    if (p.get('source_global_attempts'), p.get('source_global_updates')) != (102372, 102335):
        raise ValueError('source global counters')
    historical_plan = ROOT/'runs/owt_fuzzy_reconstruction_20261029_v31/plan.json'
    if support.sha(historical_plan) != SOURCE_PLAN or p.get('external_closed_assessment_calls') != io.load_json(historical_plan)['external_closed_assessment_calls']:
        raise ValueError('separate external historical ledger')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start'] != 900 or p['cutoff'] != p['deadline']-120):
        raise ValueError('fresh fixed clock/reserve')
    for name, digest in p['files'].items():
        if support.sha(name) != digest: raise ValueError('pin changed '+name)
    if worker and str(Path(worker).resolve()) not in p['files']: raise ValueError('actual worker pin')
    if support.core_sha(ROOT) != support.CORE_SHA or p['files'].get(str(Path(p['source']).resolve())) != SOURCE_SHA:
        raise ValueError('source/core pin')
    q = Path(p['focused_artifact']); a = Path(p['artifact']); cpu = Path(p['cpu_artifact'])
    if (support.sha(q/'manifest.json'), support.sha(a/'manifest.json'), support.sha(cpu/'manifest.json')) != (FOCUSED_MANIFEST, CORPUS_MANIFEST, CPU_MANIFEST):
        raise ValueError('immutable data/candidate evidence')
    manifest = io.load_json(q/'manifest.json')
    for name, digest in manifest['file_SHA256'].items():
        if support.sha(q/name) != digest: raise ValueError('focused data byte hash')
    broadf = io.load_json(a/'features.json'); broady = io.load_json(a/'targets.json'); docs = io.load_json(a/'documents.json')
    f = io.load_json(q/'features.json'); y = io.load_json(q/'targets.json')
    selected = sorted(docs['train'][4096:], key=lambda d: (d['normalized_sha256'], d['id']))[:64]
    indices = []
    for doc in selected:
        rows = [i for i in range(16384, 32768) if broady['train'][i]['document_id'] == doc['id']]
        if len(rows) != 4: raise ValueError('exact source four-span selection')
        indices.append(min(rows, key=lambda i: (len(broadf['train'][i]['prompt'].encode()), broadf['train'][i]['id'])))
    if indices != manifest['selected_train_indices'] or f != [broadf['train'][i] for i in indices] or y != [broady['train'][i] for i in indices]:
        raise ValueError('independent input selection')
    if len(f) != 64 or len(broadf['dev']) != 256 or len({r['document_id'] for r in y}) != 64:
        raise ValueError('exact panel dimensions')
    if {d['normalized_sha256'] for d in selected} & {d['normalized_sha256'] for d in docs['dev']}:
        raise ValueError('train/held document collision')
    for features, targets in ((f, y), (broadf['dev'], broady['dev'])):
        for feature, target in zip(features, targets):
            if set(feature) != {'id', 'prompt'} or feature['id'] != target['id'] or feature['prompt'] != target['target'] or target['target_ids'] != list(feature['prompt'].encode())+[256]:
                raise ValueError('SAME input and offline target separation')
    orders = finite_orders(64, range(8331, 8587))
    exposure = {'unique_training_rows': 64, 'selected_rows': 16384, 'target_UTF8_bytes': 1101824, 'finite_epochs': 256}
    if p.get('training_exposure') != exposure or sum(len(r['prompt'].encode()) for r in f) != 4304:
        raise ValueError('exact finite byte exposure')
    if p.get('generation_indices') != [4*i for i in range(16)] or p.get('initialization_seed') != 833:
        raise ValueError('fixed observed panel/initialization')
    return p, {'train': f, 'dev': broadf['dev']}, {'train': y, 'dev': broady['dev']}, orders


def load_import(p, features, targets, orders, budget, plan_sha, device):
    state = torch.load(p['source'], map_location='cpu', weights_only=True)
    if support.state_sha(state) != SOURCE_ALGORITHM or (state['attempts'], state['updates'], state['cursor']) != (2048, 2048, 16384):
        raise ValueError('actual complete V31 source algorithm')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(833); model = ThreeMemoryCodec(Config(**state['core']))
    engine = CodecEngine(model, features['train'], targets['train'], orders, budget, plan_sha, device,
        source_parameter_groups=state['optimizer_parameter_names'], data_revision=CORPUS_MANIFEST)
    report = engine.import_parent(state, SOURCE_PLAN, SOURCE_SHA)
    if sum(x.numel() for x in model.parameters()) != 111732306 or len(engine.optimizer.state) != 81:
        raise ValueError('actual new model/old81 optimizer inventory')
    return engine, report


def amp_for(device):
    return torch.autocast('cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()


def teacher_assessment(engine, features, targets, budget, device, disabled=(), modes=('normal', 'swap', 'zero'), trace_indices=()):
    engine.model.eval(); sums = {m: [0., 0, 0, 0] for m in modes}
    stored = {m: [] for m in modes}; ids = []; meanings = []; traces = []
    with torch.no_grad(), amp_for(device):
        for start in range(0, len(features), 8):
            rows = features[start:start+8]; ys = targets[start:start+8]
            selected = [i for i in range(len(rows)) if start+i in trace_indices]
            captured = {}
            def capture_query(module, args, output):
                captured['query'] = output.detach().float().cpu()
                captured['base'] = args[0].detach().float().cpu()
            handle = engine.model.memory_codec.exact_query.register_forward_hook(capture_query) if selected else None
            try: episode = engine.encode_features(rows, budget.charge, disabled)
            finally:
                if handle: handle.remove()
            for i in selected:
                bundle = episode.memories; d = episode.diagnostics; bank = bundle.contextual
                traces.append({'id': rows[i]['id'], 'source': bundle.exact.sources[i].raw.decode('utf-8'),
                    'source_sha256': bundle.exact.sources[i].sha256, 'source_version': bundle.exact.sources[i].version,
                    'model_epoch': bundle.model_epoch, 'spans': bundle.exact.spans[i],
                    'exact_keys': bundle.exact.keys[i].detach().float().cpu(), 'exact_query': captured['query'][i],
                    'pre_exact_semantic': captured['base'][i], 'exact_valid': bundle.exact.valid[i].cpu(),
                    'exact_null_logit': engine.model.memory_codec.exact_null.detach().float().cpu(),
                    'candidate_ids': d['candidate_ids'][i].cpu(), 'candidate_weights': d['candidate_weights'][i].cpu(),
                    'null_weight': d['null_weight'][i].cpu(), 'token_mu': d['token_memberships'][i].cpu(),
                    'token_values': d['token_values'][i].cpu(), 'decay': bank.decay.cpu(),
                    'long_N': bank.long_N[i].cpu(), 'long_D': bank.long_D[i].cpu(),
                    'recent_N': bank.recent_N[i].cpu(), 'recent_D': bank.recent_D[i].cpu(),
                    'query_mu': d['memberships'][i].cpu(), 'rho': d['retrieval_weights'][i].cpu(),
                    'gamma': d['confidence'][i].cpu(), 'long_gate': d['long_gate'][i].cpu(),
                    'exact_gate': d['exact_gate'][i].cpu(), 'copy_bytes': 0, 'denoise_calls': 0})
            target = pack_answers([r['target'] for r in ys], 129, device); active = target != PAD
            ids.append(target.cpu()); meanings.append(episode.commit.reader_tensor().float().cpu())
            for mode in modes:
                budget.charge('decoder'); logits = engine.model.teacher_logits(episode.commit, target, mode)
                nll = logits.logsumexp(-1)-logits.gather(-1, target[..., None]).squeeze(-1)
                accumulator = sums[mode]
                accumulator[0] += float(nll[active].double().sum()); accumulator[1] += int((logits.argmax(-1)[active] == target[active]).sum())
                accumulator[2] += int(active.sum()); accumulator[3] += int(((target == 256) & (logits.argmax(-1) == 256)).sum())
                stored[mode].append(logits.half().cpu())
    scores = {m: {'NLL': nll/symbols, 'byte_accuracy': correct/symbols, 'correct': correct, 'symbols': symbols,
        'sentence_EOS_correct': eos, 'sentence_EOS_total': len(features)} for m, (nll, correct, symbols, eos) in sums.items()}
    tensors = {m: torch.cat(rows) for m, rows in stored.items()}
    tensors.update(target_ids=torch.cat(ids), semantic_commit=torch.cat(meanings), memory_trace=traces)
    return scores, tensors


def reconstruct_inputs(engine, features, budget, device):
    # Feature-only API. Desired output, gold prefix/length and Exact bytes never
    # become realizer arguments. BOS + model's own produced prefix only.
    engine.model.eval(); result = []
    with torch.no_grad(), amp_for(device):
        for start in range(0, len(features), 4):
            rows = features[start:start+4]; episode = engine.encode_features(rows, budget.charge)
            output = engine.model.reconstruct(episode.commit, budget.charge).cpu()
            for i, row in enumerate(rows):
                generated = decode_bytes(output[i].tolist())
                result.append({'id': row['id'], 'source': row['prompt'], 'generated': generated,
                    'reconstruction_exact': generated['text'] == row['prompt'], 'copy_bytes': 0, 'denoise_calls': 0})
    return result


def run(path, output):
    p, f, y, orders = validate(path, output, __file__)
    if not p['authorized'] or p['deadline'] is None: raise PermissionError('new plan not sealed')
    if (output/'progress.json').exists(): raise FileExistsError('ONE invocation only')
    configure_determinism(); torch.set_num_threads(4)
    budget = RecoveryBudget(p); engine = None; latest = None; phase = 'importing'; began = time.time(); guard = False
    def progress():
        if engine: engine.worker_stage = phase
        io.atomic_json(output/'progress.json', {'phase': phase, 'attempts': engine.attempts if engine else 0,
            'updates': engine.updates if engine else 0, 'actual_calls': dict(budget.calls), 'allocation_calls': dict(budget.allocation_calls),
            'native_pid': os.getpid(), 'parent_pid': os.getppid(), 'time_unix': time.time(), 'elapsed': time.time()-began,
            'latest': latest, 'resource_scans': budget.resource_scans})
    def commit(kind):
        nonlocal latest
        if kind == 'emergency':
            if time.time() >= p['deadline']: raise TimeoutError('owner deadline')
            budget.check_resources(force=True, upcoming_bytes=RESOURCE_POLICY['checkpoint_write_reserve_bytes'])
        else:
            budget.check(force=True, upcoming_bytes=RESOURCE_POLICY['checkpoint_write_reserve_bytes']); engine.verify_frozen()
        engine.worker_stage = phase; directory = output/kind; directory.mkdir(exist_ok=True)
        latest = {**support.save_full(directory, engine.export()), 'relative_directory': kind}; io.atomic_json(output/'latest.json', latest)
    def assess(label, features, targets, raw=False, disabled=(), modes=('normal', 'swap', 'zero')):
        nonlocal phase
        phase = label+'_teacher'; progress()
        trace_indices = p['generation_indices'] if label == 'held_final' else ()
        scores, tensors = teacher_assessment(engine, features, targets, budget, 'cuda', disabled, modes, trace_indices)
        budget.check(force=True, upcoming_bytes=128*1024**2)
        torch.save(tensors, output/(label+'_logits.pt')); del tensors
        io.atomic_json(output/(label+'_metrics.json'), scores)
        for suffix in ('logits.pt', 'metrics.json'):
            engine.evaluations[label+'_'+suffix] = support.sha(output/(label+'_'+suffix))
        if raw:
            phase = label+'_same_input_BOS'; progress()
            io.atomic_json(output/(label+'_reconstruction.json'), reconstruct_inputs(engine, features, budget, 'cuda'))
            engine.evaluations[label+'_reconstruction'] = support.sha(output/(label+'_reconstruction.json'))
        return scores
    budget.tick = progress; progress()
    try:
        budget.check(force=True)
        if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); guard = True
        engine, report = load_import(p, f, y, orders, budget, support.sha(path), 'cuda')
        io.atomic_json(output/'import_review.json', report)
        phase = 'initial_commit'; progress(); commit('import')
        initial = assess('focused_initial', f['train'], y['train'], raw=True)
        phase = 'codec_training'; progress()
        while engine.attempts < LIMITS['attempts']:
            if time.time()+90 >= p['cutoff']: raise TimeoutError('final assessment reserve')
            engine.step()
            if engine.attempts == 1024: commit('model')
            progress()
        final = assess('focused_final', f['train'], y['train'], raw=True)
        held = assess('held_final', f['dev'], y['dev'])
        selected_f = [f['dev'][i] for i in p['generation_indices']]
        selected_y = [y['dev'][i] for i in p['generation_indices']]
        phase = 'held_final_same_input_BOS'; progress()
        io.atomic_json(output/'held_final_reconstruction.json', reconstruct_inputs(engine, selected_f, budget, 'cuda'))
        engine.evaluations['held_final_reconstruction'] = support.sha(output/'held_final_reconstruction.json')
        ablations = {}
        for memory in ('persistent', 'contextual', 'exact'):
            ablations[memory] = assess('held16_without_'+memory, selected_f, selected_y, disabled=(memory,), modes=('normal',))
        phase = 'final_commit'; progress(); commit('model')
        if support.sha(p['source']) != SOURCE_SHA or support.core_sha(ROOT) != support.CORE_SHA: raise RuntimeError('source/core changed')
        io.atomic_json(output/'result.json', {'status': 'THREE_MEMORY_CODEC_V33_COMPLETE',
            'new_attempts': engine.attempts, 'new_updates': engine.updates, 'AMP_skips': engine.skips,
            'global_attempts': 102372+engine.attempts, 'global_updates': 102335+engine.updates,
            'actual_calls': dict(budget.allocation_calls), 'cumulative_calls': dict(budget.calls),
            'external_closed_assessment_calls': p['external_closed_assessment_calls'], 'exposure': engine.exposure,
            'focused_initial': initial, 'focused_final': final, 'held_final': held, 'held16_ablations': ablations,
            'latest': latest, 'seconds': time.time()-began, 'resource_scans': budget.resource_scans,
            'reasoning_or_diffusion_training': False, 'automatic_source_copy': False, 'E2_final_accessed': False,
            'interpretation': 'New three-memory input-meaning-input function; old81 Adam retained/new10 heads fresh. Repeated64 TRAIN capacity including memorization, observed held; no certified Writer/full vNext/learned literal copy intent/general semantic completeness claim.'})
        phase = 'complete'; progress(); return 0
    except BaseException as exc:
        phase = 'stopped_or_error'
        if engine:
            engine.optimizer.zero_grad(set_to_none=True)
            try: commit('emergency')
            except BaseException as save: io.atomic_json(output/'save_error.json', {'error': repr(save)})
        io.atomic_json(output/'incident.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'new_calls': budget.allocation_calls, 'cumulative_calls': budget.calls}); progress(); return 2
    finally:
        if guard: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        p, f, y, orders = validate(args.plan, args.output, __file__); torch.set_num_threads(2)
        load_import(p, f, y, orders, RecoveryBudget(p), support.sha(args.plan), 'cpu')
        if torch.cuda.is_initialized(): raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS'); return 0
    return run(args.plan.resolve(), args.output.resolve())


if __name__ == '__main__': raise SystemExit(main())
