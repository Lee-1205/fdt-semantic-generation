"""Prospective actual worker schema/pins; CPU only, no execution authorization."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import run_three_memory_codec_20261029_v33 as worker
from three_memory_codec_engine_v33 import CONTRACT, OWNERSHIP
from owt_resource_budget_v29 import RESOURCE_POLICY
from owt_fuzzy_engine_v17_base import AMP_STOP_POLICY
import text_generation_support_v1 as support


def main():
    artifact = ROOT/'research/artifacts/three_memory_codec_20261029_v33_worker_cpu'
    revision = sys.argv[1] if len(sys.argv) == 2 else 'r1'
    if revision not in ('r1', 'r2', 'r3'): raise ValueError('fixed prospective revision')
    artifact.mkdir(exist_ok=revision != 'r1')
    plan_path = artifact/('plan_unsealed.json' if revision == 'r1' else 'plan_unsealed_'+revision+'.json')
    if plan_path.exists(): raise FileExistsError('prospective plan record exists')
    output = ROOT/'runs/three_memory_codec_20261029_v33'
    if output.exists(): raise FileExistsError('new output must not exist yet')
    data = ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final'
    focused = ROOT/'research/artifacts/owt_focused_reconstruction_20261029_v31_cpu'
    cpu = ROOT/'research/artifacts/three_memory_codec_20261029_v33_cpu'
    source = ROOT/'runs/owt_fuzzy_reconstruction_20261029_v31/model/checkpoints/state_2048.pt'
    for current in ('research/three_memory_codec_v33.py', 'scripts/three_memory_codec_engine_v33.py'):
        if support.sha(ROOT/current) != support.sha(cpu/'code_snapshot'/Path(current).name):
            raise ValueError('tested candidate/driver changed')
    code = ['scripts/run_three_memory_codec_20261029_v33.py', 'scripts/supervise_three_memory_codec_20261029_v33.py',
        'scripts/three_memory_codec_engine_v33.py', 'scripts/prepare_three_memory_worker_v33_cpu.py',
        'research/three_memory_codec_v33.py', 'research/three_memory_contract_v32.py',
        'research/owt_extraction_tail_v29.py', 'research/owt_prefix_hold_v28.py', 'research/owt_prefix_reconstruction_v19.py',
        'research/owt_prefix_curriculum_v27.py', 'research/owt_fuzzy_reconstruction_v12.py',
        'research/text_context_decoder_v2.py', 'research/text_diffusion_v1.py',
        'scripts/owt_fuzzy_engine_v17_base.py', 'scripts/owt_fuzzy_engine_v12.py', 'scripts/owt_resource_budget_v29.py',
        'scripts/text_language_runtime_support_v6.py', 'scripts/text_generation_support_v1.py',
        'scripts/learning_io_v2.py', 'scripts/run_owt_fuzzy_reconstruction_20261028_v12.py',
        'scripts/supervise_querycount_partial9_20261027_v3.py']
    paths = [ROOT/name for name in code]+list((ROOT/'src/fdt_local').glob('*.py'))
    source_plan = ROOT/'runs/owt_fuzzy_reconstruction_20261029_v31/plan.json'
    paths += [source, source_plan, cpu/'manifest.json']
    paths += [data/name for name in ('manifest.json', 'features.json', 'targets.json', 'documents.json')]
    paths += [focused/name for name in ('manifest.json', 'features.json', 'targets.json', 'orders.json')]
    plan = {'schema': worker.SCHEMA, 'authorized': False, 'root': str(ROOT), 'output': str(output),
        'artifact': str(data), 'focused_artifact': str(focused), 'cpu_artifact': str(cpu),
        'source': str(source), 'source_sha': worker.SOURCE_SHA, 'source_plan_sha': worker.SOURCE_PLAN,
        'source_algorithm': worker.SOURCE_ALGORITHM, 'source_global_attempts': 102372, 'source_global_updates': 102335,
        'external_closed_assessment_calls': json.loads(source_plan.read_text(encoding='utf-8'))['external_closed_assessment_calls'],
        'limits': worker.LIMITS, 'cost_contract': worker.COST, 'codec_contract': CONTRACT, 'ownership_policy': OWNERSHIP,
        'start_storage_contract': worker.START_STORAGE,
        'resource_policy': RESOURCE_POLICY, 'amp_stop_policy': AMP_STOP_POLICY,
        'epochs': 256, 'order_seed_range': [8331, 8586], 'initialization_seed': 833,
        'training_exposure': {'unique_training_rows': 64, 'selected_rows': 16384, 'target_UTF8_bytes': 1101824, 'finite_epochs': 256},
        'generation_indices': [4*i for i in range(16)], 'automatic_source_copy': False,
        'reasoning_diffusion_training': False, 'E2_final_accessed': False,
        'allocation_start': None, 'deadline': None, 'cutoff': None,
        'purpose': 'Adapt preserved language codec to newly wired three-memory meaning Commit; SAME source reconstruction capacity. This repeated TRAIN64 is not generalization or denoising.',
        'files': {str(p.resolve()): support.sha(p) for p in paths}}
    path = plan_path
    with path.open('x', encoding='utf-8') as f: json.dump(plan, f, ensure_ascii=False, indent=2)
    worker.validate(path, output, worker.__file__)
    print(json.dumps({'status': 'ACTUAL_SCHEMA_PROSPECTIVE_CPU_READY_NOT_SEALED', 'plan': str(path),
        'output_exists': output.exists(), 'new_model_calls': 0, 'cost_max': worker.COST['total_max']}))


if __name__ == '__main__': main()
