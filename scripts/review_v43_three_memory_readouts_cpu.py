"""Saved V43 logits/memory traces only; no model construction or forward."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import text_generation_support_v1 as support
from review_three_memory_codec_v33_result_cpu import read, trace_review, logits_review


def main():
    torch.set_num_threads(2)
    run = ROOT/'runs/three_memory_codec_20261030_v43'
    plan, result, end = [read(run/n) for n in ('plan.json', 'result.json', 'supervisor/final_status.json')]
    terminal = read(run/'code_snapshot/root_terminal_native_before_analysis.json')
    assert end['status'] == 'COMPLETE' and end['exit']['all_owned_processes_gone']
    assert terminal['native'] == [] and terminal['all_owned_proxy_gone_before_analysis']
    assert support.sha(run/'plan.json') == end['plan_sha256'] and support.core_sha(ROOT) == support.CORE_SHA
    for name in ('research/three_memory_codec_v33.py', 'scripts/three_memory_codec_engine_v43.py',
                 'scripts/run_three_memory_codec_20261030_v43.py'):
        path = ROOT/name
        assert support.sha(path) == plan['files'][str(path.resolve())]
    targets = read(Path(plan['artifact'])/'targets.json')['dev']
    tensor, metrics = logits_review(run/'held_final_logits.pt', targets, result['held_final'])
    trace = trace_review(tensor['memory_trace'])
    index = plan['generation_indices']
    gold = tensor['target_ids'][index]
    active = gold != 257
    logits = tensor['normal'][index].double()
    nll = logits.logsumexp(-1)-logits.gather(-1, gold[..., None]).squeeze(-1)
    normal = float(nll[active].mean())
    controls = {}
    for memory in ('persistent', 'contextual', 'exact'):
        _, checked = logits_review(run/f'held16_without_{memory}_logits.pt',
            [targets[i] for i in index], result['held16_ablations'][memory], modes=('normal',))
        controls[memory] = {**checked['normal'],
            'NLL_increase_when_disabled': checked['normal']['NLL_from_saved_FP16']-normal}
    assert all(v['NLL_increase_when_disabled'] > 0 for v in controls.values())
    assert not torch.cuda.is_initialized()
    out = {'status': 'V43_SAVED_THREE_MEMORY_READ_PATHS_VERIFIED',
        'plan_sha256': end['plan_sha256'], 'normal_same16_NLL': normal,
        'held256_saved_logit_metrics': metrics, 'each_memory_disabled_same16': controls,
        'full_Exact_top16_NULL_and_sequential_context_ND_reconstruction': trace,
        'all_owned_gone_before_analysis': True, 'additional_model_construction_forward': 0,
        'CUDA_initialized': False, 'denoise_calls': 0, 'literal_copy_bytes': 0,
        'scope': 'Three encoded memory READ paths affect this observed panel. Certified persistent Writer, '
                 'full layerwise prefix K_C, and Exact literal selector/transport remain unimplemented. '
                 'No claim of full original architecture or complete meaning fidelity.'}
    with (run/'root_three_memory_readouts_review.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ('status', 'normal_same16_NLL', 'each_memory_disabled_same16',
                                         'additional_model_construction_forward')}))


if __name__ == '__main__':
    main()
