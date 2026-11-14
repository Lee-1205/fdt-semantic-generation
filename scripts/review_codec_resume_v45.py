"""Review the saved V45 outputs on CPU after the owned worker has exited."""
import sys
from pathlib import Path

import codec_resume_v45_contract as c
import learning_io_v2 as io

sys.path[:0] = [str(c.ROOT), str(c.ROOT/'src')]
import torch
import text_generation_support_v1 as support
from review_three_memory_codec_v33_result_cpu import logits_review, raw_rows
from supervise_querycount_partial9_20261027_v3 import run_native


def owned_workers():
    query = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" "
             "| Where-Object {$_.CommandLine -match '(run|supervise)_three_memory_codec_20261111_v45.py'} "
             "| Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress")
    return run_native(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],
                      check=True, capture_output=True, text=True, timeout=10).stdout.strip()


def validate_state(plan, result):
    marker = result['latest']
    path = c.RUN/marker['relative_directory']/'checkpoints'/marker['file']
    if c.sha256(path) != marker['sha256'] or c.sha256(c.SOURCE) != c.SOURCE_SHA:
        raise ValueError('checkpoint file SHA mismatch')
    state = torch.load(path, weights_only=True, map_location='cpu')
    source = torch.load(c.SOURCE, weights_only=True, map_location='cpu')
    assert support.state_sha(state) == marker['algorithm_sha256']
    assert support.state_sha(source) == c.SOURCE_ALGORITHM
    assert state['plan_sha256'] == c.sha256(c.RUN/'plan.json')
    assert state['cursor'] == 8*state['attempts'] == c.FRESH+c.REPLAY
    assert state['attempts'] == result['new_attempts'] == c.LIMITS['attempts']
    assert state['updates'] == result['new_updates'] == state['attempts']-state['skips']
    assert not state['gradients'] and state['frozen_sha'] == source['frozen_sha']
    assert state['optimizer_parameter_names'] == source['optimizer_parameter_names']
    trainable = {n for group in state['optimizer_parameter_names'] for n in group}
    assert all(torch.equal(v, source['model'][name]) for name, v in state['model'].items()
               if name not in trainable)
    original_adam, final_adam = source['optimizer']['state'], state['optimizer']['state']
    assert len(original_adam) == len(final_adam) == 91 and original_adam.keys() == final_adam.keys()
    assert all(int(v['step']) == int(original_adam[i]['step'])+state['updates']
               for i,v in final_adam.items())
    assert [g['lr'] for g in source['optimizer']['param_groups']] == [
        g['lr'] for g in state['optimizer']['param_groups']]
    assert state['orders'] == c.read(c.ARTIFACT/'orders.json') == [r['rows'] for r in state['history']]
    assert state['allocation_calls'] == result['actual_calls'] and result['actual_calls']['denoise'] == 0
    assert result['minimum_C_free_bytes'] >= c.PROTECTED
    assert state['exposure']['target_UTF8_bytes'] == plan['training_exposure']['target_UTF8_bytes']
    for file, digest in state['evaluations'].items():
        name = file if file.endswith(('.pt', '.json')) else file+'.json'
        assert c.sha256(c.RUN/name) == digest
    return dict(path=str(path), bytes=path.stat().st_size, sha256=marker['sha256'],
                algorithm_sha256=marker['algorithm_sha256'], attempts=state['attempts'],
                updates=state['updates'], AMP_skips=state['skips'], full_state=True,
                preserved_adam_states=91, frozen_state_verified=True)


def main():
    if owned_workers():
        raise RuntimeError('wait for all V45 owner and worker processes to exit')
    torch.set_num_threads(2)
    plan, result, owner = [c.read(c.RUN/name) for name in
        ('plan.json','result.json','supervisor/final_status.json')]
    assert owner['status'] == 'COMPLETE' and owner['exit_code'] == 0
    assert owner['launches'] == 1 and owner['exit']['all_owned_processes_gone']
    assert result['status'] == c.STATUS and owner['plan_sha256'] == c.sha256(c.RUN/'plan.json')
    checkpoint = validate_state(plan, result)
    corpus, focus = Path(plan['artifact']), Path(plan['focused_artifact'])
    dev = c.read(corpus/'features.json')['dev']
    targets = c.read(corpus/'targets.json')['dev']
    indices = plan['generation_indices']
    evals = {
        'focused64': ('focused_final', c.read(focus/'features.json'), c.read(focus/'targets.json')),
        'held16': ('held_final', [dev[i] for i in indices], targets),
    }
    raw_panels, scores, metrics = {}, {}, {}
    for name, (label, features, labels) in evals.items():
        rows = c.read(c.RUN/(label+'_reconstruction.json'))
        raw_panels[name], scores[name] = raw_rows(rows, features)
        metric_panel = 'held256_teacher' if name == 'held16' else name
        tensors, metrics[metric_panel] = logits_review(c.RUN/(label+'_logits.pt'), labels, result[label])
        if name == 'held16':
            normal_held_NLL = float(
                (tensors['normal'][indices].double().logsumexp(-1)
                 - tensors['normal'][indices].double().gather(
                     -1, tensors['target_ids'][indices, :, None]).squeeze(-1))[
                         tensors['target_ids'][indices] != 257].mean())
        del tensors
    ablations = {}
    for memory, expected in result['held16_ablations'].items():
        _, report = logits_review(c.RUN/('held16_without_'+memory+'_logits.pt'),
                                  [targets[i] for i in indices], expected, ('normal',))
        ablations[memory] = report['normal']['NLL_from_saved_FP16']-normal_held_NLL
    previous = c.read(c.PARENT/'root_result.json')
    report = dict(status='V45_COMPLETE_CPU_REVIEWED', plan_sha256=owner['plan_sha256'],
                  checkpoint=checkpoint, reconstruction=scores, saved_logits=metrics,
                  held16_disabled_memory_NLL_increase=ablations,
                  source_held_teacher=previous['saved_logits']['held256_teacher']['normal'],
                  fresh_spans=c.FRESH, replay_spans=c.REPLAY,
                  actual_calls=result['actual_calls'], minimum_C_free_bytes=result['minimum_C_free_bytes'],
                  new_model_calls=0, all_owned_processes_gone=True)
    io.atomic_json(c.RUN/'root_reconstruction_similarity.json', raw_panels)
    io.atomic_json(c.RUN/'root_result.json', report)
    print(c.read(c.RUN/'root_result.json'))


if __name__ == '__main__':
    main()
