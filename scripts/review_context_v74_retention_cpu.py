"""Inspect existing TRAIN teacher packets and output confusions; no NN calls."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(4)
    run=ROOT/'runs/context_verbalizer_20261112_v74'
    assert io.load_json(run/'ROOT_terminal_native.json')['all4owned_and_proxy_gone']
    saved=io.load_json(run/'root_saved_review.json')
    assert saved['status']=='ROOT_V74_SAVED_REVIEW_PASS'
    plan=io.load_json(run/'plan.json');data=Path(plan['data'])
    targets=io.load_json(data/'train_targets.json');counts=collections.Counter();spans=collections.defaultdict(collections.Counter)
    for i,target in enumerate(targets):
        t=torch.load(run/'cache'/f'{i:04d}.targets.pt',map_location='cpu',weights_only=True)
        y=t['labels'];critical=t['critical_mask'];correct=t['teacher_correct'];valid=y<256
        counts['body_bytes']+=int(valid.sum())
        counts['critical_bytes']+=int(critical.sum())
        counts['initial_correct_critical_excluded_from_KL']+=int((valid&critical&correct).sum())
        counts['initial_wrong_critical']+=int((critical&~correct).sum())
        counts['actual_noncritical_KL_positions']+=int((valid&~critical&correct).sum())
        for span in target['critical_spans']:
            all_correct=bool(correct.flatten()[span['start']:span['end']].all())
            spans[span['kind']]['spans']+=1
            spans[span['kind']]['initial_all_bytes_correct']+=int(all_correct)
    assert counts['critical_bytes']==34194
    assert counts['actual_noncritical_KL_positions']==saved['anchor_positions']
    assert counts['initial_correct_critical_excluded_from_KL']+counts['initial_wrong_critical']==counts['critical_bytes']
    report=dict(status='ROOT_V74_TRAIN_RETENTION_CODE_REVIEW_COMPLETE',counts=dict(counts),spans=dict(spans),
        code=dict(file='research/frozen_verbalizer_objective_v72.py',SHA=support.sha(ROOT/'research/frozen_verbalizer_objective_v72.py'),
            KL_eligibility='body & initially_correct & NOT critical',critical_CE_still_active=True,
            inference_changed=False,sealed_V74_code_edited=False),
        interpretation='Correct critical TRAIN bytes lack the separate KL anchor, but already receive ordinary and .25 critical CE. This is a prospective retention-loss choice, not a demonstrated bug or single cause of held long regressions. TRAIN-only protection cannot guarantee unseen-context retention.',
        source_reference='Existing teacher probabilities/flags, not newly recomputed teacher logits; no held output or labels enter TRAIN.',
        model_calls=0,optimizer_steps=0,CUDA_initialized=torch.cuda.is_initialized())
    assert not report['CUDA_initialized']
    io.atomic_json(run/'root_retention_code_review.json',report)
    print(json.dumps(report))
if __name__=='__main__':main()
