"""Actual saved 7-tensor scope, stale-plan rejection, prospective CPU guard."""
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from plan_scope_contract_v78_cpu import validate_scope


def main():
    run = ROOT/'runs/update_balance_20261113_v77'
    original = io.load_json(run/'plan.json')
    plan_SHA_before = support.sha(run/'plan.json')
    env = torch.load(run/'model/runtime.pt',map_location='cpu',weights_only=True)
    active = env['active_weights']
    assert len(active) == 7 and sum(t.numel() for t in active.values()) == 2908192
    rejects = []
    def reject(label, plan):
        try: validate_scope(plan,active)
        except ValueError as error: rejects.append(dict(case=label,error=str(error)))
        else: raise AssertionError('invalid scope accepted: '+label)
    reject('actual preserved V77 plan descriptive metadata',original)
    # This projection is deliberately NOT a runnable plan or a V78 launch.
    canonical = {k:copy.deepcopy(original[k]) for k in ('max_output_bytes','start_free_bytes','protected_free_bytes','cost_detail','cost','max_calls')}
    canonical['model_scope'] = dict(tensors=len(active),parameters=sum(t.numel() for t in active.values()),names=sorted(active))
    canonical['resource_estimate'] = dict(components=copy.deepcopy(original['output_estimate_detail']),total_bytes=sum(original['output_estimate_detail'].values()))
    assert validate_scope(canonical,active)
    for label,modify in (
        ('wrong parameter ownership',lambda p:p['model_scope'].update(parameters=17677856)),
        ('bad estimate sum',lambda p:p['resource_estimate'].update(total_bytes=1)),
        ('hard cap exceeded',lambda p:p.update(max_output_bytes=1)),
        ('headroom undercounted',lambda p:p.update(start_free_bytes=p['protected_free_bytes']+p['max_output_bytes'])),
        ('call total contradicted',lambda p:p.update(cost=p['cost']+1)),
        ('inherited estimate returned',lambda p:p.update(output_estimate_including_old_STOP=5656937760))):
        candidate=copy.deepcopy(canonical);modify(candidate);reject(label,candidate)
    assert len(rejects)==7 and support.sha(run/'plan.json')==plan_SHA_before and not torch.cuda.is_initialized()
    proof=dict(status='ROOT_V78_SCOPE_GUARD_CPU_PASS',canonical_scope_only_projection=canonical,actual_V77_plan_unmodified_SHA=plan_SHA_before,
        tests_passed=8,rejected=rejects,guard_SHA=support.sha(ROOT/'scripts/plan_scope_contract_v78_cpu.py'),
        actual_V78_worker_or_plan_or_GPU=False,extra_neural_forwards=0,optimizer_steps=0,CUDA_initialized=False,
        limitation='Future worker must call guard with actual optimizer-owned tensors before seal and before Popen; this CPU module does not retrofit the immutable V77 plan.')
    io.atomic_json(run/'root_plan_metadata_and_future_guard.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='canonical_scope_only_projection'}))


if __name__=='__main__':main()
