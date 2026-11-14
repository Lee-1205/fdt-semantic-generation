"""Saved-only answer to the user's three-memory status question; no model."""
import json,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import text_generation_support_v1 as support
from review_three_memory_codec_v33_result_cpu import read,trace_review,logits_review

def main():
    torch.set_num_threads(2);run=R/'runs/three_memory_codec_20261029_v37'
    p,result,end=[read(run/name) for name in ('plan.json','result.json','supervisor/final_status.json')]
    native=read(run/'code_snapshot/root_terminal_native_observation.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_count']==0
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.core_sha(R)==support.CORE_SHA
    for name in ('research/three_memory_codec_v33.py','scripts/three_memory_codec_engine_v37.py','scripts/run_three_memory_codec_20261029_v37.py'):
        path=R/name;assert support.sha(path)==p['files'][str(path.resolve())]
    targets=read(Path(p['artifact'])/'targets.json')['dev']
    tensor,all_metrics=logits_review(run/'held_final_logits.pt',targets,result['held_final'])
    traces=trace_review(tensor['memory_trace']);idx=p['generation_indices'];ids=tensor['target_ids'][idx];active=ids!=257
    z=tensor['normal'][idx].double();nll=z.logsumexp(-1)-z.gather(-1,ids[...,None]).squeeze(-1)
    base=float(nll[active].mean());controls={}
    for memory in ('persistent','contextual','exact'):
        _,metrics=logits_review(run/('held16_without_'+memory+'_logits.pt'),[targets[i] for i in idx],result['held16_ablations'][memory],modes=('normal',))
        controls[memory]={**metrics['normal'],'NLL_increase_when_disabled':metrics['normal']['NLL_from_saved_FP16']-base}
    assert all(v['NLL_increase_when_disabled']>0 for v in controls.values()) and not torch.cuda.is_initialized()
    out={'status':'V37_SAVED_THREE_MEMORY_READ_PATHS_VERIFIED','plan_sha256':end['plan_sha256'],
        'normal_same16_NLL':base,'held256_saved_logit_metrics':all_metrics,'each_memory_disabled_same16':controls,
        'full_Exact_top16_NULL_and_sequential_context_ND_reconstruction':traces,
        'all_owned_gone_before_analysis':True,'additional_model_construction_forward':0,'CUDA_initialized':False,
        'denoise_calls':0,'literal_copy_bytes':0,'fullstate_and_all80_raw_review_completed_by_this_script':False,
        'scope':'All three encoded-memory read paths affect this observed panel. K_P is prior-only without certified persistent Writer; K_C is input-complete per-token long/recent snapshot, not full-vNext layerwise prefix writer; K_E is a learned input index without literal selector/transport. No claim of complete original architecture or complete meaning fidelity.'}
    with (run/'root_three_memory_readouts_review.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:out[k] for k in ('status','normal_same16_NLL','each_memory_disabled_same16','additional_model_construction_forward','scope')},ensure_ascii=True))
if __name__=='__main__':main()
