"""Compare existing teacher/BOS errors and scheduler arithmetic; no NN forward."""
import json,math,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2);runs={v:ROOT/f'runs/three_memory_codec_20261030_v{v}' for v in (41,42)}
    raw={v:json.loads((p/'focused_final_reconstruction.json').read_text()) for v,p in runs.items()}
    tensors={v:torch.load(p/'focused_final_logits.pt',weights_only=True,map_location='cpu') for v,p in runs.items()}
    results={v:json.loads((p/'root_result.json').read_text()) for v,p in runs.items()}
    rows=[];totals={}
    for v,t in tensors.items():
        ids=t['target_ids'];active=ids!=257;pred=t['normal'].argmax(-1)
        totals[v]={'teacher_errors':int(((pred!=ids)&active).sum()),
                   'rows_with_teacher_error':int((((pred!=ids)&active).any(-1)).sum())}
    for i in range(64):
        r={'index':i,'exact41':raw[41][i]['reconstruction_exact'],'exact42':raw[42][i]['reconstruction_exact']}
        for v,t in tensors.items():
            ids=t['target_ids'][i];z=t['normal'][i].double();n=int((ids!=257).sum());pred=z.argmax(-1)
            first=next((j for j in range(n) if pred[j]!=ids[j]),None)
            value={'teacher_error_count':int((pred[:n]!=ids[:n]).sum()),'first_teacher_error':first}
            if first is not None:
                gold=int(ids[first]);other=z[first].clone();other[gold]=-torch.inf
                value.update(gold_id=gold,predicted_id=int(pred[first]),
                             saved_FP16_gold_minus_best_other=float(z[first,gold]-other.max()))
            r[str(v)]=value
        rows.append(r)
    losses=[r for r in rows if r['exact41'] and not r['exact42']]
    factor=.1+.9*(1+math.cos(math.pi*8191/8192))/2
    stats={'status':'V42_SAVED_ERROR_LOCATION_AND_LR_ARITHMETIC_COMPLETE','all_model_calls':0,
        'teacher_error_distribution':totals,'exact_losses':losses,'all_focused_rows':rows,
        'loss_first_margin_abs_median':statistics.median(abs(r['42']['saved_FP16_gold_minus_best_other']) for r in losses),
        'loss_first_margin_abs_le_1':sum(abs(r['42']['saved_FP16_gold_minus_best_other'])<=1 for r in losses),
        'V42_teacher_BOS_first_error_mismatches':results[42]['teacher_BOS_first_error_mismatch_count'],
        'terminal_cosine_factor':factor,'next_same_base_cosine_restart_LR_multiplier':1/factor,
        'candidate_constant_terminal_LRs':[b*factor for b in (1e-5,1e-4,3e-6,1e-4)],
        'interpretation':'Teacher errors already appear at the first BOS divergence; later repetition alone cannot explain them. Same-base restart would increase all group LRs about10x. Neither observation proves encoder-vs-decoder or LR causality. A low-rate continuation is a prospective stabilization experiment, not a matched causal comparison.'}
    assert not torch.cuda.is_initialized()
    path=runs[42]/'root_saved_error_margins_and_LR.json'
    with path.open('x',encoding='utf-8') as f:json.dump(stats,f,indent=2)
    print(json.dumps({k:v for k,v in stats.items() if k not in ('all_focused_rows','exact_losses')}))
if __name__=='__main__':main()
