"""One bounded saved-logit/prefix calculation; no model or new evaluation."""
import json,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(2);run=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v26'
    out=ROOT/'research/artifacts/owt_v26_prefix_diagnostic_20261029_cpu';out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    native=read(run/'code_snapshot/root_terminal_native_observation.json')
    assert native['native_python_count']==0 and read(run/'supervisor/final_status.json')['exit']['all_owned_processes_gone']
    p=read(run/'plan.json');data=torch.load(run/'final_logits.pt',weights_only=True,map_location='cpu')
    ids=data['target_ids'];pred=data['normal'].argmax(-1);active=ids!=257
    first=[];later_correct=later_count=0
    for i in range(len(ids)):
        length=int(active[i].sum());errors=(pred[i,:length]!=ids[i,:length]).nonzero().flatten()
        j=int(errors[0]) if len(errors) else length;first.append(j)
        if j<length:
            later_correct+=int((pred[i,j+1:length]==ids[i,j+1:length]).sum());later_count+=length-j-1
    raw=read(run/'final_generation.json')['normal'];rows=[]
    for i,row in enumerate(raw):
        index=p['generation_indices'][i];gold=ids[index,:int(active[index].sum())].tolist();actual=row['generated']['token_ids']
        j=next((n for n,(a,b) in enumerate(zip(gold,actual)) if a!=b),min(len(gold),len(actual)))
        t=first[index]
        rows.append({'index':i,'id':row['id'],'teacher_first_mismatch_token':t,'state_only_first_mismatch_token':j,
            'same_first_error_position':j==t,'same_predicted_symbol_at_first_error':j==t and t<len(gold) and actual[t]==int(pred[index,t]),
            'source_bytes':len(row['source'].encode()),'initial_state_only_bytes':j})
    result={'status':'SAVED_V26_PREFIX_CALCULATION_COMPLETE','teacher_rows':len(ids),
        'teacher_all_argmax_rows_exact':sum(first[i]==int(active[i].sum()) for i in range(len(ids))),
        'teacher_first_mismatch_mean':statistics.mean(first),'teacher_first_mismatch_median':statistics.median(first),
        'teacher_after_first_mismatch_correct':later_correct,'teacher_after_first_mismatch_symbols':later_count,
        'teacher_after_first_mismatch_accuracy':later_correct/later_count,
        'fixed16_first_error_position_matches':sum(r['same_first_error_position'] for r in rows),
        'fixed16_first_error_symbol_matches':sum(r['same_predicted_symbol_at_first_error'] for r in rows),
        'fixed16_teacher_initial_mean':statistics.mean(first[p['generation_indices'][i]] for i in range(16)),
        'fixed16_state_only_initial_mean':statistics.mean(r['initial_state_only_bytes'] for r in rows),'rows':rows,
        'evidence_sha256':{n:support.sha(run/n) for n in ('plan.json','final_logits.pt','final_generation.json')},
        'new_model_forwards':0,'optimizer_updates':0,'new_eval_rows':0,'E2_final_accessed':False,
        'interpretation':'Exact agreement up to an early first error is compatible with causal decoding. Teacher later accuracy uses restored GOLD history, while state-only decoding does not. This supports investigating prefix-training mismatch, but the first error itself precedes any erroneous self-history and cannot be blamed on accumulated prefix errors. Not sole extractor/decoder cause proof.'}
    with (out/'result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','evidence_sha256')},ensure_ascii=True))
if __name__=='__main__':main()
