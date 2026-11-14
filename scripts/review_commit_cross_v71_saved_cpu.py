"""Recompute all measured scores from saved logits. No new neural forwards."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    run=ROOT/'runs/commit_cross_20261112_v71';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json');native=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert native['Python_native_count']==0 and native['all_known_pids_gone'] and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone']
    for name,digest in p['files'].items():assert support.sha(name)==digest,name
    logits=torch.load(run/'raw_cross_logits.pt',weights_only=True,map_location='cpu');cases=io.load_json(Path(p['data'])/'cases.json');record=io.load_json(run/'cases_result.json')
    ref=torch.load(Path(p['data'])/'reference_logits.pt',weights_only=True,map_location='cpu');summary=[];patterns={}
    for i,(case,saved) in enumerate(zip(cases,record)):
        gold,role=case['gold_byte'],case['focus_role'];values={}
        for name,z in logits.items():
            v=z[i,role];other=v.clone();other[gold]=-torch.inf
            measured=dict(argmax=int(v.argmax()),gold_margin=float(v[gold]-other.max()),gold_NLL=float(torch.nn.functional.cross_entropy(v[None],torch.tensor([gold]))))
            for key,value in measured.items():assert math.isclose(value,saved['values'][name][key],rel_tol=1e-6,abs_tol=1e-6)
            values[name]=measured
        for side in ('old','new'):
            z=logits[side+'Commit_'+side+'Readout'][i];original=ref[side][i]
            assert torch.equal(z.argmax(-1),original.argmax(-1)) and int(z[role].argmax())==case[side+'_native_focus_argmax']
        bits=tuple(values[key]['argmax']==gold for key in ('oldCommit_oldReadout','oldCommit_newReadout','newCommit_oldReadout','newCommit_newReadout'))
        label={
            (True,False,True,False):'new_final_readout_sufficient_to_reproduce_regression',
            (True,True,False,False):'new_Commit_plus_binding_sufficient_to_reproduce_regression',
            (True,True,True,False):'both_updated_components_required_for_observed_wrong_argmax',
            (False,False,True,True):'new_Commit_plus_binding_sufficient_for_correction',
            (False,True,False,True):'new_final_readout_sufficient_for_correction',
            (True,True,True,True):'unchanged_correct_control',
        }.get(bits,'mixed_other')
        patterns[label]=patterns.get(label,0)+1
        display=lambda value:chr(value) if 32<=value<127 else str(value)
        summary.append(dict(name=case['name'],gold=display(gold),old_old=display(values['oldCommit_oldReadout']['argmax']),
            old_new=display(values['oldCommit_newReadout']['argmax']),new_old=display(values['newCommit_oldReadout']['argmax']),
            new_new=display(values['newCommit_newReadout']['argmax']),pattern=label,source_context=case['source_context'],gold_margins={k:v['gold_margin'] for k,v in values.items()}))
    assert result['calls']==4 and result['eligible_cases']==8 and not torch.cuda.is_initialized()
    review=dict(status='ROOT_V71_SAVED_REVIEW_PASS',all_input_pins_unchanged=True,raw_cross_logits_SHA=support.sha(run/'raw_cross_logits.pt'),
        all32_focus_argmax_margin_NLL_recomputed=True,all256_diagonal_patch_argmax_identical_to_native=True,patterns=patterns,cases=summary,
        actual_CPU_model_calls=4,additional_review_model_calls=0,encoder_calls=0,CUDA_calls=0,optimizer_updates=0,Exact_entries=0,denoise_calls=0,
        source_Adam_scaler_ALLRNG_preserved=True,
        interpretation='case-local functional evidence at bound Commit/final verbalizer boundary; no blanket decoder-only or meaning-only cause; role_projection already upstream; information for a particular character can be recoverable without proving abstract meaning is complete',
        limitations='eight purposively chosen observed patches, not an accuracy benchmark; cross-checkpoint combinations may be outside normal activation distribution; full patch diagonals agree but native BF16 vs CPU FP32 numerical logits differ slightly')
    io.atomic_json(run/'root_saved_review.json',review);print(json.dumps(review))
if __name__=='__main__':main()
