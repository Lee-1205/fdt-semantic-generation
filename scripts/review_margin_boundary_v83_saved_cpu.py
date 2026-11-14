"""Independently reconstruct saved TRAIN projection margins, no NN calls."""
import bisect,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(4);run=ROOT/'runs/margin_boundary_diagnostic_20261113_v83';p=io.load_json(run/'plan.json');r=io.load_json(run/'result.json');term=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert term['AllOwnedGone'] and not term['PythonProcesses'] and term['BeforeAnalysis'] and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
    for f,s in p['files'].items():assert support.sha(f)==s,f
    assert support.sha(run/'raw_projection_statistics.pt')==r['raw_SHA'];raw=torch.load(run/'raw_projection_statistics.pt',map_location='cpu',weights_only=True);t=torch.load(p['bank_targets'],map_location='cpu',weights_only=True);y=t['labels'];valid=y!=257;critical=t['critical_mask']
    sample_verified=0
    for key,v in raw.items():
        assert torch.equal(v['margin'],v['gold_score']-v['best_other_score']) and torch.isfinite(v['margin']).all()
        sample=v['sample_full_logits'];idx=v['sample_patch_indices'];labels=y[idx];gold=sample.gather(-1,labels[...,None]).squeeze(-1);other=sample.clone();other.scatter_(-1,labels[...,None],-torch.inf)
        assert torch.equal(sample.argmax(-1),v['argmax'][idx].long()) and torch.equal(gold,v['gold_score'][idx]) and torch.equal(other.max(-1).values,v['best_other_score'][idx]) and torch.equal(other.max(-1).indices,v['best_other_index'][idx].long());sample_verified+=labels.numel()
    assert torch.equal(raw['V75/bf16']['argmax'].long(),t['initial_argmax'].long())
    final=torch.load(p['bank_final_predictions'],map_location='cpu',weights_only=True);assert torch.equal(raw['V80/bf16']['argmax'].long(),final['argmax'].long())
    old=io.load_json(ROOT/'runs/update_retention_20261113_v80/plan.json');f=Path(old['data'])/'train_features.json';features=io.load_json(f);lengths=[(len(features[i]['prompt'].encode())+15)//16+1 for i in t['source_indices']];ends=[];s=0
    for n in lengths:s+=n;ends.append(s)
    assert s==7487
    summary={};cards=[]
    for mode in ('bf16','last_fp32'):
        before=raw['V75/'+mode];after=raw['V80/'+mode];mask=(before['argmax'].long()==y)&valid;ok=(after['argmax'].long()==y)&valid
        gap=before['margin'];nextgap=after['margin'];floor=gap.clamp(min=0,max=1.)*.5;erosion=mask&(nextgap<floor);flips=mask&~ok
        delta=nextgap-gap
        summary[mode]=dict(valid_positions=int(valid.sum()),initial_correct=int(mask.sum()),final_correct=int(ok.sum()),initial_correct_lost=int(flips.sum()),initial_wrong_corrected=int((~mask&ok&valid).sum()),
            protected_margin_floor_violations=int(erosion.sum()),critical_margin_violations=int((erosion&critical).sum()),noncritical_EOS_margin_violations=int((erosion&~critical).sum()),
            protected_margins_decreased=int((mask&(delta<0)).sum()),protected_margins_increased=int((mask&(delta>0)).sum()),min_initial_correct_margin=float(gap[mask].min()),min_final_on_protected_margin=float(nextgap[mask].min()),
            delta_quantiles=[float(v) for v in torch.quantile(delta[mask],torch.tensor([0.,.01,.5,.99,1.]))],prospective_fraction=.5,prospective_cap=1.)
        for patch,role in (erosion|flips).nonzero().tolist():
            sourcepos=bisect.bisect_right(ends,patch);start=0 if sourcepos==0 else ends[sourcepos-1];sourceindex=t['source_indices'][sourcepos];row=features[sourceindex];offset=(patch-start)*16+role;source=row['prompt'].encode();g=int(y[patch,role]);a=int(before['argmax'][patch,role]);b=int(after['argmax'][patch,role])
            assert g==(source[offset] if offset<len(source) else 256)
            cards.append(dict(mode=mode,source_index=sourceindex,source_id=row['id'],byte_offset=offset,critical=bool(critical[patch,role]),gold=g,gold_char=chr(g) if g<256 else 'EOS',before=a,after=b,
                before_char=chr(a) if a<256 else str(a),after_char=chr(b) if b<256 else str(b),initial_margin=float(gap[patch,role]),final_margin=float(nextgap[patch,role]),floor=float(floor[patch,role]),
                source_context=source[max(0,offset-55):offset+65].decode(errors='replace'),top1_flipped=bool(flips[patch,role])))
    receipt=r['margin_candidate_endpoint_receipt'];assert receipt['measurements']['new_top1_flips']==summary['last_fp32']['initial_correct_lost'] and receipt['measurements']['margin_floor_violations']==summary['last_fp32']['protected_margin_floor_violations'] and receipt['accept']==(not summary['last_fp32']['protected_margin_floor_violations'] and not summary['last_fp32']['initial_correct_lost'])
    report=dict(status='ROOT_V83_TRAIN_MARGIN_BOUNDARY_INDEPENDENT_REVIEW_COMPLETE',actual_new_GPU_projection_calls=4,extra_model_calls=0,summary=summary,cards=cards,
        verified_native_BF16_argmax_positions=2*y.numel(),full_saved_margins_reconstructed=4*y.numel(),full_class_sample_verifications=sample_verified,
        TRAIN_feature_source_SHA=support.sha(f),both_NATIVE_BF16_diagonals_exact=True,weights_Adam_scaler_RNG_unchanged=True,
        caveat='Only128patches per endpoint have full258-class scores saved; remaining argmax maxima originate in worker statistics and BF16 native full-bank replay checks. Last-FP32 retains historical BF16prefix, not full-FP32 runtime. Two endpoints, no historical per-step replay; no causal proof of an unrun new training trajectory or general semantic collapse.')
    io.atomic_json(run/'root_saved_review.json',report);print(json.dumps(dict(summary=summary,cards=cards,full_class_sample_verifications=sample_verified),ensure_ascii=False))

if __name__=='__main__':main()
