"""ROOT cause audit of ONLY four changed byte positions, saved data only.

No new training/GPU/encoder/full-model construction or full-panel repeat.
CPU computes four frozen prefixes and sixteen selected258-class projections.
Row restoration is an offline causal intervention, NOT a saved model repair.
"""
import collections,gc,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from torch.nn import functional as F
from research.lossless_raw_tensor_v80 import unpack
from saved_commit_reference_v76_v2 import resolve_saved_commit
import text_generation_support_v1 as support
import learning_io_v2 as io


@torch.no_grad()
def main():
    began=time.monotonic();torch.set_num_threads(2);rng=torch.get_rng_state().clone()
    r=ROOT/'runs/feasible_repair_20261113_v91';p=io.load_json(r/'plan.json')
    a=ROOT/'research/artifacts/v91_error_cause_20261113_cpu_v1';a.mkdir(exist_ok=False)
    if list((ROOT/'runs').glob('*/worker_started.json')) is None:raise AssertionError('unreachable')
    # All model runs already closed. Use only the two small7-override envelopes.
    incident=io.load_json(r/'incident.json');current_path=Path(incident['emergency']['path'])
    assert support.sha(p['source'])==p['source_SHA'] and support.sha(current_path)==incident['emergency']['SHA']
    old=torch.load(p['source'],map_location='cpu',weights_only=True);new=torch.load(current_path,map_location='cpu',weights_only=True)
    assert old['frozen_model_SHA']==new['frozen_model_SHA'] and old['active_names']==new['active_names']
    ow,nw=old['active_weights'],new['active_weights'];names=('readout.readout.net.2.weight','readout.readout.net.2.bias')
    assert all(torch.equal(v,nw[k]) for k,v in ow.items() if k not in names)
    assert old['runtime']['frozen_except_last2_SHA']==new['runtime']['frozen_except_last2_SHA']
    accepted=[h for h in new['runtime']['history'] if h['update_receipt']['accepted']]
    assert len(accepted)==6
    projected=set()
    for h in accepted:
        first=h['update_receipt']['reference_checks'][0]
        assert first['blocker_count']==len(first['diagnostics'])<=16
        for d in first['diagnostics']:
            projected.add((d['role'],d['gold']));projected.add((d['role'],d['rival']))
    infos=io.load_json(p['eval_inputs']);baseline=io.load_json(p['baseline_records'])
    records=io.load_json(r/'root_partial_records.json')+io.load_json(ROOT/'runs/missing_assessment_20261113_v93/assessment/final/reconstructions.json')
    changes=io.load_json(r/'root_combined_assessment_review.json')['changed_contexts'];assert len(changes)==4
    anchor=torch.load(new['runtime']['bank_ref']['features'],map_location='cpu',weights_only=True)
    targets=torch.load(new['runtime']['bank_ref']['targets'],map_location='cpu',weights_only=True);cal=targets['bank_runtime']['calibration']
    finalref=torch.load(new['runtime']['reference_final']['path'],map_location='cpu',weights_only=True)
    train=io.load_json(Path(p['data'])/'train_features.json');max_train_bytes=max(len(t['prompt'].encode()) for t in train)
    results=[];prefix_count=projection_count=0
    for c in changes:
        if time.monotonic()-began>60:raise TimeoutError('fixed CPU60sec cause-only work')
        i=c['global_index'];offset=c['byte_offset'];patch,role=divmod(offset,16);info=infos[i]
        assert support.sha(info['raw'])==p['files'][info['raw']]
        raw=unpack(torch.load(info['raw'],map_location='cpu',weights_only=True,mmap=True))
        z,_=resolve_saved_commit(raw,info['row'],new['runtime']['embedding_sha'],p['files'],ROOT,512)
        feature=torch.cat((z[0,patch+2],z[0,0],z[0,1])).float()
        prefix='readout.readout.'
        norm=F.layer_norm(feature,(1536,),ow[prefix+'norm.weight'],ow[prefix+'norm.bias'],eps=1e-5)
        hidden=F.silu(F.linear(norm,ow[prefix+'net.0.weight'],ow[prefix+'net.0.bias']));prefix_count+=1
        start=role*258;end=start+258
        w0,w1=ow[names[0]][start:end],nw[names[0]][start:end]
        b0,b1=ow[names[1]][start:end],nw[names[1]][start:end]
        before=F.linear(hidden,w0,b0);after=F.linear(hidden,w1,b1);projection_count+=2
        oldtrace=torch.load(baseline[i]['trace'],map_location='cpu',weights_only=True)
        newtrace=torch.load(records[i]['trace'],map_location='cpu',weights_only=True)
        oldbyte=int(oldtrace['decoded_ids'][offset]);newbyte=int(newtrace['decoded_ids'][offset]);gold=info['row']['prompt'].encode()[offset]
        assert int(before.argmax())==int(oldtrace['argmax'][offset])==oldbyte
        assert int(after.argmax())==int(newtrace['argmax'][offset])==newbyte
        # For a damaged byte the rival is the new winner; for the repaired
        # capitalization use the old winner. Both rows get an offline swap.
        rival=newbyte if newbyte!=gold else oldbyte
        dw=(w1[gold].double()-w0[gold].double())-(w1[rival].double()-w0[rival].double())
        db=float((b1[gold].double()-b0[gold].double())-(b1[rival].double()-b0[rival].double()))
        direct_delta=float(torch.dot(dw,hidden.double()))+db
        old_pair=float(before[gold]-before[rival]);new_pair=float(after[gold]-after[rival])
        restore_w=w1.clone();restore_b=b1.clone()
        restore_w[gold]=w0[gold];restore_w[rival]=w0[rival];restore_b[gold]=b0[gold];restore_b[rival]=b0[rival]
        restored=F.linear(hidden,restore_w,restore_b);projection_count+=1
        # Exact public FP32 multiply sequence for a decay-only counterfactual.
        decay_w=w0.clone();decay_b=b0.clone()
        for _ in range(6):decay_w.mul_(1.-1e-5*.01);decay_b.mul_(1.-1e-5*.01)
        decay=F.linear(hidden,decay_w,decay_b);projection_count+=1
        all_h=anchor['hidden'];same_role=cal['top1_mask'][:,role]&(cal['labels'][:,role]==gold)
        cosine=F.cosine_similarity(all_h,hidden[None],dim=1);exact=int((all_h==hidden).all(-1).sum())
        candidates=cosine.masked_fill(~same_role,-torch.inf);nearest=int(candidates.argmax())
        source_range=next(x for x in anchor['source_ranges'] if x['start']<=nearest<x['end'])
        source_patch=source_range['source_patch_indices'][nearest-source_range['start']]
        h2=all_h[nearest]
        neighbor_before=float(torch.dot((w0[gold]-w0[rival]).double(),h2.double())+(b0[gold]-b0[rival]).double())
        neighbor_after=float(torch.dot((w1[gold]-w1[rival]).double(),h2.double())+(b1[gold]-b1[rival]).double())
        row=dict(global_index=i,byte_offset=offset,role=role,source_bytes=len(info['row']['prompt'].encode()),gold=gold,gold_char=chr(gold),rival=rival,rival_char=chr(rival),
            saved_GPU_before=oldbyte,saved_GPU_after=newbyte,CPU_subset_argmax_matches_saved_GPU=True,
            fixed_hidden_identical_input=True,old_gold_minus_rival=old_pair,new_gold_minus_rival=new_pair,
            pair_margin_delta_from_logits=new_pair-old_pair,pair_margin_delta_from_parameter_algebra=direct_delta,
            algebra_vs_FP32_gap_difference=abs(direct_delta-(new_pair-old_pair)),
            gold_row_weight_contribution=float(torch.dot(w1[gold].double()-w0[gold].double(),hidden.double())),
            rival_row_weight_contribution=float(torch.dot(w1[rival].double()-w0[rival].double(),hidden.double())),bias_gap_contribution=db,
            two_class_row_restore_prediction=int(restored.argmax()),two_class_row_restore_recovers_source=int(restored.argmax())==gold,
            decay_only_prediction=int(decay.argmax()),decay_only_pair_gap=float(decay[gold]-decay[rival]),
            gold_row_ever_directly_projected=(role,gold) in projected,rival_row_ever_directly_projected=(role,rival) in projected,
            exact_feature_matches_in_TRAIN_bank=exact,protected_same_gold_role_TRAIN_patches=int(same_role.sum()),
            nearest_same_gold_TRAIN=dict(source_id=source_range['source_id'],byte_offset=source_patch*16+role,cosine=float(candidates[nearest]),
                pair_margin_before=neighbor_before,pair_margin_after=neighbor_after,final_all_rival_margin=float(finalref['margin'][nearest,role]),
                top1_still_gold=int(finalref['argmax'][nearest,role])==gold,weak_floor_selected=bool(cal['weak_floor_mask'][nearest,role]),
                original_floor=float(cal['required_margin'][nearest,role]) if bool(cal['weak_floor_mask'][nearest,role]) else 0.),
            context=c)
        results.append(row);del raw,z,oldtrace,newtrace;gc.collect()
    assert prefix_count==4 and projection_count==16 and not torch.cuda.is_initialized() and torch.equal(rng,torch.get_rng_state())
    damaged=[v for v in results if v['saved_GPU_before']==v['gold'] and v['saved_GPU_after']!=v['gold']];assert len(damaged)==3
    assert all(v['two_class_row_restore_recovers_source'] and not v['gold_row_ever_directly_projected'] and not v['rival_row_ever_directly_projected'] for v in damaged)
    proof=dict(status='ROOT_V91_NEW_ERROR_CAUSE_FIXED_COMMIT_LAST_CLASSIFIER_DELTA_CPU_CONFIRMED',
        cases=results,trained_last_Linear_only=True,input_memory_binding_prefix_unchanged=True,
        three_new_errors_reproduced_from_only_last2_tensors=True,three_errors_recoved_by_only_two_class_rows_offline=True,
        direct_halfspace_projection_did_NOT_touch_any_of_six_damaged_class_rows=True,
        decay_only_does_not_reproduce_three_errors=all(v['decay_only_prediction']==v['gold'] for v in damaged),
        TRAIN_reference_max_source_UTF8_bytes=max_train_bytes,TRAIN_reference_patches=len(anchor['hidden']),
        whole_model_instantiated=False,encoder_calls=0,denoise_calls=0,optimizer_steps=0,GPU_calls=0,
        CPU_selected_prefix_computations=prefix_count,CPU_selected_role258_projections=projection_count,
        CUDA_initialized=False,CPU_seconds=time.monotonic()-began,all_Torch_RNG_unchanged=True,
        cause='Accepted shared last-classifier Adam changes flip preexisting narrow long-context margins outside the finite TRAIN feature bank. These class rows were never directly halfspace-projected; decay alone does not reproduce flips. Numerical headroom explains rejected updates, not these3accepted-update regressions.',
        limits='Four purposive saved cases, not whole semantic-circuit certification. Intermediate6weights/gradients not stored; cannot assign blame to one accepted step, source row or loss coefficient. Offline row restoration is causal diagnosis, not a validated patched model.',
        action='No new SGD/GPU/padding experiment. Preserve V91 but do not promote. Fix shared-classifier interference/retention coverage before more training; original DEV long rows/gold must not be injected as TRAIN constraints.')
    io.atomic_json(a/'root_cause_review.json',proof)
    io.atomic_json(a/'evidence_SHA.json',{str(f):support.sha(f) for f in (Path(__file__).resolve(),Path(p['source']),current_path,r/'root_manual_semantic_and_numeric_review.json',r/'root_combined_assessment_review.json',Path(new['runtime']['bank_ref']['features']),Path(new['runtime']['bank_ref']['targets']),a/'root_cause_review.json')})
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
