"""Saved-only fullstate/trace review plus velocity-range algebra. No E/R/D run."""
from pathlib import Path
import collections,hashlib,json,math,sys,time
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'scripts'),str(P),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from review_pause_20261114_cpu import independent_decode,validate_trace,sha

def main():
    began=time.monotonic();torch.set_num_threads(2);run=Path('D:/Codex/FDT/runs/identity_thought_20261114_v109')
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json');result=io.load_json(run/'result.json');p=io.load_json(run/'plan.json')
    assert terminal['PythonCount']==0 and terminal['All4OwnedProxyGone'] and terminal['BeforeResultAnalysis']
    assert owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
    assert result['attempts']==result['updates']==134 and result['AMP_skips']==0 and result['calls']==dict(denoise=21808,readout=1296,encode=0)
    states=[]
    for ref in result['checkpoints']:
        assert sha(ref['path'])==ref['SHA'] and Path(ref['path']).stat().st_size==ref['bytes']
        q=torch.load(ref['path'],map_location='cpu',weights_only=True);assert support.state_sha(q)==ref['algorithm_SHA'] and q['plan_SHA']==sha(run/'plan.json');states.append(q)
    initial,middle,final=states;b=initial['engine'];f=final['engine'];r=f['runtime']
    assert [s['engine']['runtime']['updates'] for s in states]==[0,67,134]
    assert f['schema']=='runtime_aligned_fullstate_v107' and r['phase']=='DONE' and r['cursor']==1072 and len(r['history'])==134
    assert r['exposure_bytes']==result['exposure_target_bytes']==67832 and r['calls']==dict(denoise=18224,readout=1072)
    assert b['optimizer']['state']=={} and all(int(x['step'])==134 for x in f['optimizer']['state'].values())
    assert f['scaler']['scale']==1024 and f['scaler']['_growth_tracker']==134 and all(g is None for g in f['gradients'].values())
    counts=collections.Counter(f['orders']);assert len(counts)==134 and set(counts.values())=={8}
    selected=io.load_json(Path(p['screen'])/'selected_TRAIN_only.json');assert set(counts)=={x['id'] for x in selected}
    assert all(not h['skip'] and h['rows']==8 for h in r['history'])
    assert [i for h in r['history'] for i in h['source_ids']]==f['orders']
    gen=torch.Generator().set_state(b['ALL_RNG']['torch'])
    for _ in range(1072):torch.randn(1,35,512,generator=gen)
    assert torch.equal(gen.get_state(),f['ALL_RNG']['torch']) and b['ALL_RNG']['python']==f['ALL_RNG']['python'] and b['ALL_RNG']['numpy']==f['ALL_RNG']['numpy']
    assert all(s['engine']['ALL_RNG']['CUDA']==[] for s in states)
    changed=[k for k in f['thought'] if not torch.equal(b['thought'][k],f['thought'][k])];assert len(changed)==64
    from research import linked_lastlinear_runtime_v79 as linked
    choice=io.load_json(p['selection']);frozen=linked.load_verified(choice['selected_runtime'],choice['selected_SHA'],P,support.state_sha)
    assert support.state_sha(frozen['model'])==choice['source_model_SHA']==result['frozen_codec_SHA_before_after']
    from review_memory_thought_v105_cpu import endpoint_digest
    frozen_digest=endpoint_digest(frozen['model']);assert all(s['engine']['frozen_endpoint_digest']==frozen_digest for s in states)
    terminal_vector=frozen['model']['readout.readout.terminal'];del frozen
    cache=Path(p['cache']);conditions={x['id']:x for vv in io.load_json(cache/'input_index.json').values() for x in vv}
    rows=io.load_json(run/'assessment.json');assert len(rows)==224;groups=collections.defaultdict(list)
    for row in rows:
        ref=row['raw'];assert sha(ref['path'])==ref['SHA'];raw=torch.load(ref['path'],map_location='cpu',weights_only=True)
        text,eos,length=independent_decode(raw['trace']);assert text==row['output'] and eos==row['EOS']
        score=validate_trace(raw['trace'],row['expected']);assert abs(score['NLL']-row['byte_NLL'])<1e-6
        source=torch.load(conditions[row['input_id']]['path'],map_location='cpu',weights_only=True)['source']
        target=terminal_vector.reshape(1,1,512).expand(1,35,512).clone();target[:,:source.shape[1]]=source
        error=(raw['final_semantic']-target).square().mean(-1)[0];valid=source.shape[1]+1
        groups[row['phase']+'/'+row['split']].append(dict(row,length=length,early_EOS=eos and length<len(row['expected'].encode()),
            byte_accuracy=score['argmax_byte_accuracy'],valid_RMSE=float(error[:valid].mean().sqrt()),tail_RMSE=float(error[valid:].mean().sqrt())))
    summaries={}
    for group,rr in groups.items():
        summaries[group]=dict(rows=len(rr),exact=sum(x['exact'] for x in rr),mean_NLL=sum(x['byte_NLL'] for x in rr)/len(rr),
            EOS=sum(x['EOS'] for x in rr),early_EOS=sum(x['early_EOS'] for x in rr),byte_accuracy=sum(x['byte_accuracy'] for x in rr)/len(rr),
            valid_RMSE=sum(x['valid_RMSE'] for x in rr)/len(rr),tail_RMSE=sum(x['tail_RMSE'] for x in rr)/len(rr))
        good=[x for x in rr if x['codec_roundtrip_exact']]
        summaries[group]['codec_pass_subset']=dict(rows=len(good),exact=sum(x['exact'] for x in good),mean_NLL=sum(x['byte_NLL'] for x in good)/len(good) if good else None)
    curve={}
    for name,hh in [('first8',r['history'][:8]),('last8',r['history'][-8:])]:
        curve[name]=dict(loss={k:sum(h['loss'][k] for h in hh)/len(hh) for k in ('latent','byte_CE','EOS_CE')},
            actual_trajectory={k:sum(h['runtime_loss'][k] for h in hh)/len(hh) for k in ('runtime_body_MSE','runtime_tail_MSE','aux_VP')})
    # Exact algebraic restriction: v = W*h + bias, W512x256. In the
    # orthogonal complement of W, the DDIM recursion retains cos(delta)^N
    # of the initial independent noise. This is not a new model evaluation.
    weights=f['thought'];matrix=weights['velocity.weight'].double();U,S,Vh=torch.linalg.svd(matrix,full_matrices=True)
    rank=int((S>max(matrix.shape)*torch.finfo(torch.float64).eps*S[0]).sum());Q=U[:,rank:]
    assert matrix.shape==(512,256) and rank==256 and float((matrix.T@Q).abs().max())<1e-12
    factors={str(n):math.cos(math.pi/(2*n))**n for n in (1,4,16)};k=factors['16'];checks=[]
    for row in [x for x in rows if x['phase']=='final' and x['split']=='TRAIN_probe']:
        raw=torch.load(row['raw']['path'],map_location='cpu',weights_only=True);source=torch.load(conditions[row['input_id']]['path'],map_location='cpu',weights_only=True)['source']
        summary=source[:,:2].float().mean(1)
        # One affine stored-parameter calculation per row ONLY to recover the
        # input-dependent normalization. No E/denoiser/D module is constructed.
        scale=.5+(summary@weights['prior_scale.weight'].T+weights['prior_scale.bias']).sigmoid()
        noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(raw['noise_seed']))
        z=raw['final_semantic'][0].double();normalized_difference=(z[1:]-z[:1])/scale.double()
        projected=normalized_difference@Q;predicted=k*((noise[0,1:]-noise[0,:1]).double()@Q)
        checks.append(dict(id=row['input_id'],relative_L2=float((projected-predicted).norm()/predicted.norm()),
            max_abs_error=float((projected-predicted).abs().max()),observed_retention_ratio=float(projected.norm()/(((noise[0,1:]-noise[0,:1]).double()@Q).norm()))))
    assert max(x['relative_L2'] for x in checks)<2e-5
    algebra=dict(status='SAVED_V109_VELOCITY_RANK_RESTRICTION_CONFIRMED',velocity_shape=list(matrix.shape),rank=rank,
        uncontrolled_orthogonal_dimensions=512-rank,initial_noise_std_retention=factors,
        final16_noise_variance_retention=k*k,per_dimension_expected_MSE_lower_bound_from_scale_min_half=.25*k*k*(512-rank)/512,
        formula='Q r_next = cos(delta) Q r - sin(delta) Q bias; QW=0; afterN: noise coefficient cos(delta)^N',
        proof_by_slot_differences_cancels_source_centre_and_velocity_bias=True,saved_TRAIN_probe16_checks=checks,
        no_E_R_D_forward=True,optimizer0=True,stored_affine_scale_calculations=16,SVD512x256=1,
        qualification='Hard obstruction to removing full512 independent Gaussian noise, not proof all decoder errors caused by this or all output meaning absent.')
    io.atomic_json(run/'root_velocity_rank_algebra.json',algebra)
    review=dict(status='ROOT_VERIFIED_MEANING_IDENTITY134_REVIEWED',whole_codec_frozen_digest_verified=frozen_digest,
        only64_thought_tensors_changed=True,checkpoint3_allSHA_recursive_safeload=True,all_Adam_steps134=True,scaler1024_tracker134=True,
        all1072_training_noise_draws_exact=True,selected134_each8_finite_only=True,all224_saved_trace_independent_review=True,
        NOT_all258_all_positions_saved=True,groups=summaries,training_curve=curve,
        velocity_rank256_uncontrolled_noise_dims256=True,all16_saved_sample_projection_matches_formula=True,
        E_R_D_additional_forwards0=True,affine_normalization_algebra16=True,CUDA_initialized=torch.cuda.is_initialized(),
        elapsed=time.monotonic()-began,all_trial_budgets_closed=True,unused396calls_no_reuse=True)
    io.atomic_json(run/'root_saved_review.json',review)
    io.atomic_json(run/'root_result.json',dict(status='IDENTITY_CONTROL_COMPLETE_DENOISER_NOT_RECOVERING_YET_RANK_DEFECT_FOUND',actual_updates134=True,
        checkpoint_SHA=result['checkpoints'][-1]['SHA'],review_SHA=sha(run/'root_saved_review.json'),algebra_SHA=sha(run/'root_velocity_rank_algebra.json'),
        source_codec_unchanged=True,no_new_training_after_terminal=True,no_promotion=True))
    print(json.dumps(dict(groups=summaries,curve=curve,rank=rank,noise_retention=factors,max_projection_relative_error=max(x['relative_L2'] for x in checks),seconds=review['elapsed'])))

if __name__=='__main__':main()
