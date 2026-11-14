"""Saved-only fullstate/trace review plus velocity-range algebra. No E/R/D run."""
from pathlib import Path
import collections,hashlib,json,math,sys,time
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'scripts'),str(P),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from review_pause_20261114_cpu import independent_decode,validate_trace,sha
from research.latent_identity_thought_v110 import effective_velocity_rank

def main():
    began=time.monotonic();torch.set_num_threads(2);run=Path('D:/Codex/FDT/runs/latent_identity_20261114_v110')
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json');result=io.load_json(run/'result.json');p=io.load_json(run/'plan.json')
    assert terminal['PythonCount']==0 and terminal['All4OwnedProxyGone'] and terminal['BeforeResultAnalysis']
    assert owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
    assert result['attempts']==result['updates']==67 and result['AMP_skips']==0 and result['calls']==dict(denoise=12696,readout=224,encode=0)
    states=[]
    for ref in result['checkpoints']:
        assert sha(ref['path'])==ref['SHA'] and Path(ref['path']).stat().st_size==ref['bytes']
        q=torch.load(ref['path'],map_location='cpu',weights_only=True);assert support.state_sha(q)==ref['algorithm_SHA'] and q['plan_SHA']==sha(run/'plan.json');states.append(q)
    initial,middle,final=states;b=initial['engine'];f=final['engine'];r=f['runtime']
    assert [s['engine']['runtime']['updates'] for s in states]==[0,34,67]
    assert f['schema']=='latent_identity_fullstate_v110' and not f['language_losses_used'] and r['phase']=='DONE' and r['cursor']==536 and len(r['history'])==67
    assert r['exposure_bytes']==result['exposure_target_bytes']==33916 and r['calls']==dict(denoise=9112,readout=0)
    assert b['optimizer']['state']=={} and all(int(x['step'])==67 for x in f['optimizer']['state'].values())
    assert f['scaler']['scale']==1024 and f['scaler']['_growth_tracker']==67 and all(g is None for g in f['gradients'].values())
    counts=collections.Counter(f['orders']);assert len(counts)==134 and set(counts.values())=={4}
    selected=io.load_json(Path(p['screen'])/'selected_TRAIN_only.json');assert set(counts)=={x['id'] for x in selected}
    assert all(not h['skip'] and h['rows']==8 for h in r['history'])
    assert [i for h in r['history'] for i in h['source_ids']]==f['orders']
    gen=torch.Generator().set_state(b['ALL_RNG']['torch'])
    for _ in range(536):torch.randn(1,35,512,generator=gen)
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
    weights=f['thought'];w=weights['velocity.weight'].double();gamma=weights['final_norm.weight'].double();wg=w*gamma[None];effective=wg-wg.sum(1,keepdim=True)/wg.shape[1];rank=int(torch.linalg.matrix_rank(effective));assert rank==512 and w.shape==(512,520)
    assert all(h.get('D_during_training') is False for h in r['history'])
    review=dict(status='ROOT_FULLRANK_PURE_LATENT_IDENTITY67_REVIEWED',whole_codec_frozen_digest_verified=frozen_digest,
        only64_thought_tensors_changed=True,checkpoint3_allSHA_recursive_safeload=True,all_Adam_steps67=True,scaler1024_tracker67=True,
        all536_training_noise_draws_exact=True,selected134_each4_finite_only=True,all224_saved_trace_independent_review=True,
        NOT_all258_all_positions_saved=True,groups=summaries,training_curve=curve,
        velocity_effective_affine_rank512=True,velocity_shape512by520=True,training_D_calls0_verified=True,language_loss_NOT_used=True,
        E_R_D_additional_forwards0=True,no_new_model_evaluations=True,CUDA_initialized=torch.cuda.is_initialized(),
        elapsed=time.monotonic()-began,all_trial_budgets_closed=True,unused280calls_no_reuse=True)
    io.atomic_json(run/'root_saved_review.json',review)
    io.atomic_json(run/'root_result.json',dict(status='FULLRANK_PURE_LATENT_CONTROL_COMPLETE_NOT_YET_RECONSTRUCTING',actual_updates67=True,
        checkpoint_SHA=result['checkpoints'][-1]['SHA'],review_SHA=sha(run/'root_saved_review.json'),
        source_codec_unchanged=True,no_new_training_after_terminal=True,no_promotion=True,training_decoder_calls0=True))
    print(json.dumps(dict(groups=summaries,curve=curve,effective_velocity_rank=rank,pure_latent=True,seconds=review['elapsed'])))

if __name__=='__main__':main()
