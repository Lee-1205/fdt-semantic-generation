"""Saved-only review of first CPU thought trial. NO new model forward/SGD."""
import collections,hashlib,json,math,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.readout_decision_trace_v88 import decode,masks


def endpoint_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous();h.update(name.encode());h.update(str((tuple(value.shape),str(value.dtype))).encode());h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    began=time.monotonic();torch.set_num_threads(2);run=ROOT/'runs/memory_thought_20261114_v102';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json')
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert terminal['before_analysis'] and terminal['Python_count']==0 and terminal['all_owned_native_proxy_gone']
    assert owner['status']=='COMPLETE' and owner['exit_code']==0 and owner['exit']['all_owned_processes_gone']
    assert result['execution_device']=='cpu' and not result['CUDA_initialized'] and result['attempts']==result['updates']==192 and result['AMP_skips']==0
    for name,sha in p['files'].items():assert support.sha(name)==sha,name
    states=[]
    for ref in result['checkpoints']:
        assert support.sha(ref['path'])==ref['SHA'] and Path(ref['path']).stat().st_size==ref['bytes']
        s=torch.load(ref['path'],map_location='cpu',weights_only=True);assert support.state_sha(s)==ref['algorithm_SHA']
        assert s['plan_SHA']==support.sha(run/'plan.json') and s['engine']['source_metadata']['execution_device']=='cpu'
        states.append(s)
    first,middle,last=states;before=first['engine'];after=last['engine'];r=after['runtime']
    assert [s['engine']['runtime']['updates'] for s in states]==[0,96,192]
    assert r['phase']=='DONE' and r['cursor']==1536 and r['exposure_bytes']==27136 and len(r['history'])==192
    assert all(h['rows']==8 and not h['skip'] for h in r['history']) and all(g is None for g in after['gradients'].values())
    assert [x for h in r['history'] for x in h['source_ids']]==after['orders']
    assert len(set(after['orders']))==384 and set(collections.Counter(after['orders']).values())=={4}
    assert r['calls']==dict(denoise=1536,readout=1536) and after['scaler']['scale']==1024 and after['scaler']['_growth_tracker']==192
    assert before['optimizer']['state']=={} and all(int(s['step'])==192 for s in after['optimizer']['state'].values())
    assert before['optimizer_names']==after['optimizer_names'] and len(after['optimizer_names'])==len(after['optimizer']['state'])
    assert all(s['engine']['ALL_RNG']['CUDA']==[] for s in states)
    # Independently reconstruct every global train time/noise draw. Evaluation
    # uses its own per-row generator and must not consume the global stream.
    generator=torch.Generator(device='cpu');generator.set_state(before['ALL_RNG']['torch'])
    for _ in range(1536):torch.rand(1,generator=generator);torch.randn((1,35,512),generator=generator)
    assert torch.equal(generator.get_state(),after['ALL_RNG']['torch'])
    assert before['ALL_RNG']['python']==after['ALL_RNG']['python'] and before['ALL_RNG']['numpy']==after['ALL_RNG']['numpy']
    changed=[k for k in before['thought'] if not torch.equal(before['thought'][k],after['thought'][k])]
    assert changed and all(not k.startswith(('codec.','encoder.','readout.')) for k in after['optimizer_names'])
    # Verify the frozen complete endpoint including buffers, without creating a
    # full NN or running E/D again. Parent Adam/RNG fullstate remains external.
    from research import linked_lastlinear_runtime_v79 as linked
    selection=io.load_json(p['selection']);parent=linked.load_verified(selection['selected_runtime'],selection['selected_SHA'],ROOT,support.state_sha)
    frozen=endpoint_digest(parent['model']);assert all(s['engine']['frozen_endpoint_digest']==frozen for s in states)
    assert support.state_sha(parent['model'])==result['frozen_codec_SHA_before_after']==selection['source_model_SHA']
    cache=Path(p['cache']);target_refs=io.load_json(cache/'target_index_loss_only.json');targets={x['answer_SHA']:torch.load(x['target']['path'],map_location='cpu',weights_only=True) for x in target_refs}
    rows=io.load_json(run/'assessment.json');assert len(rows)==208;groups=collections.defaultdict(list)
    for row in rows:
        ref=row['raw'];assert support.sha(ref['path'])==ref['SHA'] and Path(ref['path']).stat().st_size==ref['bytes']
        raw=torch.load(ref['path'],map_location='cpu',weights_only=True);trace=raw['trace'];out=decode(trace)
        assert raw['input_id']==row['input_id'] and raw['no_target_condition'] and raw['denoise_steps16'] and raw['final_semantic'].shape==(1,35,512)
        assert out['text']==row['output'] and out['EOS']==row['EOS'] and out['UTF8']==row['UTF8']
        target=targets[hashlib.sha256(row['expected'].encode()).hexdigest()];labels=target['labels'];active=labels.flatten()!=257
        assert abs(float(-trace['gold_log_probability'][active].mean())-row['byte_NLL'])<1e-6
        for sample,values in zip(trace['sample_patch_indices'].tolist(),trace['sample_full_logits']):
            start=sample*16;assert torch.equal(values.argmax(-1).to(torch.int16),trace['argmax'][start:start+16])
            decisions=torch.stack([values.masked_fill(~m,-torch.inf).argmax(-1).to(torch.int16) for m in masks('cpu')])
            assert torch.equal(decisions,trace['choices'][:,start:start+16])
            lp=values.gather(-1,labels[sample,:,None]).squeeze(-1)-values.logsumexp(-1)
            assert torch.allclose(lp,trace['gold_log_probability'][start:start+16],atol=3e-5,rtol=1e-5)
        ids=out['ids'];n=len(row['expected'].encode());length=out['output_bytes'];valid=target['valid_extent_mask'][0]
        error=(raw['final_semantic']-target['semantic_target']).square().mean(-1)[0]
        measured=dict(row,output_bytes=length,early_EOS=bool(out['EOS'] and length<n),same_length_EOS=bool(out['EOS'] and length==n),late_EOS=bool(out['EOS'] and length>n),
            target_byte_argmax_accuracy=float((trace['argmax'][:n].long()==labels.flatten()[:n]).float().mean()),
            latent_valid_RMSE=float(error[valid].mean().sqrt()),latent_tail_RMSE=float(error[~valid].mean().sqrt()))
        groups[(row['phase'],row['split'])].append(measured)
    summaries={}
    for (phase,split),rr in groups.items():
        summaries[phase+'/'+split]=dict(n=len(rr),exact=sum(x['exact'] for x in rr),normalized=sum(x['normalized_exact'] for x in rr),
            expected_relation_present=sum(x['expected_relation_present'] for x in rr),extractable_contradictions=sum(x['contradictory_extractable_fact'] for x in rr),
            EOS=sum(x['EOS'] for x in rr),early_EOS=sum(x['early_EOS'] for x in rr),same_length_EOS=sum(x['same_length_EOS'] for x in rr),late_EOS=sum(x['late_EOS'] for x in rr),
            mean_output_bytes=sum(x['output_bytes'] for x in rr)/len(rr),mean_NLL=sum(x['byte_NLL'] for x in rr)/len(rr),
            mean_byte_argmax_accuracy=sum(x['target_byte_argmax_accuracy'] for x in rr)/len(rr),
            mean_valid_RMSE=sum(x['latent_valid_RMSE'] for x in rr)/len(rr),mean_tail_RMSE=sum(x['latent_tail_RMSE'] for x in rr)/len(rr))
    normal={x['input_id']:x for x in rows if x['phase']=='final'};control={}
    for phase in ('final_zero_KC','final_zero_KP'):
        rr=[x for x in rows if x['phase']==phase]
        control[phase]=dict(rows=len(rr),same_seed_matched=True,mean_NLL_delta=sum(x['byte_NLL']-normal[x['input_id']]['byte_NLL'] for x in rr)/len(rr),
            output_changed=sum(x['output']!=normal[x['input_id']]['output'] for x in rr))
    curves={}
    for name,hh in [('first8',r['history'][:8]),('last8',r['history'][-8:])]:curves[name]={k:sum(h['loss'][k] for h in hh)/len(hh) for k in ('latent','byte_CE','EOS_CE')}
    packet_masks=[t['valid_extent_mask'].sum().item() for t in targets.values()]
    report=dict(status='ROOT_FIRST_CPU_THOUGHT_TRIAL_REVIEWED',actual_attempts192=True,actual_updates192=True,AMPskip0=True,
        old_entire_codec_parameters_and_buffers_verified_frozen=True,frozen_endpoint_digest=frozen,only_new_thought_changed_tensors=len(changed),
        checkpoint_allSHA_safeload_recursive_digest=True,all_Adam_steps192=True,scaler1024_tracker192=True,all_global_training_RNG_draws_reconstructed=True,
        sourceTRAIN_finite_order_and_27136targetbytes_reconstructed=True,raw208_SHA_decode_UTF8_argmax_goldLP_and_sampled258_verified=True,
        NOT_all258_allpositions_saved=True,group_summaries=summaries,matched_memory_controls=control,training_losses=curves,
        target_extent=dict(total_slots=35,valid_min=min(packet_masks),valid_max=max(packet_masks),ignored_tail_slots_per_target=[35-x for x in sorted(set(packet_masks))],
            tail_can_attend_prefix=True,tail_velocity_MSE_was_not_supervised=True),
        interpretation='First192 updates reduce NLL and emitEOS more often, but no96 normal outputs express the required answer. Most early EOS and garbled text; no semantic completion. Original frozen codec/typed relation history did NOT regress.',
        next='Do not blindly repeat192. Inspect train-vs16step rollout and fixed-capacity unsupervised tail/selfattention contract. Tail loss coverage is a concrete hypothesis, not isolated causal proof; no frozenD repair or DEV fitting.',
        calls=result['calls'],actual_new_NN_calls_during_this_review=0,CUDA_initialized=torch.cuda.is_initialized(),all_this_trial_time_call_budget_closed=True,
        Cfree=shutil.disk_usage(ROOT).free,seconds=time.monotonic()-began)
    io.atomic_json(run/'root_saved_review.json',report)
    io.atomic_json(run/'root_result.json',dict(status='FIRST_CPU_THOUGHT192_COMPLETE_EARLY_LEARNING_NOT_SEMANTIC_SUCCESS',review_SHA=support.sha(run/'root_saved_review.json'),
        source_frozen_verified=True,actual_updates=192,new_thought_final_SHA=result['checkpoints'][-1]['SHA'],assessment208=True,all_owned_gone_before_analysis=True,no_automatic_extraSGD=True))
    print(json.dumps(dict(groups=summaries,controls=control,losses=curves,target_extent=report['target_extent'],seconds=report['seconds'],NNcalls0=True)))


if __name__=='__main__':main()
