"""Independent saved-only review. No E/R/D construction/forward or optimizer.
All decoding/logprob/field/mean/prototype calculations use stored tensors.
Prototype comparison is an offline diagnostic, NEVER inference or accuracy.
"""
import collections,hashlib,json,math,re,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'scripts'),str(P),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from review_pause_20261114_cpu import independent_decode,validate_trace,sha


def canon(answer):
    m=re.search(r'([A-H]) is (larger|smaller) than ([A-H])',answer)
    return None if m is None else ((m[1],m[3]) if m[2]=='larger' else (m[3],m[1]))


def main():
    began=time.monotonic();torch.set_num_threads(2);run=P/'runs/answer_meaning_20261114_v111'
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json');result=io.load_json(run/'result.json');p=io.load_json(run/'plan.json')
    assert terminal['PythonCount']==0 and terminal['All4OwnedProxyGone'] and terminal['BEFORE_scientific_analysis']
    assert owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
    assert result['attempts']==result['updates']==268 and result['AMP_skips']==0 and result['calls']==dict(denoise=40032,readout=224,encode=0)
    plan_sha=sha(run/'plan.json');states=[]
    for ref in result['checkpoints']:
        assert sha(ref['path'])==ref['SHA'] and Path(ref['path']).stat().st_size==ref['bytes']
        value=torch.load(ref['path'],map_location='cpu',weights_only=True)
        assert support.state_sha(value)==ref['algorithm_SHA'] and value['plan_SHA']==plan_sha
        assert value['schema']=='conditional_answer_complete_fullstate_v111';states.append(value)
    initial,middle,final=states;b=initial['engine'];f=final['engine'];r=f['runtime']
    assert [v['engine']['runtime']['updates'] for v in states]==[0,134,268]
    assert f['schema']=='conditional_answer_meaning_fullstate_v111' and f['target_is']=='E(answer)_NOT_E(input)' and not f['language_losses_used']
    assert r['phase']=='DONE' and r['cursor']==2144 and len(r['history'])==268 and r['exposure_bytes']==37088
    assert r['calls']==dict(denoise=36448,readout=0) and b['optimizer']['state']=={}
    assert all(int(v['step'])==268 for v in f['optimizer']['state'].values())
    assert f['scaler']['scale']==1024 and f['scaler']['_growth_tracker']==268 and all(g is None for g in f['gradients'].values())
    selected=io.load_json(Path(p['answer_contract'])/'train_conditions.json');ids={x['id'] for x in selected}
    count=collections.Counter(f['orders']);assert len(count)==134 and set(count.values())=={16} and set(count)==ids
    assert [i for h in r['history'] for i in h['source_ids']]==f['orders'] and all(h['rows']==8 and not h['skip'] and h['D_during_training'] is False for h in r['history'])
    parent=torch.load(p['warm']['path'],map_location='cpu',weights_only=True)
    assert sha(p['warm']['path'])==p['warm']['SHA'] and support.state_sha(parent)==p['warm']['algorithm_SHA']
    assert all(torch.equal(v,parent['engine']['thought'][k]) for k,v in b['thought'].items());del parent
    gen=torch.Generator().set_state(b['ALL_RNG']['torch'])
    for _ in range(2144):torch.randn(1,35,512,generator=gen)
    assert torch.equal(gen.get_state(),f['ALL_RNG']['torch']) and b['ALL_RNG']['python']==f['ALL_RNG']['python'] and b['ALL_RNG']['numpy']==f['ALL_RNG']['numpy']
    assert all(s['engine']['ALL_RNG']['CUDA']==[] for s in states)
    changed=[k for k,v in b['thought'].items() if not torch.equal(v,f['thought'][k])]
    assert len(changed)==64
    from research import linked_lastlinear_runtime_v79 as linked
    from review_memory_thought_v105_cpu import endpoint_digest
    choice=io.load_json(p['selection']);frozen=linked.load_verified(choice['selected_runtime'],choice['selected_SHA'],P,support.state_sha)
    assert support.state_sha(frozen['model'])==choice['source_model_SHA']==result['frozen_codec_SHA_before_after']
    frozen_digest=endpoint_digest(frozen['model']);assert all(s['engine']['frozen_endpoint_digest']==frozen_digest for s in states)
    del frozen
    data=Path(p['data']);original={x['id']:x for x in io.load_json(data/'train_targets.json')}
    train=[original[i] for i in sorted(ids)];assert len({x['answer'] for x in train})==27
    index=io.load_json(Path(p['cache'])/'target_index_loss_only.json')
    packets={x['answer_SHA']:torch.load(x['target']['path'],map_location='cpu',weights_only=True) for x in index}
    proto_keys=[x['answer_SHA'] for x in index];proto_fact=[canon(x['expected']) for x in index]
    proto=torch.cat([packets[k]['semantic_target'] for k in proto_keys]).float()
    numerator=torch.zeros(35,512,dtype=torch.float64);denominator=torch.zeros(35,1,dtype=torch.float64)
    for row in train:
        packet=packets[hashlib.sha256(row['answer'].encode()).hexdigest()];mask=packet['valid_extent_mask'][0]
        w=mask.double()/mask.sum()+.05*(~mask).double()/(~mask).sum()
        numerator+=w[:,None]*packet['semantic_target'][0].double();denominator+=w[:,None]
    constant=(numerator/denominator).float()[None]
    allrows=io.load_json(run/'assessment.json');assert len(allrows)==224;groups=collections.defaultdict(list);saved=[]
    for row in allrows:
        ref=row['raw'];assert sha(ref['path'])==ref['SHA'];raw=torch.load(ref['path'],map_location='cpu',weights_only=True)
        assert raw['task_is_answer_NOT_source_reconstruction'] and raw['no_target_in_inference']
        text,eos,length=independent_decode(raw['trace']);assert text==row['output'] and eos==row['EOS']
        checked=validate_trace(raw['trace'],row['expected']);assert abs(checked['NLL']-row['byte_NLL'])<1e-6
        packet=packets[hashlib.sha256(row['expected'].encode()).hexdigest()];mask=packet['valid_extent_mask'][0];z=raw['final_semantic']
        assert torch.isfinite(z).all() and z.shape==(1,35,512)
        error=(z-packet['semantic_target']).square().mean(-1)[0];body=float(error[mask].mean());tail=float(error[~mask].mean())
        assert abs(body-row['body_MSE'])<1e-6 and abs(tail-row['tail_MSE'])<1e-6
        mean_error=(constant-packet['semantic_target']).square().mean(-1)[0]
        # All target bytes/EOS are within32 symbols, so both saved full patches
        # cover ALL answer scores (not arbitrary later generated tail scores).
        full=raw['trace']['sample_full_logits'];assert raw['trace']['sample_patch_indices'].tolist()==[0,1]
        full=full.reshape(-1,258);bs=row['expected'].encode();gold=torch.tensor(list(bs)+[256])
        full_nll=float((-full.log_softmax(-1)[torch.arange(len(gold)),gold]).mean());assert abs(full_nll-row['byte_NLL'])<2e-6
        argmax=full.argmax(-1);role={}
        if row['label']!='unknown':
            obj=17 if row['label']=='greater' else 18;end=11 if row['label']=='greater' else 12
            role=dict(subject_correct=int(argmax[0])==bs[0],object_correct=int(argmax[obj])==bs[obj],
                relation_word_exact=argmax[5:end].tolist()==list(bs[5:end]),
                scaffold_correct=sum(int(argmax[j])==bs[j] for j in (1,2,3,4)),scaffold_total=4)
        dist=(proto[:,:4]-z[:,:4]).square().mean((1,2));nearest=int(dist.argmin());fact=canon(row['expected'])
        opposite=None
        if fact is not None:
            okay=[i for i,c in enumerate(proto_fact) if c==fact];wrong=[i for i,c in enumerate(proto_fact) if c==(fact[1],fact[0])]
            if okay and wrong:opposite=dict(correct_distance=float(dist[okay].min()),opposite_distance=float(dist[wrong].min()),closer_to_correct=bool(dist[okay].min()<dist[wrong].min()))
        actual_ids=raw['trace']['argmax'][:len(bs)].tolist()
        sample=dict(row,length=length,early_EOS=eos and length<len(bs),late_EOS=eos and length>len(bs),
            exact_prefix=text.startswith(row['expected']),extra_bytes=max(0,length-len(bs)),
            byte_accuracy=checked['argmax_byte_accuracy'],body_MSE_rebuilt=body,tail_MSE_rebuilt=tail,
            constant_TRAIN_body_MSE=float(mean_error[mask].mean()),constant_TRAIN_tail_MSE=float(mean_error[~mask].mean()),
            latent_variance=float(z.var()),field_scores=role,
            offline_prototype_NOT_inference=dict(nearest_answer=index[nearest]['expected'],nearest_canonical_matches=proto_fact[nearest]==fact,
                all_first4_MSE=float(dist[nearest]),opposite_pair_only=opposite,NOT_semantic_accuracy=True),
            expected_byte_argmax=actual_ids)
        groups[row['phase']+'/'+row['split']].append(sample);saved.append(sample)
    summaries={}
    for key,rr in groups.items():
        positive=[x for x in rr if x['label']!='unknown'];unknown=[x for x in rr if x['label']=='unknown']
        summaries[key]=dict(rows=len(rr),positive_rows=len(positive),unknown_rows=len(unknown),
            exact=sum(x['exact'] for x in rr),prefix=sum(x['exact_prefix'] for x in rr),
            partial_grammar_matches=sum(x['partial_grammar_semantic_match'] for x in rr),
            partial_grammar_positive=sum(x['partial_grammar_semantic_match'] for x in positive),
            EOS=sum(x['EOS'] for x in rr),no_EOS=sum(not x['EOS'] for x in rr),early_EOS=sum(x['early_EOS'] for x in rr),late_EOS=sum(x['late_EOS'] for x in rr),
            mean_NLL=sum(x['byte_NLL'] for x in rr)/len(rr),body_MSE=sum(x['body_MSE_rebuilt'] for x in rr)/len(rr),tail_MSE=sum(x['tail_MSE_rebuilt'] for x in rr)/len(rr),
            constant_TRAIN_body_MSE=sum(x['constant_TRAIN_body_MSE'] for x in rr)/len(rr),constant_TRAIN_tail_MSE=sum(x['constant_TRAIN_tail_MSE'] for x in rr)/len(rr),
            byte_accuracy=sum(x['byte_accuracy'] for x in rr)/len(rr),
            subject_correct=sum(x['field_scores']['subject_correct'] for x in positive),object_correct=sum(x['field_scores']['object_correct'] for x in positive),
            relation_word_exact=sum(x['field_scores']['relation_word_exact'] for x in positive),
            scaffold_correct=sum(x['field_scores']['scaffold_correct'] for x in positive),scaffold_total=4*len(positive),
            offline_nearest_canonical_NOT_accuracy=sum(x['offline_prototype_NOT_inference']['nearest_canonical_matches'] for x in rr),
            latent_variance_mean=sum(x['latent_variance'] for x in rr)/len(rr))
    curve={}
    for name,hh in [('first8',r['history'][:8]),('last8',r['history'][-8:])]:
        curve[name]=dict(loss={k:sum(h['loss'][k] for h in hh)/len(hh) for k in ('latent','byte_CE','EOS_CE')},
            actual_trajectory={k:sum(h['runtime_loss'][k] for h in hh)/len(hh) for k in ('runtime_body_MSE','runtime_tail_MSE','aux_VP')},
            mean_preclip_gradient_norm=sum(h['grad_norm'] for h in hh)/len(hh))
    w=f['thought']['velocity.weight'].double();gamma=f['thought']['final_norm.weight'].double();wg=w*gamma[None];effective=wg-wg.sum(1,keepdim=True)/wg.shape[1]
    rank=int(torch.linalg.matrix_rank(effective));assert rank==512 and w.shape==(512,520)
    io.atomic_json(run/'root_saved_rows224.json',saved)
    review=dict(status='ROOT_CONDITIONAL_EY_V111_FULLSTATE_TRACE_REVIEWED',checkpoint3_allSHA_safeload_recursive=True,
        warm_V110_ALLtheta_exact_freshAdam=True,only64_thought_tensors_changed=True,all_Adam_steps268=True,scaler1024_tracker268=True,
        all2144_noise_draws_exact=True,selected134_each16_finite=True,TRAIN_exposure37088=True,training_D_calls0_verified=True,
        whole_codec_frozen_digest_verified=frozen_digest,all224_traces_independent_decode_NLL_MSE=True,
        all_answer_positions_FULL258_verified=True,NOT_all_generated_tail_FULL258=True,groups=summaries,training_curve=curve,
        effective_velocity_rank512=True,mean_reference_fit_TRAIN134_ONLY=True,prototype_algebra_diagnostic_only_NOT_inference=True,
        scientific_success_NOT_inferred_from_loss=True,additional_E_R_D_forwards=0,additional_optimizer_steps=0,CUDA_initialized=torch.cuda.is_initialized(),
        final_checkpoint_phase='TrainingDONE before finalassessment; final224assessment storedseparately, not a claimed postevalfullstate',
        elapsed=time.monotonic()-began,trial_budget_closed=True,unused744calls_NO_reuse=True)
    io.atomic_json(run/'root_saved_review.json',review)
    print(json.dumps(dict(groups=summaries,curve=curve,rank=rank,seconds=review['elapsed'])))


if __name__=='__main__':main()
