"""Independent saved tensors only: no E/R/D objects/forwards or optimizer."""
import argparse,collections,hashlib,json,math,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from review_pause_20261114_cpu import independent_decode,validate_trace,sha


def main(run):
    began=time.monotonic();torch.set_num_threads(2)
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert terminal['PythonCount']==0 and terminal['All4OwnedProxyGone'] and terminal['BEFORE_scientific_analysis']
    assert owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
    p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json');planSHA=sha(run/'plan.json')
    states=[]
    for ref in result['checkpoints']:
        assert sha(ref['path'])==ref['SHA'] and Path(ref['path']).stat().st_size==ref['bytes']
        value=torch.load(ref['path'],map_location='cpu',weights_only=True)
        assert support.state_sha(value)==ref['algorithm_SHA'] and value['plan_SHA']==planSHA;states.append(value)
    initial,middle,final=states;b=initial['engine'];f=final['engine'];r=f['runtime']
    assert [v['engine']['runtime']['updates'] for v in states]==[0,134,268]
    assert r['phase']=='DONE' and r['cursor']==2144 and r['exposure_bytes']==37088 and r['updates']==r['attempts']==268
    assert b['optimizer']['state']=={} and all(int(v['step'])==268 for v in f['optimizer']['state'].values())
    assert f['scaler']['scale']==1024 and f['scaler']['_growth_tracker']==268 and all(v is None for v in f['gradients'].values())
    assert r['AMP_skips']==0
    count=collections.Counter(f['orders']);assert len(count)==134 and set(count.values())=={16}
    assert [i for h in r['history'] for i in h['source_ids']]==f['orders']
    assert all(h['rows']==8 and not h['skip'] for h in r['history']) and sum(h['actual_pair_rows'] for h in r['history'])==1856
    units=f['source_metadata']['pair_contrast_v112']['units'];assert len(units)==1072
    gen=torch.Generator().set_state(b['ALL_RNG']['torch']);draws=0
    for u in units:
        for _ in range(1 if u['real_pair'] else 2):torch.randn(1,35,512,generator=gen);draws+=1
    assert draws==1216==sum(h['unit_RNG_draws'] for h in r['history']) and torch.equal(gen.get_state(),f['ALL_RNG']['torch'])
    assert b['ALL_RNG']['python']==f['ALL_RNG']['python'] and b['ALL_RNG']['numpy']==f['ALL_RNG']['numpy'] and f['ALL_RNG']['CUDA']==[]
    parent=torch.load(p['warm']['path'],map_location='cpu',weights_only=True)
    assert sha(p['warm']['path'])==p['warm']['SHA'] and support.state_sha(parent)==p['warm']['algorithm_SHA']
    assert all(torch.equal(v,parent['engine']['thought'][k]) for k,v in b['thought'].items());del parent
    changed=[k for k,v in b['thought'].items() if not torch.equal(v,f['thought'][k])];assert len(changed)==64
    assert all(torch.isfinite(v).all() for v in f['thought'].values())
    from research import linked_lastlinear_runtime_v79 as linked
    from review_memory_thought_v105_cpu import endpoint_digest
    choice=io.load_json(p['selection']);frozen=linked.load_verified(choice['selected_runtime'],choice['selected_SHA'],P,support.state_sha)
    frozenSHA=support.state_sha(frozen['model']);assert frozenSHA==choice['source_model_SHA']==result['frozen_codec_SHA_before_after']
    digest=endpoint_digest(frozen['model']);assert all(s['engine']['frozen_endpoint_digest']==digest for s in states);del frozen
    index=io.load_json(Path(p['cache'])/'target_index_loss_only.json')
    packets={x['answer_SHA']:torch.load(x['target']['path'],map_location='cpu',weights_only=True) for x in index}
    original={x['id']:x for x in io.load_json(Path(p['data'])/'train_targets.json')}
    for split in ('dev_pair','dev_surface','dev_position'):
        original.update({x['id']:x for x in io.load_json(Path(p['data'])/(split+'_targets.json'))})
    rows=io.load_json(run/'assessment.json');assert len(rows)==460;groups=collections.defaultdict(list);saved=[];latent={};NLL64errors=[]
    for row in rows:
        ref=row['raw'];assert sha(ref['path'])==ref['SHA'];raw=torch.load(ref['path'],map_location='cpu',weights_only=True)
        assert raw['no_target_in_inference'];text,eos,length=independent_decode(raw['trace'])
        assert text==row['output'] and eos==row['EOS'];checked=validate_trace(raw['trace'],row['expected'])
        assert abs(checked['NLL']-row['byte_NLL'])<2e-6
        packet=packets[hashlib.sha256(row['expected'].encode()).hexdigest()];mask=packet['valid_extent_mask'][0]
        z=raw['final_semantic'];err=(z-packet['semantic_target']).square().mean(-1)[0]
        assert torch.isfinite(z).all() and z.shape==(1,35,512) and abs(float(err[mask].mean())-row['body_MSE'])<1e-6
        assert abs(float(err[~mask].mean())-row['tail_MSE'])<1e-6
        full=raw['trace']['sample_full_logits'].reshape(-1,258);bs=row['expected'].encode();gold=torch.tensor(list(bs)+[256])
        selected=full[:len(gold)]
        # The saved trace uses gather-logsumexp, not the algebraically equal
        # but differently rounded log_softmax. First reproduce it BITWISE.
        exact_lp=selected.gather(-1,gold[:,None]).squeeze(-1)-selected.logsumexp(-1)
        assert torch.equal(exact_lp,raw['trace']['gold_log_probability'][:len(gold)])
        nll64=float(-selected.double().log_softmax(-1)[torch.arange(len(gold)),gold].mean())
        delta=abs(nll64-row['byte_NLL']);NLL64errors.append(delta)
        bound=8*torch.finfo(torch.float32).eps*max(1.,float(selected.abs().max()),abs(nll64))
        assert delta<=bound
        argmax=full.argmax(-1);field={}
        if row['label']!='unknown':
            obj=17 if row['label']=='greater' else 18;end=11 if row['label']=='greater' else 12
            field=dict(subject_correct=int(argmax[0])==bs[0],object_correct=int(argmax[obj])==bs[obj],
                relation_word_exact=argmax[5:end].tolist()==list(bs[5:end]),scaffold_correct=sum(int(argmax[j])==bs[j] for j in (1,2,3,4)))
        sample=dict(row,length=length,prefix=text.startswith(row['expected']),extra_bytes=max(0,length-len(bs)),
            early_EOS=eos and length<len(bs),late_EOS=eos and length>len(bs),field_scores=field,
            byte_accuracy=checked['argmax_byte_accuracy'],latent_variance=float(z.var()),
            pair_key=original[row['input_id']]['pair_key'])
        saved.append(sample);groups[row['phase']+'/'+row['split']].append(sample)
        latent[row['phase'],row['input_id']]=z
    summaries={}
    for key,rr in groups.items():
        positive=[x for x in rr if x['label']!='unknown'];unknown=[x for x in rr if x['label']=='unknown']
        summaries[key]=dict(rows=len(rr),positive_rows=len(positive),unknown_rows=len(unknown),
            exact=sum(x['exact'] for x in rr),prefix=sum(x['prefix'] for x in rr),
            positive_grammar=sum(x['partial_grammar_semantic_match'] for x in positive),
            unknown_keyword_NOT_accuracy=sum(x['partial_grammar_semantic_match'] for x in unknown),
            EOS=sum(x['EOS'] for x in rr),no_EOS=sum(not x['EOS'] for x in rr),
            early_EOS=sum(x['early_EOS'] for x in rr),late_EOS=sum(x['late_EOS'] for x in rr),
            NLL=sum(x['byte_NLL'] for x in rr)/len(rr),body_MSE=sum(x['body_MSE'] for x in rr)/len(rr),
            tail_MSE=sum(x['tail_MSE'] for x in rr)/len(rr),byte_accuracy=sum(x['byte_accuracy'] for x in rr)/len(rr),
            subject_correct=sum(x['field_scores']['subject_correct'] for x in positive),object_correct=sum(x['field_scores']['object_correct'] for x in positive),
            relation_word_exact=sum(x['field_scores']['relation_word_exact'] for x in positive))
    pair_summary={};pair_rows=[]
    for phase in ('baseline','final'):
        pairs=collections.defaultdict(list)
        for row in groups[phase+'/TRAIN_all']:pairs[row['pair_key']].append(row)
        entries=[]
        for key,rr in pairs.items():
            if len(rr)!=2:continue
            a,c=rr;za,zc=(latent[phase,x['input_id']] for x in rr)
            pa,pc=(packets[hashlib.sha256(x['expected'].encode()).hexdigest()] for x in rr)
            delta=za-zc;target=pa['semantic_target']-pc['semantic_target'];mask=(pa['valid_extent_mask']&pc['valid_extent_mask'])[0]
            first_delta=delta[:,:4];first_target=target[:,:4]
            positive=a['label']!='unknown'
            entry=dict(phase=phase,key=key,positive=positive,ids=[a['input_id'],c['input_id']],
                generated_difference_RMS=float(first_delta.square().mean().sqrt()),
                target_difference_RMS=float(first_target.square().mean().sqrt()),
                body_difference_MSE=float((delta-target).square().mean(-1)[0,mask].mean()),
                outputs_identical=a['output']==c['output'],both_grammar=all(x['partial_grammar_semantic_match'] for x in rr))
            entry['response_ratio_NOT_accuracy']=entry['generated_difference_RMS']/entry['target_difference_RMS'] if positive else None
            if positive:
                dot=float((first_delta*first_target).sum());norm2=float(first_target.square().sum())
                entry['target_direction_projection_ratio_NOT_accuracy']=dot/norm2
                entry['direction_cosine_NOT_accuracy']=dot/float(first_delta.norm()*first_target.norm()).__float__()
            entries.append(entry);pair_rows.append(entry)
        positive=[x for x in entries if x['positive']];assert len(positive)==32 and len(entries)==58
        pair_summary[phase]=dict(positive_pairs=32,positive_both_grammar=sum(x['both_grammar'] for x in positive),
            positive_identical=sum(x['outputs_identical'] for x in positive),
            median_ratio_NOT_accuracy=float(torch.tensor([x['response_ratio_NOT_accuracy'] for x in positive]).quantile(.5)),
            mean_direction_cosine_NOT_accuracy=sum(x['direction_cosine_NOT_accuracy'] for x in positive)/32,
            mean_projection_ratio_NOT_accuracy=sum(x['target_direction_projection_ratio_NOT_accuracy'] for x in positive)/32,
            mean_body_difference_MSE=sum(x['body_difference_MSE'] for x in positive)/32)
    curve={}
    for name,hh in [('first8',r['history'][:8]),('last8',r['history'][-8:])]:
        curve[name]=dict(body=sum(h['runtime_loss']['runtime_body_MSE'] for h in hh)/len(hh),
            byte_CE=sum(h['loss']['byte_CE'] for h in hh)/len(hh),
            combined_loss=sum(h.get('combined_latent_plus_byte_CE',h['loss']['latent']) for h in hh)/len(hh),
            tail=sum(h['runtime_loss']['runtime_tail_MSE'] for h in hh)/len(hh),
            aux=sum(h['runtime_loss']['aux_VP'] for h in hh)/len(hh),
            contrast=sum(h['pair_contrast_MSE'] for h in hh if h['pair_contrast_MSE'] is not None)/sum(h['pair_contrast_MSE'] is not None for h in hh),
            preclip_norm=sum(h['grad_norm'] for h in hh)/len(hh))
    review=dict(status='ROOT_PAIR_REPAIR_SAVED_REVIEWED',all3_fullstates_SHA_safeload_recursive=True,all64_theta_changed=True,
        frozen_endpoint_digest=digest,frozen_codec_SHA=frozenSHA,RNG_draws1216_exact=True,all134_each16=True,
        Adam_step268_scaler_tracker268=True,source_only_raw460_independently_decoded=True,
        all_answer_byte_EOS_full258_verified=True,NOT_all_generated_tail_full258=True,
        all460_answer_logprobs_gather_logsumexp_bitwise=True,FP64_NLL_max_error=max(NLL64errors),
        groups=summaries,pairs=pair_summary,training_curve=curve,training_calls=r['calls'],
        additional_model_calls=0,additional_optimizer_steps=0,CUDA_initialized=torch.cuda.is_initialized(),
        manual_semantic_review_required=True,seconds=time.monotonic()-began)
    io.atomic_json(run/'root_saved_review.json',review);io.atomic_json(run/'root_rows460.json',saved);io.atomic_json(run/'root_pair_response.json',pair_rows)
    print(json.dumps(dict(groups=summaries,pairs=pair_summary,curve=curve,seconds=review['seconds'])))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--run',type=Path,required=True);a=q.parse_args();main(a.run.resolve())
