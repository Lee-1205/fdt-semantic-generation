"""Saved-only independent full TRAIN134/paired-epsilon review, no NN/update."""
from pathlib import Path
import collections,hashlib,json,sys,time
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import torch
import learning_io_v2 as io
from review_pause_20261114_cpu import independent_decode,validate_trace,sha


def main():
    began=time.monotonic();torch.set_num_threads(2);out=P/'research/artifacts/answer_meaning_v111_fullTRAIN_pair_readonly_20261114_v1';run=P/'runs/answer_meaning_20261114_v111'
    terminal=io.load_json(out/'ROOT_terminal_native.json');state=io.load_json(out/'final_status.json');result=io.load_json(out/'result.json')
    assert terminal['PythonCount']==0 and terminal['All4OwnedProxyGone'] and terminal['BEFORE_analysis']
    assert state['status']=='COMPLETE' and state['exit']['all_owned_processes_gone'] and result['optimizer_steps']==0
    assert result['modelcalls']==dict(denoise=2144,readout=134,encode=0)
    p=io.load_json(run/'plan.json');data=Path(p['data']);alltrain={r['id']:r for r in io.load_json(data/'train_targets.json')}
    features={r['id']:r for r in io.load_json(data/'train_features.json')}
    index=io.load_json(Path(p['cache'])/'target_index_loss_only.json');packets={r['answer_SHA']:torch.load(r['target']['path'],weights_only=True,map_location='cpu') for r in index}
    source_index={r['id']:r for vv in io.load_json(Path(p['cache'])/'input_index.json').values() for r in vv}
    rows=io.load_json(out/'assessment.json');assert len(rows)==134;by_pair=collections.defaultdict(list);raws={};checked=[]
    for row in rows:
        assert sha(row['raw']['path'])==row['raw']['SHA'];raw=torch.load(row['raw']['path'],weights_only=True,map_location='cpu');raws[row['input_id']]=raw
        text,eos,n=independent_decode(raw['trace']);check=validate_trace(raw['trace'],row['expected'])
        assert text==row['output'] and eos==row['EOS'] and abs(check['NLL']-row['byte_NLL'])<1e-6
        target=packets[hashlib.sha256(row['expected'].encode()).hexdigest()];mask=target['valid_extent_mask'];err=(raw['final_semantic']-target['semantic_target']).square().mean(-1)
        assert abs(float((err*mask).sum()/mask.sum())-row['body_MSE'])<1e-6
        z=raw['trace']['sample_full_logits'].reshape(-1,258);bs=row['expected'].encode();gold=torch.tensor(list(bs)+[256])
        assert len(bs)<32 and abs(float((-z.log_softmax(-1)[torch.arange(len(gold)),gold]).mean())-row['byte_NLL'])<2e-6
        argmax=z.argmax(-1);role={}
        if row['label']!='unknown':
            obj=17 if row['label']=='greater' else 18;end=11 if row['label']=='greater' else 12
            role=dict(subject_correct=int(argmax[0])==bs[0],object_correct=int(argmax[obj])==bs[obj],relation_word_exact=argmax[5:end].tolist()==list(bs[5:end]))
        copied=dict(row,source_prompt=features[row['input_id']]['prompt'],length=n,exact_prefix=text.startswith(row['expected']),
            early_EOS=eos and n<len(bs),late_EOS=eos and n>len(bs),byte_accuracy=check['argmax_byte_accuracy'],field_scores=role)
        by_pair[row['pair_key']].append(copied);checked.append(copied)
    pair_results=[]
    for key,rr in by_pair.items():
        if len(rr)!=2:continue
        a,b=rr;assert a['noise_seed']==b['noise_seed'];za=raws[a['input_id']]['final_semantic'];zb=raws[b['input_id']]['final_semantic']
        ta=packets[hashlib.sha256(a['expected'].encode()).hexdigest()]['semantic_target'];tb=packets[hashlib.sha256(b['expected'].encode()).hexdigest()]['semantic_target']
        generated=float((za[:,:4]-zb[:,:4]).square().mean().sqrt());desired=float((ta[:,:4]-tb[:,:4]).square().mean().sqrt())
        sa=torch.load(source_index[a['input_id']]['path'],map_location='cpu',weights_only=True)['source'];sb=torch.load(source_index[b['input_id']]['path'],map_location='cpu',weights_only=True)['source']
        source_rms=float((sa-sb).square().mean().sqrt()) if sa.shape==sb.shape else None
        pair_results.append(dict(pair_key=key,label_type='unknown' if a['label']=='unknown' else 'positive',ids=[a['input_id'],b['input_id']],
            same_noise_seed=True,source_shapes=[list(sa.shape),list(sb.shape)],source_difference_RMS=source_rms,
            generated_first4slot_difference_RMS=generated,desired_answer_code_difference_RMS=desired,
            response_ratio_to_task_targets=generated/desired if desired>0 else None,outputs_identical=a['output']==b['output'],
            both_partial_grammar_correct=a['partial_grammar_semantic_match'] and b['partial_grammar_semantic_match'],
            note='Positive reverse answers can express the SAME canonical fact; ratio measures query-role/target-code response, not opposite truth values.'))
    positive=[r for r in checked if r['label']!='unknown'];unknown=[r for r in checked if r['label']=='unknown'];pp=[r for r in pair_results if r['label_type']=='positive'];up=[r for r in pair_results if r['label_type']=='unknown']
    assert len(positive)==80 and len(unknown)==54 and len(pp)==32 and len(up)==26
    median=lambda values:float(torch.tensor(values,dtype=torch.float64).median())
    summary=dict(status='ROOT_TRAIN134_COMMONPAIR_SAVED_REVIEWED',rows134_allSHA_decode_NLL_bodyMSE_full_answer_logits_verified=True,
        model_forwards0=True,optimizer_steps0=True,positive80_partial_grammar_correct=sum(r['partial_grammar_semantic_match'] for r in positive),
        unknown54_weak_substring_matches=sum(r['partial_grammar_semantic_match'] for r in unknown),
        exact=sum(r['exact'] for r in checked),correct_prefix=sum(r['exact_prefix'] for r in checked),
        no_EOS=sum(not r['EOS'] for r in checked),early_EOS=sum(r['early_EOS'] for r in checked),late_EOS=sum(r['late_EOS'] for r in checked),
        mean_NLL=sum(r['byte_NLL'] for r in checked)/134,mean_body_MSE=sum(r['body_MSE'] for r in checked)/134,mean_tail_MSE=sum(r['tail_MSE'] for r in checked)/134,
        positive_subject_argmax_correct=sum(r['field_scores']['subject_correct'] for r in positive),positive_object_argmax_correct=sum(r['field_scores']['object_correct'] for r in positive),
        positive_relation_word_argmax_exact=sum(r['field_scores']['relation_word_exact'] for r in positive),
        actual_positive_pairs32_common_epsilon=True,actual_unknown_pairs26_common_epsilon=True,
        positive_pairs_both_partial_grammar_correct=sum(r['both_partial_grammar_correct'] for r in pp),unknown_pairs_both_weak_substring=sum(r['both_partial_grammar_correct'] for r in up),
        positive_pair_response=dict(median_generated_first4_difference=median([r['generated_first4slot_difference_RMS'] for r in pp]),
            median_desired_answer_code_difference=median([r['desired_answer_code_difference_RMS'] for r in pp]),
            median_response_ratio=median([r['response_ratio_to_task_targets'] for r in pp]),identical_text_pairs=sum(r['outputs_identical'] for r in pp)),
        pair_response_NOT_independent_memory_causality_or_semantic_accuracy=True,elapsed=time.monotonic()-began)
    io.atomic_json(out/'root_saved_review.json',summary);io.atomic_json(out/'root_rows134.json',checked);io.atomic_json(out/'root_actual58_pair_response.json',pair_results)
    print(json.dumps(summary))


if __name__=='__main__':main()
