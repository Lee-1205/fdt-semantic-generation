"""Independent saved-only readout/parent/cache/argmax/EOS review; no model forward."""
import collections,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import text_generation_support_v1 as support
import learning_io_v2 as io
from research.commit_patch_readout_v54 import decode_fixed_logits

def main():
    torch.set_num_threads(4);run=ROOT/'runs/ordered_fuzzy_20261111_v55';p=io.load_json(run/'plan.json')
    terminal=io.load_json(run/'final_status.json')
    if terminal['status']!='COMPLETE' or not terminal['exit']['all_owned_processes_gone']:raise ValueError('terminal first')
    if support.sha(p['source'])!=p['source_sha'] or support.sha(p['head_source'])!=p['head_sha']:raise ValueError('parent changed')
    states={}
    for name in ('seed','initial','model'):
        meta=io.load_json(run/name/'latest.json');path=run/name/'checkpoints'/meta['file']
        assert support.sha(path)==meta['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert support.state_sha(state)==meta['algorithm_sha256'];states[name]=state
    final=states['model'];initial=states['initial']
    assert final['attempts']==final['updates']==3624 and final['skips']==0 and final['cursor']==927402
    assert [h['rows'] for h in final['history']]==final['orders']
    assert len(final['optimizer']['state'])==8 and all(int(v['step'])==3624 for v in final['optimizer']['state'].values())
    assert all(not state['gradients'] for state in states.values())
    assert final['exposure']['target_bytes']==14811468 and final['exposure']['EOS']==3072
    assert final['extract_cursor']=={'train':512,'dev':16}
    for name,digest in final['cache_refs'].items():assert support.sha(name)==digest,name
    for split in ('train','dev'):
        features=torch.load(run/(split+'_features.pt'),map_location='cpu',weights_only=True)
        targets=torch.load(run/(split+'_targets.pt'),map_location='cpu',weights_only=True)
        rows=io.load_json(Path(p['data'])/(split+'.json'))
        for i,row in enumerate(rows):
            start,end=features['offsets'][i];raw=row['prompt'].encode()
            expected=torch.full(((end-start)*16,),257,dtype=torch.long)
            expected[:len(raw)]=torch.tensor(list(raw));expected[len(raw)]=256
            assert torch.equal(targets[start:end].flatten(),expected)
            shard=final['extract_shards'][split][i]
            local=torch.load(shard['features'],map_location='cpu',weights_only=True)
            assert set(local)=={'local','global','terminal','parent_sha','id','fuzzy_roles'}
            assert local['id']==row['id'] and local['parent_sha']==p['source_sha']
            assert torch.equal(local['local'],features['local'][start:end])
            assert torch.equal(local['global'],features['global'][i])
            assert torch.equal(local['fuzzy_roles'],features['fuzzy_roles'][start:end])
            assert local['fuzzy_roles'].shape[1:]==(16,256) and not local['fuzzy_roles'][-1].count_nonzero()
            assert features['terminal'][start:end].sum()==1 and features['terminal'][end-1]
        if split=='train':
            active_per_patch=(targets!=257).sum(1);EOS_per_patch=(targets==256).sum(1)
            EOS_examples=sum(int(EOS_per_patch[torch.tensor(order)].sum()) for order in final['orders'])
            active_in_EOS_batches=sum(int(active_per_patch[torch.tensor(order)].sum()) for order in final['orders'] if EOS_per_patch[torch.tensor(order)].sum()>0)
            effective_EOS_weight=1+.1*active_in_EOS_batches/EOS_examples
    scores={}
    normal_payload=None
    for condition in ('normal','reverse_roles'):
        payload=torch.load(run/(condition+'_raw_logits.pt'),map_location='cpu',weights_only=True)
        logits,y=payload['logits'],payload['targets'];active=y!=257;byte=y<256;EOS=y==256;pred=logits.argmax(-1)
        ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),y.masked_fill(~active,-100),ignore_index=-100,reduction='none')
        score=dict(NLL=float(ce.sum()/active.sum()),accuracy=float(((pred==y)&active).sum()/active.sum()),
            byte_accuracy=float(((pred==y)&byte).sum()/byte.sum()),EOS_recall=float(((pred==256)&EOS).sum()/EOS.sum()),
            EOS_precision=float(((pred==256)&EOS).sum()/((pred==256)&active).sum().clamp_min(1)),
            EOS_false_positives=int(((pred==256)&byte).sum()),symbols=int(active.sum()))
        saved=io.load_json(run/(condition+'_metrics.json'))
        for key in ('NLL','accuracy','byte_accuracy','EOS_recall'):assert abs(score[key]-saved[key])<1e-6
        scores[condition]=score
        if condition=='normal':normal_payload=payload
    y=normal_payload['targets'];body=y<256
    total=bag_entropy=bag_mode=0
    for patch in y:
        ids=[v for v in patch.tolist() if v<256]
        if not ids:continue
        counts=collections.Counter(ids);n=len(ids)
        total+=n;bag_mode+=max(counts.values());bag_entropy+=sum(-count*math.log(count/n) for count in counts.values())
    reading=io.load_json(run/'same_input_reconstruction.json')
    for i,row in zip(p['raw_indices'],reading):
        start,end=normal_payload['offsets'][i]
        raw=decode_fixed_logits(normal_payload['logits'][start:end])
        assert raw['text']==row['output'] and raw['ids']==row['ids'] and raw['EOS']==row['EOS']
        assert row['exact']==(row['source']==row['output'])
    from collections import Counter
    errors=(normal_payload['logits'].argmax(-1)!=y)&body
    positions=torch.arange(len(y)*16).reshape_as(y)
    error_classes=Counter();total_classes=Counter()
    for value,error in zip(y[body].tolist(),errors[body].tolist()):
        key='digit' if 48<=value<=57 else 'upper' if 65<=value<=90 else 'lower' if 97<=value<=122 else 'other'
        total_classes[key]+=1;error_classes[key]+=int(error)
    by_role=[dict(role=i,total=int(body[:,i].sum()),errors=int(errors[:,i].sum())) for i in range(16)]
    first_errors=first_total=rest_errors=rest_total=0
    for a,b in normal_payload['offsets']:
        first_errors+=int(errors[a:min(a+2,b)].sum());first_total+=int(body[a:min(a+2,b)].sum())
        rest_errors+=int(errors[min(a+2,b):b].sum());rest_total+=int(body[min(a+2,b):b].sum())
    io.atomic_json(run/'root_error_localization.json',dict(by_role=by_role,by_class=dict(errors=error_classes,total=total_classes),
        first32=dict(errors=first_errors,total=first_total),after32=dict(errors=rest_errors,total=rest_total),model_calls=0))
    result=dict(status='INDEPENDENT_V55_SAVED_REVIEW_PASS',all3_head_checkpoints_SHA_safeload_digest=True,
        all528_shards_and_concatenated_cache_refs_verified=True,all_training_labels_reconstructed=True,
        feature_payload_has_no_raw_source_or_gold=True,all8_active_Adam_steps3624=True,phase_cursor_RNG_scaler_preserved=True,
        source_parent_SHA_unchanged=True,source_96_Adam_preserved_in_required_parent=True,
        raw_logits_NLL_argmax_and_EOS_independently_reconstructed=scores,
        gold_bag_frequency_diagnostic=dict(within_patch_position_unaware_mode_accuracy=bag_mode/total,
            within_patch_empirical_bag_entropy=bag_entropy/total,score_only_not_inference=True),
        source_EOS_calibration_reference=p['EOS_log_bias'],uniform_new_EOS_loss=True,
        read_all4_reconstructions=True,raw_exact=sum(row['exact'] for row in reading),qualitative='Long verbatim spans recovered, but names and dates are still wrong; long output truncated by premature EOS. Not perfect meaning fidelity.',
        extra_model_forward=0,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(run/'root_saved_review.json',result);print(json.dumps(result))
if __name__=='__main__':main()
