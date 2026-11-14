"""Saved-only independent fullstate/exposure review, no model forward or CUDA."""
import json,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import text_generation_support_v1 as support
import learning_io_v2 as io

def main():
    torch.set_num_threads(4)
    run=ROOT/'runs/prefix_independent_codec_20261111_v53'
    terminal=io.load_json(run/'final_status.json')
    if terminal['status']!='COMPLETE' or not terminal['exit']['all_owned_processes_gone']:raise ValueError('terminal first')
    plan=io.load_json(run/'plan.json');source=torch.load(plan['source'],weights_only=True,map_location='cpu')
    states={}
    for kind in ('initial','model'):
        directory=run/kind;meta=io.load_json(directory/'latest.json');path=directory/'checkpoints'/meta['file']
        if support.sha(path)!=meta['sha256']:raise ValueError('file SHA')
        state=torch.load(path,weights_only=True,map_location='cpu')
        if support.state_sha(state)!=meta['algorithm_sha256']:raise ValueError('recursive fullstate digest')
        states[kind]=state
    initial,final=states['initial'],states['model']
    for name,tensor in source['model'].items():
        if not torch.equal(tensor,initial['model'][name]):raise ValueError('initial source tensor '+name)
    for group0,group1 in zip(source['optimizer_parameter_names'],initial['optimizer_parameter_names']):
        assert group0==group1
    for key,value in source['optimizer']['state'].items():
        if support.state_sha(value)!=support.state_sha(initial['optimizer']['state'][key]):raise ValueError('source96 moments')
        if int(final['optimizer']['state'][key]['step'])!=int(value['step'])+8192:raise ValueError('all95 next Adam steps')
    assert len(source['optimizer']['state'])==96 and len(final['optimizer']['state'])==96
    assert final['attempts']==final['updates']==final['cursor']==8192 and final['skips']==0
    assert [h['rows'] for h in final['history']]==final['orders']
    for state in (initial,final):
        for tensor in state['model'].values():assert bool(torch.isfinite(tensor).all())
        assert not state['gradients']
        assert all(key in state for key in ('torch_rng','cuda_rng','numpy_rng','python_rng','scaler'))
    assert initial['frozen_sha']==final['frozen_sha']==source['frozen_sha']
    for history in final['history']:
        assert history['optimizer_group_lrs'][:6]==[g['lr'] for g in source['optimizer']['param_groups']]
    assert final['exposure']['target_UTF8_bytes']==40416999
    rederived=[]
    for label in ('baseline','final'):
        assessment=io.load_json((ROOT/'runs/local_three_bank_codec_20261111_v52_r2/final_assessment.json') if label=='baseline' else run/(label+'_assessment.json'));correct=total=0
        for row in assessment['rows']:
            assert len(row['teacher_predicted_ids'])==len(row['target_ids'])==row['bytes']+1
            correct+=sum(a==b for a,b in zip(row['teacher_predicted_ids'],row['target_ids']));total+=len(row['target_ids'])
            if 'output_ids' in row:
                ids=row['output_ids'];raw=bytes(i for i in ids[:ids.index(256) if 256 in ids else len(ids)] if i<256)
                assert raw.decode()==row['output']
                assert row['source'].encode()+b''==bytes(row['target_ids'][:-1])
                assert (row['output']==row['source'])==row['raw_exact']
        assert abs(correct/total-assessment['teacher_accuracy'])<1e-12
        rederived.append(dict(label=label,accuracy=correct/total,symbols=total,
            raw_summary=[dict(id=row['id'],input_bytes=row['bytes'],output_bytes=len(row['output'].encode()),
                EOS=row['EOS'],UTF8=row['UTF8'],exact=row['raw_exact']) for row in assessment['rows'] if 'output' in row]))
    result=dict(status='SAVED_FULLSTATE_REVIEW_PASS',both_SHA_safeload_recursive_digests=True,
        source96_Adam_and_all_parent_tensors_exact=True,final96_Adam=True,all_old_steps_plus8192=True,
        frozen_digest_unchanged=True,RNG_scaler_order_cursor_preserved=True,metrics_independently_rederived=rederived,
        CUDA_initialized=torch.cuda.is_initialized(),extra_model_forward=0,source_SHA_unchanged=support.sha(plan['source'])==plan['source_sha'])
    assert all(h['objective']=='half_clean_half_all_previous_bytes_masked' for h in final['history'])
    rows=io.load_json(run/'final_assessment.json')['rows']
    total=sum(row['bytes']+1 for row in rows)
    result['reported_memory_only_NLL_weighted']=sum(row['memory_only_NLL']*(row['bytes']+1) for row in rows)/total
    result['reported_memory_only_accuracy_weighted']=sum(row['memory_only_accuracy']*(row['bytes']+1) for row in rows)/total
    result['memory_only_argmax_saved']=False
    ablations=io.load_json(run/'memory_ablations.json')
    weight=sum(row['bytes']+1 for row in ablations)
    result['weighted_memory_NLL_delta']={key:sum((row['disabled_NLL'][key]-row['normal_NLL'])*(row['bytes']+1) for row in ablations)/weight for key in ('persistent','contextual','exact')}
    result['trained_local_bank_gains_tanh']=(128*final['model']['memory_codec.local_bank_gains']).tanh().tolist()
    io.atomic_json(run/'root_saved_review.json',result);print(json.dumps(result))
if __name__=='__main__':main()
