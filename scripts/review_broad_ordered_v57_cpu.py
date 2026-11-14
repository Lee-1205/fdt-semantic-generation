"""Saved-only independent trajectory, whole-document exposure and raw review."""
import collections,difflib,json,math,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import text_generation_support_v1 as support
import learning_io_v2 as io
from research.codec_commit_stop_v56 import reconstruct_commit_logits

def main():
    torch.set_num_threads(4);run=ROOT/'runs/broad_ordered_20261111_v57';p=io.load_json(run/'plan.json')
    terminal=io.load_json(run/'final_status.json');native=io.load_json(run/'root_terminal_native.json')
    assert terminal['status']=='COMPLETE' and terminal['exit']['all_owned_processes_gone'] and native['fresh_python_count']==0
    assert support.sha(p['source'])==p['source_sha'] and support.sha(p['head_source'])==p['head_sha']
    parent=torch.load(p['head_source'],weights_only=True,map_location='cpu');states={}
    for step in (0,1,2048,4096,6144,8192):
        path=run/'model/checkpoints'/f'state_{step:04d}.pt';meta=io.load_json(path.with_suffix('.json'))
        assert support.sha(path)==meta['sha256'];s=torch.load(path,weights_only=True,map_location='cpu')
        assert support.state_sha(s)==meta['algorithm_sha256'];states[step]=s
        assert s['attempts']==s['updates']==s['cursor']==step and not s['skips']
        assert len(s['optimizer']['state'])==8 and all(int(v['step'])==7248+step for v in s['optimizer']['state'].values())
        if step:
            factor=.1+.9*(1+math.cos(math.pi*(step-1)/8192))/2
            for group,base in zip(s['optimizer']['param_groups'],(1e-4,3e-4)):assert abs(group['lr']-base*factor)<1e-14
    initial,final=states[0],states[8192]
    for key in ('head','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):
        assert support.state_sha({key:initial[key]})==support.state_sha({key:parent[key]})
    for state in states.values():
        assert not state['gradients'] and not state['cache_refs']
        for key in ('torch_rng','cuda_rng','python_rng','numpy_rng'):
            assert support.state_sha({key:state[key]})==support.state_sha({key:initial[key]})
    assert final['scaler']['scale']==131072 and final['scaler']['_growth_tracker']==1440
    train=io.load_json(Path(p['data'])/'train.json');dev=io.load_json(Path(p['data'])/'dev.json')
    order=list(range(8192));random.Random(57111).shuffle(order)
    assert final['orders']==[[i] for i in order] and [h['rows'] for h in final['history']]==final['orders']
    assert [h['source_id'] for h in final['history']]==[train[i]['id'] for i in order]
    assert len({r['id'] for r in train})==8192
    exposure=dict(selected_patches=sum((r['actual_bytes']+16)//16 for r in train),active_symbols=sum(r['actual_bytes']+1 for r in train),
        target_bytes=sum(r['actual_bytes'] for r in train),EOS=8192)
    assert exposure==final['exposure'] and exposure['target_bytes']==40416999
    saved=io.load_json(run/'same_input_reconstruction.json');result=io.load_json(run/'result.json')
    totals=collections.Counter();errors=collections.Counter();scores=[];readings=[];NLL=correct=count=0;first_error=first_total=0
    def cls(i):return 'upper' if 65<=i<=90 else 'lower' if 97<=i<=122 else 'digit' if 48<=i<=57 else 'other'
    for i,row in enumerate(dev):
        payload=torch.load(run/f'raw_dev_{i:02d}.pt',weights_only=True,map_location='cpu')
        logits,y=payload['logits'],payload['targets'];raw=row['prompt'].encode();expected=torch.full_like(y.flatten(),257)
        expected[:len(raw)]=torch.tensor(list(raw));expected[len(raw)]=256;assert torch.equal(y.flatten(),expected)
        active=y!=257;body=y<256;EOS=y==256;pred=logits.argmax(-1)
        ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),y.masked_fill(~active,-100),ignore_index=-100,reduction='sum')
        score=dict(id=row['id'],NLL_sum=float(ce),correct=int(((pred==y)&active).sum()),symbols=int(active.sum()),
            errors=int(((pred!=y)&body).sum()),EOS_false_positives=int(((pred==256)&body).sum()))
        for key in ('NLL_sum','correct','symbols'):assert abs(score[key]-result['metrics']['scores'][i][key])<1e-5
        NLL+=score['NLL_sum'];correct+=score['correct'];count+=score['symbols'];scores.append(score)
        for a,b in zip(y[body].tolist(),pred[body].tolist()):totals[cls(a)]+=1;errors[cls(a)]+=int(a!=b)
        first_total+=int(body[:2].sum());first_error+=int(((pred[:2]!=y[:2])&body[:2]).sum())
        decoded=reconstruct_commit_logits(logits);out=decoded['text'];source=row['prompt']
        if i in p['raw_indices']:
            original=saved[p['raw_indices'].index(i)]
            assert original['output']==out and original['ids']==decoded['ids'] and original['source']==source
        a=collections.Counter(re.findall(r'\w+',source.lower()));b=collections.Counter(re.findall(r'\w+',out.lower()))
        diff=[dict(at=x,source=source[x:xx],output=out[z:zz],tag=t,
            source_context=source[max(0,x-35):min(len(source),xx+35)]) for t,x,xx,z,zz in difflib.SequenceMatcher(None,source,out,autojunk=False).get_opcodes() if t!='equal']
        readings.append(dict(id=row['id'],source=source,output=out,source_bytes=len(raw),output_bytes=len(out.encode()),
            exact=source==out,EOS=decoded['EOS'],UTF8=decoded['UTF8'],character_similarity=difflib.SequenceMatcher(None,source,out,autojunk=False).ratio(),
            word_overlap_F1=2*sum((a&b).values())/max(1,sum(a.values())+sum(b.values())),changes=diff))
    assert abs(NLL/count-result['metrics']['NLL'])<1e-8 and abs(correct/count-result['metrics']['accuracy'])<1e-8
    report=dict(status='INDEPENDENT_V57_SAVED_REVIEW_PASS',all6_checkpoint_SHA_safeload_digest=True,
        initial8Adam_scaler_ALLRNG_exactV56=True,all8_Adam_steps7248to15440=True,all_checkpoint_LRs_independent_cosine=True,
        scaler131072_tracker1440=True,all_stored_RNG_unchanged=True,all8192source_ids_once_order_history_exposure_verified=True,
        source_model_SHA_unchanged=True,no_growing_cache=True,raw16_scores_targets_reconstructed=True,
        accuracy=correct/count,NLL=NLL/count,class_errors=dict(errors),class_totals=dict(totals),
        first32=dict(errors=first_error,total=first_total),raw4_exact=sum(readings[i]['exact'] for i in p['raw_indices']),
        all16_saved_logits_CPU_decode_exact=sum(r['exact'] for r in readings),scores=scores,
        extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
        limitation='same observed DEV; all16 CPU decoding uses already saved logits, no new model evaluation or blind benchmark')
    io.atomic_json(run/'root_saved_review.json',report);io.atomic_json(run/'root_reading_similarity.json',readings)
    print(json.dumps(report))
if __name__=='__main__':main()
