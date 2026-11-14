"""Independent checkpoint trajectory and saved same-input outputs, no NN call."""
import collections,difflib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.codec_commit_stop_v56 import reconstruct_commit_logits

def main():
    torch.set_num_threads(4);run=ROOT/'runs/balanced_fuzzy_20261111_v56';p=io.load_json(run/'plan.json')
    terminal=io.load_json(run/'final_status.json');native=io.load_json(run/'root_terminal_native.json')
    assert terminal['status']=='COMPLETE' and terminal['exit']['all_owned_processes_gone'] and native['fresh_python_count']==0
    parent=torch.load(p['head_source'],weights_only=True,map_location='cpu')
    assert support.sha(p['head_source'])==p['head_sha'] and support.sha(p['source'])==p['source_sha']
    states={}
    for name in ('initial','model'):
        meta=io.load_json(run/name/'latest.json');path=run/name/'checkpoints'/meta['file']
        assert support.sha(path)==meta['sha256'];s=torch.load(path,weights_only=True,map_location='cpu')
        assert support.state_sha(s)==meta['algorithm_sha256'];states[name]=s
    a,b=states['initial'],states['model']
    for key in ('head','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):
        assert support.state_sha({key:parent[key]})==support.state_sha({key:a[key]})
    for key in ('torch_rng','cuda_rng','python_rng','numpy_rng'):
        assert support.state_sha({key:a[key]})==support.state_sha({key:b[key]})
    assert b['scaler']['scale']==8192 and b['scaler']['_growth_tracker']==1248
    assert b['attempts']==b['updates']==3624 and b['cursor']==927402 and not b['skips']
    assert [h['rows'] for h in b['history']]==b['orders'] and len(b['orders'])==3624
    assert len(b['optimizer']['state'])==8 and all(int(s['step'])==7248 for s in b['optimizer']['state'].values())
    assert b['cache_refs']==parent['cache_refs'] and b['extract_cursor']==parent['extract_cursor']
    for name in ('train_features','train_targets','dev_features','dev_targets'):
        assert support.sha(p[name])==p['files'][p[name]]
    assert b['exposure']==dict(selected_patches=927402,active_symbols=14814540,target_bytes=14811468,EOS=3072)
    packet=torch.load(run/'normal_raw_logits.pt',weights_only=True,map_location='cpu')
    logits,y=packet['logits'],packet['targets'];active=y!=257;body=y<256;EOS=y==256;pred=logits.argmax(-1)
    loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),y.masked_fill(~active,-100),ignore_index=-100,reduction='sum')
    score=dict(NLL=float(loss/active.sum()),accuracy=float(((pred==y)&active).sum()/active.sum()),
        byte_accuracy=float(((pred==y)&body).sum()/body.sum()),EOS_recall=float(((pred==256)&EOS).sum()/EOS.sum()),
        EOS_false_positives=int(((pred==256)&body).sum()),symbols=int(active.sum()))
    result=io.load_json(run/'result.json')
    for key in score:assert abs(score[key]-result['metrics'][key])<1e-6
    counts=collections.Counter();errors=collections.Counter();by_class=[]
    def cls(i):return 'digit' if 48<=i<=57 else 'upper' if 65<=i<=90 else 'lower' if 97<=i<=122 else 'other'
    for expected,actual in zip(y[body].tolist(),pred[body].tolist()):
        key=cls(expected);counts[key]+=1;errors[key]+=int(expected!=actual)
    rows=io.load_json(Path(p['data'])/'dev.json');details=[]
    reading=io.load_json(run/'same_input_reconstruction.json')
    for i,saved in zip(p['raw_indices'],reading):
        start,end=packet['offsets'][i];raw=rows[i]['prompt'].encode();expected=torch.full(((end-start)*16,),257,dtype=torch.long)
        expected[:len(raw)]=torch.tensor(list(raw));expected[len(raw)]=256;assert torch.equal(y[start:end].flatten(),expected)
        decoded=reconstruct_commit_logits(logits[start:end]);out=decoded['text'];source=rows[i]['prompt']
        assert saved['source']==source and saved['output']==out and saved['ids']==decoded['ids'] and saved['exact']==(source==out)
        aw=collections.Counter(re.findall(r'\w+',source.lower()));bw=collections.Counter(re.findall(r'\w+',out.lower()))
        f1=2*sum((aw&bw).values())/max(1,sum(aw.values())+sum(bw.values()))
        diffs=[dict(at=a,source=source[a:b],output=out[c:d],tag=t) for t,a,b,c,d in difflib.SequenceMatcher(None,source,out,autojunk=False).get_opcodes() if t!='equal']
        details.append(dict(id=saved['id'],source=source,output=out,exact=source==out,
            source_bytes=len(raw),output_bytes=len(out.encode()),UTF8=decoded['UTF8'],EOS=decoded['EOS'],
            word_overlap_F1=f1,character_similarity=difflib.SequenceMatcher(None,source,out,autojunk=False).ratio(),changes=diffs))
    first_errors=first_total=rest_errors=rest_total=0
    for a,b in packet['offsets']:
        first_errors+=int(((pred[a:a+2]!=y[a:a+2])&body[a:a+2]).sum());first_total+=int(body[a:a+2].sum())
        rest_errors+=int(((pred[a+2:b]!=y[a+2:b])&body[a+2:b]).sum());rest_total+=int(body[a+2:b].sum())
    report=dict(status='INDEPENDENT_V56_SAVED_REVIEW_PASS',both_fullstate_SHA_safeload_digest=True,
        initial_active8Adam_scaler_ALLRNG_exact_V55=True,all8_final_Adam_step7248=True,
        all_stored_RNG_unchanged=True,scaler_scale8192_tracker1248=True,finite_order_cursor_exposure_verified=True,
        all4_required_concat_cache_SHA_verified=True,old528_shards_refs_exact_parent_prior_review=True,
        frozen_source_parent_SHA_unchanged=True,score=score,class_errors=dict(errors),class_totals=dict(counts),
        first32=dict(errors=first_errors,total=first_total),after32=dict(errors=rest_errors,total=rest_total),
        raw_exact=sum(r['exact'] for r in details),pairs=details,extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
        limitation='same observed DEV, class/position design adapted after V55; similarities are not semantic correctness')
    io.atomic_json(run/'root_saved_review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'}))
if __name__=='__main__':main()
