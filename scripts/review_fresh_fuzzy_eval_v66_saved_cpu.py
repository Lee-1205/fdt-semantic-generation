"""Raw saved logits, new held text, length/language, output-vs-byte gap."""
import collections,difflib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.codec_commit_stop_v56 import reconstruct_commit_logits

def aggregate(rows):
    n=sum(r['source_bytes'] for r in rows);s=sum(r['symbols'] for r in rows)
    return dict(rows=len(rows),bytes=n,byte_errors=sum(r['byte_errors'] for r in rows),byte_accuracy=1-sum(r['byte_errors'] for r in rows)/n,
        macro_window_byte_accuracy=sum(1-r['byte_errors']/r['source_bytes'] for r in rows)/len(rows),
        byte_EOS_accuracy=1-sum(r['symbol_errors'] for r in rows)/s,exact=sum(r['exact'] for r in rows),exact_rate=sum(r['exact'] for r in rows)/len(rows),
        UTF8=sum(r['UTF8'] for r in rows),EOS=sum(r['EOS'] for r in rows),NLL=sum(r['NLL_sum'] for r in rows)/s)

def main():
    torch.set_num_threads(4);run=ROOT/'runs/fresh_fuzzy_eval_20261112_v66';p=io.load_json(run/'plan.json');native=io.load_json(run/'ROOT_terminal_native.json');status=io.load_json(run/'final_status.json')
    assert native['nativePythonCount']==0 and native['allFourOwnedGone'] and status['status']=='COMPLETE' and status['exit']['all_owned_processes_gone']
    for name,digest in p['files'].items():assert support.sha(name)==digest,name
    source=io.load_json(Path(p['data'])/'rows.json');saved=io.load_json(run/'reconstructions.json');result=io.load_json(run/'result.json');rows=[]
    for i,(row,r) in enumerate(zip(source,saved)):
        raw=row['prompt'].encode();payload=torch.load(run/f'raw_{i:02d}.pt',weights_only=True,map_location='cpu');z,y=payload['logits'],payload['targets'];pred=z.argmax(-1)
        expected=torch.full_like(y.flatten(),257);expected[:len(raw)]=torch.tensor(list(raw));expected[len(raw)]=256;assert torch.equal(expected,y.flatten())
        assert payload['Exact_entries']==0 and payload['Exact_policy']==p['exact_policy'] and payload['fuzzy_commit'].count_nonzero()
        count=int((y!=257).sum());errors=int(((pred!=y)&(y<256)).sum());all_errors=int(((pred!=y)&(y!=257)).sum())
        ce=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'));d=reconstruct_commit_logits(z);out=d['text']
        assert r['id']==row['id'] and r['source']==row['prompt'] and out==r['output'] and d['ids']==r['ids'] and r['correct']==count-all_errors and r['byte_correct']==len(raw)-errors
        assert math.isclose(ce,r['NLL_sum'],rel_tol=1e-6,abs_tol=count*1e-7+1e-5)
        a,b=row['prompt'],out;changes=[]
        if a!=b:
            if row['language']=='English':
                # Corrected rows are almost identical; trim exact prefix/suffix
                # before alignment to keep long repeated contexts inexpensive.
                start=0
                while start<min(len(a),len(b)) and a[start]==b[start]:start+=1
                end=0
                while end<min(len(a),len(b))-start and a[-1-end]==b[-1-end]:end+=1
                aa=a[start:len(a)-end if end else len(a)];bb=b[start:len(b)-end if end else len(b)]
                for t,x,xx,u,uu in difflib.SequenceMatcher(None,aa,bb,autojunk=False).get_opcodes():
                    if t!='equal':changes.append(dict(kind=t,source=aa[x:xx],output=bb[u:uu],offset=start+x,context=a[max(0,start+x-65):min(len(a),start+xx+65)]))
            else:changes=[dict(kind='nonmatching_Korean_sentence',source=a,output=b)]
        word_a=collections.Counter(a.split());word_b=collections.Counter(b.split())
        rows.append(dict(index=i,id=row['id'],source_bytes=len(raw),language=row['language'],kind=row['kind'],requested_max_bytes=row.get('requested_max_bytes'),
            source=a,output=b,exact=a==b,UTF8=d['UTF8'],EOS=d['EOS'],byte_errors=errors,symbol_errors=all_errors,symbols=count,NLL_sum=ce,
            raw_early_EOS_positions=int(((pred==256)&(y<256)).sum()),output_bytes=len(b.encode()),byte_length_delta=len(b.encode())-len(raw),
            word_overlap_F1=2*sum((word_a&word_b).values())/max(1,sum(word_a.values())+sum(word_b.values())),changes=changes))
    groups={key:aggregate([r for r in rows if r['language']+'_'+r['kind']==key]) for key in sorted({r['language']+'_'+r['kind'] for r in rows})}
    english=aggregate([r for r in rows if r['language']=='English']);total=aggregate(rows)
    assert total['exact']==result['exact_windows'] and abs(total['byte_accuracy']-result['byte_accuracy'])<1e-12 and abs(total['NLL']-sum(r['NLL_sum'] for r in saved)/sum(r['symbols'] for r in saved))<1e-7
    lengths={str(size):aggregate([r for r in rows if r['requested_max_bytes']==size]) for size in (512,1024,2048,4096,8192,16384)}
    theory={str(L):dict(expected_errors_at_99_9_percent_byte_accuracy=L*.001,independent_illustration_probability_all_correct=.999**L) for L in (100,512,4096,16384)}
    review=dict(status='INDEPENDENT_FRESH_V66_SAVED_REVIEW_PASS',rows=52,optimizer_updates=0,Exact_entries=0,copy_bytes=0,denoise_calls=0,
        raw52_logits_targets_argmax_UTF8_guard_EOS_text_reconstructed=True,immutable_model_all_pins_unchanged=True,total=total,English_all=english,groups=groups,length_groups=lengths,
        illustrative_byte_to_whole_gap=theory,independence_is_NOT_established=True,
        UTF8_is_enforced_by_neural_score_FSM_not_language_quality=True,EOS_early_guard_uses_coarse_Commit_topology=True,
        extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),limitations='new authored correlated pairs/template contexts; known-local-TRAIN substring overlap0; full pretraining overlapUNKNOWN; no general semantic or natural-corpus certificate')
    io.atomic_json(run/'root_saved_review.json',review);io.atomic_json(run/'root_reading_review.json',rows)
    print(json.dumps(review))
if __name__=='__main__':main()
