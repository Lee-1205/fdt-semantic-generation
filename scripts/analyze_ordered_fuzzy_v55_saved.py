"""Saved-only stopping/error/readability calculation, no neural forward."""
import collections,difflib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.codec_commit_stop_v56 import reconstruct_commit_logits,guard_codec_stop

def main():
    torch.set_num_threads(4);run=ROOT/'runs/ordered_fuzzy_20261111_v55'
    final=io.load_json(run/'final_status.json');assert final['exit']['all_owned_processes_gone']
    states={}
    for name in ('initial','model'):
        meta=io.load_json(run/name/'latest.json')
        states[name]=torch.load(run/name/'checkpoints'/meta['file'],weights_only=True,map_location='cpu')
    a,b=states['initial'],states['model']
    for key in ('torch_rng','cuda_rng','python_rng','numpy_rng'):
        assert support.state_sha({key:a[key]})==support.state_sha({key:b[key]})
    assert b['scaler']['_growth_tracker']==1624 and b['scaler']['scale']==2048
    assert not a['head']['role_projection.weight'].count_nonzero() and b['head']['role_projection.weight'].count_nonzero()
    assert not a['optimizer']['state'] and len(b['optimizer']['state'])==8
    packet=torch.load(run/'normal_raw_logits.pt',weights_only=True,map_location='cpu')
    rows=io.load_json(Path(io.load_json(run/'plan.json')['data'])/'dev.json')
    normal=packet['logits'];labels=packet['targets'];guarded=[]
    for i,(start,end) in enumerate(packet['offsets']):
        guarded.append(guard_codec_stop(normal[start:end]))
    guarded=torch.cat(guarded);active=labels!=257;pred=guarded.argmax(-1);EOS=labels==256
    stats=dict(accuracy=float(((pred==labels)&active).sum()/active.sum()),
        EOS_false_positives=int(((pred==256)&(labels<256)).sum()),
        EOS_recall=float(((pred==256)&EOS).sum()/EOS.sum()))
    details=[]
    for i in (0,4,8,12):
        start,end=packet['offsets'][i];decoded=reconstruct_commit_logits(normal[start:end]);row=rows[i]
        source=row['prompt'];out=decoded['text'];n=min(len(source.encode()),len(out.encode()))
        source_words=collections.Counter(re.findall(r"\w+",source.lower()))
        words=collections.Counter(re.findall(r"\w+",out.lower()));hits=sum((source_words&words).values())
        f1=2*hits/max(1,sum(source_words.values())+sum(words.values()))
        changes=[dict(tag=t,source=source[a:b],output=out[c:d],at=a) for t,a,b,c,d in difflib.SequenceMatcher(None,source,out,autojunk=False).get_opcodes() if t!='equal']
        details.append(dict(id=row['id'],source=source,output=out,exact=out==source,EOS=decoded['EOS'],UTF8=decoded['UTF8'],
            source_bytes=len(source.encode()),output_bytes=decoded['output_bytes'],word_overlap_F1=f1,
            character_similarity=difflib.SequenceMatcher(None,source,out,autojunk=False).ratio(),changes=changes))
    io.atomic_json(run/'root_commit_stop_candidate.json',dict(policy='EOS legal in last local slot or structural tail only; no forced EOS or gold byte length',
        observed_DEV_candidate_not_promotion=True,raw_exact=sum(r['exact'] for r in details),metrics=stats,
        fullstate_RNG_exact=True,scaler_scale2048_tracker1624=True,projection_learned=True,new8Adam_steps3624=True,
        pairs=details,extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized()))
    print(json.dumps(dict(metrics=stats,pairs=[{k:r[k] for k in ('id','source_bytes','output_bytes','exact','word_overlap_F1','character_similarity')} for r in details],model_calls=0)))
if __name__=='__main__':main()
