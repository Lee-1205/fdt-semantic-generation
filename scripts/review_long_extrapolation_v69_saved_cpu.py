"""Independent saved-only faithfulness review. No NN calls or CUDA init."""
import json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

PATTERN=re.compile(r'in episode (?P<episode>\d+), (?P<actor>\w+) moved (?P<quantity>\d+) (?P<item>.*?) from the (?P<source_color>\w+) shelf to the (?P<destination_color>\w+) drawer (?P<order>before|after) (?P<checker>\w+) closed the (?P<gate_color>\w+) gate, and the (?P<box_color>\w+) box contained (?P<reserve>\d+) (?P<reserve_item>.*?) that (?P<negative_actor>\w+) did (?P<negative>not )?remove',re.I)
CORE=('actor','quantity','item','source_color','destination_color','order','checker','gate_color','box_color','reserve','reserve_item','negative_actor','negative')

def decode_independent(logits,ids):
    z=logits.reshape(-1,258).numpy();n=len(z);need=0;lo=128;hi=191;raw=[];early_eos_limit=n-32
    for i,chosen in enumerate(ids):
        legal=np.zeros(258,dtype=bool)
        if need:legal[lo:hi+1]=True
        else:
            legal[:128]=True;legal[256]=i>=early_eos_limit
            if n-i>=2:legal[194:224]=True
            if n-i>=3:legal[224:240]=True
            if n-i>=4:legal[240:245]=True
        best=int(np.argmax(np.where(legal,z[i],-np.inf)));assert chosen==best,(i,chosen,best)
        if chosen==256:assert not need;break
        raw.append(chosen)
        if need:need-=1
        else:need=1 if 194<=chosen<224 else 2 if 224<=chosen<240 else 3 if 240<=chosen<245 else 0
        lo=160 if chosen==224 else 144 if chosen==240 else 128;hi=159 if chosen==237 else 143 if chosen==244 else 191
    assert need==0
    return bytes(raw).decode('utf8')

def fields(match):
    d={k:v.casefold().strip() if v else '' for k,v in match.groupdict().items()}
    for k in ('episode','quantity','reserve'):d[k]=int(d[k])
    return d

def rel(a,b):
    return float((a.double()-b.double()).norm()/a.double().norm().clamp_min(1e-12))

def main():
    torch.set_num_threads(4);run=ROOT/'runs/long_extrapolation_20261112_v69';p=io.load_json(run/'plan.json');data=Path(p['data'])
    terminal=io.load_json(run/'ROOT_terminal_native.json');assert terminal['all_known_pids_gone'] and terminal['Python_native_count']==0
    owner=io.load_json(run/'final_status.json');assert owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone']
    features=io.load_json(data/'features.json');targets=io.load_json(data/'targets.json');saved=io.load_json(run/'reconstructions.json');result=io.load_json(run/'result.json')
    assert len(features)==len(targets)==len(saved)==10
    assert support.sha(p['cycle_source'])==p['cycle_sha'];assert result['calls']==dict(encode=14,readout=14,stream_chunk=500,denoise=0) and result['block_work']==4518
    for name,digest in p['files'].items():assert support.sha(name)==digest,name
    parity=[]
    for i in (0,1):
        packet=torch.load(run/f'parity_{i}.pt',map_location='cpu',weights_only=True);a,b=packet['native'],packet['stream']
        vals=dict(commit_relative_L2=rel(a['fuzzy_commit'],b['fuzzy_commit']),logits_relative_L2=rel(a['logits'],b['logits']),
            bank_relative_L2=max(rel(a['banks'][k],b['banks'][k]) for k in a['banks']),argmax_identical=bool(torch.equal(a['logits'].argmax(-1),b['logits'].argmax(-1))))
        assert all(vals[k]<=v for k,v in p['parity_thresholds'].items());parity.append(vals)
    cards=[];per=[];risks=[];samples=[];nll=0.;byte_correct=0;source_total=0
    for i,(f,t,r) in enumerate(zip(features,targets,saved)):
        packet_path=run/f'raw_{i:02d}.pt';assert support.sha(packet_path)==r['raw_sha256']
        raw=torch.load(packet_path,map_location='cpu',weights_only=True);z,y=raw['logits'],raw['targets'];source=f['prompt'];b=source.encode();out=r['output']
        expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256;assert torch.equal(expected,y.flatten())
        assert decode_independent(z,r['ids'])==out and raw['Exact_entries']==0 and raw['Exact_policy']==p['exact_policy'] and torch.isfinite(z).all()
        ce=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'));count=len(b)+1
        assert math.isclose(ce,r['NLL_sum'],rel_tol=1e-6,abs_tol=4*torch.finfo(torch.float32).eps*count+1e-5)
        correct=int(((z.argmax(-1)==y)&(y<256)).sum());assert correct==r['byte_correct'];source_total+=len(b);byte_correct+=correct;nll+=ce
        gold=list(PATTERN.finditer(source));pred=list(PATTERN.finditer(out));assert len(gold)==len(t['events'])
        # Event identity from ordinary prose, not source offsets/labels passed to the model.
        by_number={};duplicates=[]
        for m in pred:
            number=int(m.group('episode'))
            if number in by_number:duplicates.append(number)
            by_number[number]=m
        good=parsed=boundary_good=0;row_risks=[]
        for j,(g,e) in enumerate(zip(gold,t['events'])):
            gf=fields(g);number=gf['episode'];pm=by_number.get(number);pf=fields(pm) if pm else None
            if pf is None:changed=None;category='unparsed_requires_manual_review'
            else:
                parsed+=1;changed={k:dict(source=gf[k],output=pf[k]) for k in CORE if gf[k]!=pf[k]};category='all_core_fields_equal' if not changed else 'content_risk';good+=not changed
            crossing=e['byte_start']//2048!=(e['byte_end']-1)//2048
            if crossing and category=='all_core_fields_equal':boundary_good+=1
            if category!='all_core_fields_equal':
                context=out[max(0,e['byte_start']-25):min(len(out),e['byte_end']+25)]
                risk=dict(row=i,event=number,category=category,changes=changed,source=g.group(),output=pm.group() if pm else context,
                    source_byte_position=e['byte_start'],relative_position=e['byte_start']/len(b),crosses_compute_boundary=crossing)
                row_risks.append(risk);risks.append(risk)
            cards.append(dict(row=i,event=number,source_fields=gf,output_fields=pf,category=category,crosses_compute_boundary=crossing))
        for j in (0,len(gold)//2,len(gold)-1):
            g=gold[j];m=by_number.get(int(g.group('episode')));samples.append(dict(row=i,region='start' if j==0 else 'end' if j==len(gold)-1 else 'middle',source=g.group(),output=m.group() if m else 'UNPARSED'))
        differences=[dict(position=j,source=source[max(0,j-35):j+45],output=out[max(0,j-35):j+45]) for j in range(min(len(source),len(out))) if source[j]!=out[j]]
        per.append(dict(row=i,style=t['style'],source_UTF8_bytes=len(b),multiple_of_training_max=len(b)/16384,
            events=len(gold),parsed_output_events=len(pred),core_fields_equal=good,unparsed_events=len(gold)-parsed,content_risk_events=parsed-good,
            boundary_events=t['boundary_crossing_events'],boundary_core_equal=boundary_good,duplicate_episode_ids=duplicates,
            extra_parsed_episode_ids=sorted(set(by_number)-{int(m.group('episode')) for m in gold}),
            EOS=r['EOS'],UTF8=r['UTF8'],source_output_length_delta=len(out.encode())-len(b),literal_exact_auxiliary=source==out,
            raw_byte_accuracy_auxiliary=correct/len(b),positional_differences_auxiliary=len(differences),
            difference_contexts= differences,actual_stream_stats=r['stream_stats']))
    summary=dict(status='ROOT_LONG_EXTRAPOLATION_V69_SAVED_REVIEW_PASS',source_model_unchanged=True,all_raw_SHA_target_argmax_UTF8_FSM_verified=True,
        per_row=per,source_UTF8_bytes=source_total,correlated_event_bundles=len(cards),core_fields_equal=sum(x['category']=='all_core_fields_equal' for x in cards),
        unparsed_requires_manual=sum(x['category']=='unparsed_requires_manual_review' for x in cards),content_risks=len(risks),parity=parity,
        raw_byte_accuracy_auxiliary=byte_correct/source_total,NLL_auxiliary=nll/(source_total+10),
        extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),Exact_entries=0,copy_bytes=0,denoise_calls=0,
        limitations='offline restricted authored-grammar scoring; correlated repeated event prose, no blind natural corpus/general reasoning or abstract meaning certificate; unparsed/changed surface forms require ROOT reading; local memory grows with length; byte length not BPE')
    assert not torch.cuda.is_initialized()
    io.atomic_json(run/'root_saved_review.json',summary);io.atomic_json(run/'root_event_cards.json',cards);io.atomic_json(run/'root_reading_contexts.json',dict(risks=risks,region_samples=samples))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_row',)}));print(json.dumps([{k:v for k,v in r.items() if k not in ('difference_contexts','actual_stream_stats')} for r in per]));print(json.dumps(risks))
if __name__=='__main__':main()
