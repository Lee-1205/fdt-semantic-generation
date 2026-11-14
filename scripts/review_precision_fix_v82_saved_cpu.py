"""Independent FP32 fix check and affine update conflict, no NN calls."""
import collections,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.lossless_raw_tensor_v80 import unpack
from review_factual_cycle_v70_saved_cpu import verify_output
from review_v80_semantic_severity_cpu import numeric_severity

def main():
    torch.set_num_threads(4);run=ROOT/'runs/precision_fix_check_20261113_v82';p=io.load_json(run/'plan.json');oldrun=ROOT/'runs/update_retention_20261113_v80'
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json');assert terminal['AllOwnedGone'] and not terminal['PythonProcesses'] and terminal['BeforeAnalysis'] and owner['exit']['all_owned_processes_gone'] and owner['status']=='COMPLETE'
    for f,s in p['files'].items():assert support.sha(f)==s,f
    items=io.load_json(p['input_list']);records=io.load_json(run/'reconstructions.json');oldrecords=io.load_json(oldrun/'new_English_reconstructions.json');targets=io.load_json(p['targets']);counts=collections.Counter();cards=[];NLL=collections.Counter()
    for i,(item,r,oldr,t) in enumerate(zip(items,records,oldrecords,targets)):
        source=item['row']['prompt'].encode();assert item['row']['id']==r['id']==oldr['id']==t['id']
        a=unpack(torch.load(item['raw'],map_location='cpu',weights_only=True));b=torch.load(r['raw'],map_location='cpu',weights_only=True);assert support.sha(r['raw'])==r['SHA'] and b['source_id']==r['id']
        z=b['logits'];y=a['targets'];valid=y!=257;assert z.dtype==torch.float32 and z.shape==a['logits'].shape and torch.isfinite(z).all()
        assert verify_output(z,r['ids'],16385)==r['output'] and r['UTF8'] and r['EOS']
        oldcorrect=(a['logits'].argmax(-1)==y)&valid;newcorrect=(z.argmax(-1)==y)&valid
        counts['byte_EOS_positions']+=int(valid.sum());counts['old_correct']+=int(oldcorrect.sum());counts['new_correct']+=int(newcorrect.sum())
        counts['raw_fixed']+=int((~oldcorrect&newcorrect&valid).sum());counts['raw_new_wrong']+=int((oldcorrect&~newcorrect&valid).sum())
        for name,v in [('old',a['logits']),('new',z)]:NLL[name]+=float(torch.nn.functional.cross_entropy(v.transpose(1,2),y,ignore_index=257,reduction='sum'))
        out=r['output'].encode();oldout=oldr['output'].encode();counts['old_decoded_errors']+=sum(x!=v for x,v in zip(source,oldout))+abs(len(source)-len(oldout));counts['new_decoded_errors']+=sum(x!=v for x,v in zip(source,out))+abs(len(source)-len(out))
        counts['old_exact_auxiliary']+=oldout==source;counts['new_exact_auxiliary']+=out==source;counts['UTF8_EOS']+=1
        changes=[j for j,(x,v) in enumerate(zip(oldout,out)) if x!=v]
        if oldout!=out:
            cards.append(dict(row=i,id=r['id'],source=item['row']['prompt'],before=oldr['output'],after=r['output'],
                changed_offsets=changes,contexts=[dict(offset=j,source=source[max(0,j-32):j+40].decode(errors='replace'),before=oldout[max(0,j-32):j+40].decode(errors='replace'),after=out[max(0,j-32):j+40].decode(errors='replace')) for j in changes]))
    severity={'before':numeric_severity(targets,oldrecords),'after':numeric_severity(targets,records)}
    raw=torch.load(ROOT/'runs/readout_cause_20261113_v81/focused_patch_logits.pt',map_location='cpu',weights_only=True);cs=io.load_json(ROOT/'research/artifacts/readout_cause_20261113_v81_cpu_v1/cases.json');affine=[]
    oldhead=torch.load(p['heads']['V75']['path'],map_location='cpu',weights_only=True)['active_weights'];newhead=torch.load(p['heads']['V80']['path'],map_location='cpu',weights_only=True)['active_weights']
    for c in cs:
        name=c['name'];g=c['gold'];role=c['offset']%16;a=raw['V75/full_fp32/'+name][role].double();b=raw['V80/full_fp32/'+name][role].double();d=b-a;lo,hi=0.,1.;constraints=[]
        for j in range(258):
            if j==g:continue
            gap=float(a[g]-a[j]);slope=float(d[g]-d[j]);needed=1e-6-gap
            if slope>0:lo=max(lo,needed/slope)
            elif slope<0:hi=min(hi,needed/slope)
            elif gap<1e-6:lo=math.inf;hi=-math.inf
        competitor=int(b.argmax());index=role*258
        db=newhead['readout.readout.net.2.bias'].double()-oldhead['readout.readout.net.2.bias'].double()
        bias_delta=float(db[index+g]-db[index+competitor]);score_delta=float((b[g]-b[competitor])-(a[g]-a[competitor]))
        affine.append(dict(name=name,old_FP32_correct=int(a.argmax())==g,new_FP32_correct=int(b.argmax())==g,valid_alpha_min=lo,valid_alpha_max=hi,feasible=lo<=hi,
            gap_vs_final_winner_before=float(a[g]-a[competitor]),gap_vs_final_winner_after=float(b[g]-b[competitor]),gap_delta=score_delta,bias_part_of_gap_delta=bias_delta,weight_hidden_part_of_gap_delta=score_delta-bias_delta))
    pair=[v for v in affine if v['name'] in ('new8','new28')];assert len(pair)==2 and max(v['valid_alpha_min'] for v in pair)>min(v['valid_alpha_max'] for v in pair)
    report=dict(status='ROOT_V82_FIXED_FP32_CHECK_INDEPENDENTLY_VERIFIED',actual_calls=64,extra_model_calls=0,counts=dict(counts),NLL={k:v/counts['byte_EOS_positions'] for k,v in NLL.items()},
        B_numeric_severity=severity,changed_output_cards=cards,affine_final_layer_conflicts=affine,global_alpha_conflict_example=pair,
        interpretation='Identical cached Commit and five readout prefix tensors, only final Linear weights differ V75/V80. Full-FP32 affine endpoint differences isolate final projection update for selected regressions. No single old/new parameter interpolation can make both new8 and new28 correct on these observed examples; evaluation must NOT become a training selector.',
        policy='fixed_full_FP32_Commit_readout_no_Gold_no_Exact_v81',weights_unchanged=True,all_owned_gone_before_analysis=True,training_steps=0,encoder_calls=0,denoise_calls=0,
        limitation='Observed correlated panel; semantic prefix uncertainty preserved, not all nonliteral strings certified. Input meaning abstraction and unseen-long generalization remain unfinished.')
    # JSON standard does not represent infinities. Mark impossible individual
    # affine intervals explicitly without pretending an unbounded scalar is data.
    for v in affine:
        for k in ('valid_alpha_min','valid_alpha_max'):
            if not math.isfinite(v[k]):v[k]=None
    io.atomic_json(run/'root_saved_review.json',report)
    print(json.dumps(dict(counts=dict(counts),NLL=report['NLL'],severity={k:v['counts'] for k,v in severity.items()},changed_rows=[c['row'] for c in cards],alpha_conflict=pair)))

if __name__=='__main__':main()
