"""Independent saved-logit reconstruction. No model calls or output repair."""
import collections,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    run=ROOT/'runs/readout_cause_20261113_v81';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json');term=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert term['AllOwnedGone'] and not term['PythonProcesses'] and term['BeforeAnalysis'] and owner['status']=='COMPLETE' and owner['exit']['all_owned_processes_gone'] and owner['exit_code']==0
    for f,s in p['files'].items():assert support.sha(f)==s,f
    assert support.sha(run/'focused_patch_logits.pt')==result['raw_SHA'];raw=torch.load(run/'focused_patch_logits.pt',map_location='cpu',weights_only=True)
    cases=io.load_json(p['cases']);inputs=io.load_json(p['inputs']);diags=io.load_json(run/'case_diagonals.json');cards=[];counts=collections.Counter()
    for c in cases:
        name=c['name'];off=c['offset'];gold=c['gold'];role=off%16;source=inputs[c['input']]['row']['prompt'].encode();sides={}
        for key,z in raw.items():
            if key.rsplit('/',1)[-1]!=name:continue
            assert z.shape==(16,258) and z.dtype==torch.float32 and torch.isfinite(z).all()
            scores=z[role];other=scores.clone();other[gold]=-torch.inf;order=scores.argsort(descending=True)[:4].tolist();guess=int(scores.argmax())
            sides[key.rsplit('/',1)[0]]=dict(pred=guess,char=chr(guess) if guess<256 else str(guess),correct=guess==gold,margin=float(scores[gold]-other.max()),
                gold_NLL=float(-scores.log_softmax(-1)[gold]),top=[dict(byte=j,score=float(scores[j])) for j in order],
                patch_text=bytes(v if v<256 else 63 for v in z.argmax(-1).tolist()).decode('utf-8',errors='replace'))
        d=next(v for v in diags if v['case']['name']==name);normal=d['normal_mode'];native=raw['native/'+name];again=raw['V80/'+normal+'/'+name]
        assert d['diagonal_focus_matches_native']==(int(native[role].argmax())==int(again[role].argmax())) and d['diagonal_max_abs']==float((native-again).abs().max())
        counts['diagonal_focus_reproduced']+=d['diagonal_focus_matches_native']
        for mode in p['precision_modes']:counts['V80_'+mode+'_focus_correct']+=sides['V80/'+mode]['correct']
        counts['native_focus_correct']+=sides['native']['correct']
        if 'partner' in c:
            # From_commit uses only global2+one local patch for each patch.
            # Swapping both relevant memories MUST match the partner diagonal.
            b=raw['V80/both_swap/'+name];partner=raw['V80/partner_native/'+name]
            assert torch.equal(b,partner),'readout locality algebra/control failed'
            counts['both_swap_partner_patch_bitwise']+=1
        cards.append(dict(name=name,source_context=source[max(0,off-40):off+55].decode(),gold=chr(gold),offset=off,diagonal=d,conditions=sides))
    report=dict(status='ROOT_V81_READONLY_CAUSE_SAVED_REVIEW_COMPLETE',actual_calls=133,extra_CPU_GPU_model_calls=0,counts=dict(counts),cards=cards,
        selected_not_blind=True,labels_not_normal_inference_features=True,weights_unchanged=result['all_weights_unchanged'],all_owned_exit_confirmed_before_analysis=True,
        caution='Focus accuracy is a diagnosis over selected bytes, not a whole panel performance score; full FP32 residuals can be readout or bound upstream issues. No encoder/binding disaggregation without new separate intervention.')
    io.atomic_json(run/'root_saved_review.json',report)
    print(json.dumps(dict(counts=dict(counts),cases=[dict(name=c['name'],gold=c['gold'],parity=c['diagonal']['diagonal_focus_matches_native'],
        values={k:str(v['char'])+':'+str(round(v['margin'],5)) for k,v in c['conditions'].items()}) for c in cards])))

if __name__=='__main__':main()
