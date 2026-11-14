"""Select eight recorded cases BEFORE any cross-readout calls; no NN here."""
import gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from review_long_extrapolation_v69_saved_cpu import PATTERN

def main():
    out=ROOT/'research/artifacts/commit_cross_20261112_v71_cpu_v2';out.mkdir(exist_ok=False)
    old=ROOT/'runs/hard_fuzzy_cycle_20261112_v67';new=ROOT/'runs/factual_cycle_20261112_v70';long=ROOT/'runs/long_extrapolation_20261112_v69'
    held=io.load_json(new/'retained_English_reconstructions.json');natural=io.load_json(new/'final_reconstructions.json')
    long_features=io.load_json(ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu/features.json');long_targets=io.load_json(ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu/targets.json')
    cases=[]
    def add(name,oldraw,newraw,focus,source,topic):
        cases.append(dict(name=name,old_raw=str(oldraw),new_raw=str(newraw),focus_byte=focus,patch=focus//16,focus_role=focus%16,
            gold_byte=source.encode()[focus],source_context=source.encode()[max(0,focus-45):focus+65].decode('utf8',errors='replace'),topic=topic))
    add('clock_colon_regression',old/'raw_retained_English_04.pt',new/'raw_retained_English_04.pt',4,held[4]['source'],'time punctuation 6:45 -> 6045')
    add('retained_quantity_regression',old/'raw_retained_English_32.pt',new/'raw_retained_English_32.pt',3396,held[32]['source'],'27 -> 2&')
    add('retained_name_regression',old/'raw_retained_English_30.pt',new/'raw_retained_English_30.pt',2128,held[30]['source'],'Inez -> Ine9')
    pos=natural[10]['source'].encode().index(b'U-24')
    add('natural_category_correction',old/'raw_final_10.pt',new/'raw_final_10.pt',pos,natural[10]['source'],'first U-24 -> 1-24 corrected')
    for name,index,event,field,which,topic in [
        ('long_quantity_regression',8,318,'quantity','last','7 -> 6'),
        ('long_quantity_correction',9,50,'quantity','last','28 -> 24 corrected'),
        ('long_name_regression',8,49,'actor','first','Cyril -> byril')]:
        f=long_features[index];e=long_targets[index]['events'][event-1];phrase=f['prompt'].encode()[e['byte_start']:e['byte_end']].decode();m=PATTERN.fullmatch(phrase);assert m
        focus=e['byte_start']+(m.end(field)-1 if which=='last' else m.start(field))
        add(name,long/f'raw_{index:02d}.pt',new/f'raw_long_{index-8}.pt',focus,f['prompt'],topic)
    add('preserved_short_control',old/'raw_retained_English_00.pt',new/'raw_retained_English_00.pt',0,held[0]['source'],'unchanged source-bound control')
    features={k:[] for k in ('old_local','old_global','new_local','new_global')};native={'old':[],'new':[]};pins={}
    # Read each raw once, including the large long-output packets. Extract only
    # standard readout memory values. Gold offsets only select POSTHOC probes.
    for side in ('old','new'):
        loaded={}
        for file in dict.fromkeys(c[side+'_raw'] for c in cases):
            packet=torch.load(file,map_location='cpu',weights_only=True);pins[file]=support.sha(file)
            for i,c in enumerate(cases):
                if c[side+'_raw']!=file:continue
                z=packet['fuzzy_commit'];index=c['patch'];y=packet['targets'];scores=packet['logits'][index].float()
                assert z.shape[0]==1 and index+2<z.shape[1] and int(y[index,c['focus_role']])==c['gold_byte'] and packet['Exact_entries']==0
                loaded[i]=(z[0,index+2].clone(),z[0,:2].clone(),scores.clone())
            del packet;gc.collect()
        for i in range(8):
            a,b,z=loaded[i];features[side+'_local'].append(a);features[side+'_global'].append(b);native[side].append(z)
        del loaded;gc.collect()
    features={k:torch.stack(v) for k,v in features.items()};features['terminal']=torch.zeros(8,dtype=torch.bool)
    assert set(features)=={'old_local','old_global','new_local','new_global','terminal'}
    torch.save(features,out/'features.pt');torch.save({k:torch.stack(v) for k,v in native.items()},out/'reference_logits.pt')
    for i,c in enumerate(cases):
        for side in ('old','new'):c[side+'_native_focus_argmax']=int(native[side][i][c['focus_role']].argmax())
    assert all(cases[i]['old_native_focus_argmax']==cases[i]['gold_byte'] and cases[i]['new_native_focus_argmax']!=cases[i]['gold_byte'] for i in (0,1,2,4,6))
    assert all(cases[i]['old_native_focus_argmax']!=cases[i]['gold_byte'] and cases[i]['new_native_focus_argmax']==cases[i]['gold_byte'] for i in (3,5))
    assert cases[7]['old_native_focus_argmax']==cases[7]['new_native_focus_argmax']==cases[7]['gold_byte']
    io.atomic_json(out/'cases.json',cases)
    io.atomic_json(out/'manifest.json',dict(schema='commit_cross_v71_data',cases=8,source_raw_SHA=pins,source_models=['V67','V70'],NN_features=list(features),
        gold_focus_context_OFFLINE_SCORE_ONLY=True,selection='preselected recorded regressions/corrections plus unchanged control; not random or independent cases',
        actual_parent_or_head_forward=0,CUDA_initialized=torch.cuda.is_initialized(),copy_Exact_denoise=0,
        limitation='saved bound Commit includes upstream role_projection; final decode_patch skips re-binding. No far-context/abstract-meaning inference proof. CPU FP32 diagonal focus parity must be checked before cross interpretation.'))
    assert not torch.cuda.is_initialized();print(json.dumps([dict(name=c['name'],gold=c['gold_byte'],old=c['old_native_focus_argmax'],new=c['new_native_focus_argmax']) for c in cases]))
if __name__=='__main__':main()
