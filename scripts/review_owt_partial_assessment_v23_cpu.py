"""Saved inference evidence only; original partial training status stays FAILED."""
import json,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from review_owt_free_similarity_20261028_cpu import metrics,chrf,REVIEWS
from review_owt_fuzzy_reconstruction_v12_20261028_cpu import stop_count
from review_owt_state_reconstruction_v22_cpu import READINGS,TRAIN_READINGS

DEV_COMMENTS=[
 'He and people are fragments; negative acknowledgement, culpability and deceit relation are absent.',
 'Fentally is not preserved fans; waving flowers and standing are absent.',
 'Capital-letter fragments/questions only; Disney, sundae and lack-of-knowledge claim are absent.',
 'Globaly resembles globally; ransom payments, victims and backup recovery are absent.',
 'Panel, assault team, Slade Wilson and Moscow are replaced by unrelated name-like strings.',
 'Malaysia/Senate/Americans/2015 replace the Israel-Syria-Assad stable-ceasefire relation.',
 'She survives; sobriety-test failure and DUI charge are absent.',
 'I and police are fragments; taking items while not paying at the stall is absent.',
 'But more survives; speech restriction, negation and causal explanation are absent.',
 'Carlin/boomb are not correct scout names or the player/radar/three-month relationship.',
 'Asked/study are topical hints; second reviewer, anonymous article and the request relation are absent.',
 'Websuary is not Wubbels; lawyer, inaction and the concern relationship are absent.',
 'List-like names replace gunmen, vessels and the actual country origins.',
 'Russian/Twitter/Records do not state Jamaican bobsled qualification for the Olympics in Sochi.',
 'Japanies/whall resemble source spellings, but named agent/prevention/assassination/instrument are not faithfully stated.',
 'Structure/programs are domain hints; named functions, temporary string and pointer-array relationship are absent.'
]
TRAIN_COMMENTS=[
 'More people is retained; critical condition, arrival and late Friday are absent.',
 'Delay/postponement/Clinton/five-primary uncertainty are absent; international medical plants are invented.',
 'Court is retained as a word, but Brazau, intention to appeal and sentence are missing.',
 'Educations is not Eduardo; cash-payment request and bank/card fees are absent.',
 'Nother is not NATO; buying and using thumb drives/Russian network access are missing.',
 'They will is retained, but probable opinions about the bike are absent.',
 'The tweets were survives, but go fand is not flagged. The prior correct partial subject-action proposition is no longer expressed.',
 'The season-ending condition and no-more qualifying tournaments are absent.',
 'Bill survives but Gampson is not Haslam; panel discussion and named governor are absent.',
 'Donations is correctly retained, but family recipient, support correspondence and exact postal address are missing.',
 'It just goes to is retained; moral-compass judgment and even the full show-you lead are lost.',
 'The source is fragmentary; Iraq, diplomats and U.S. references are not retained. No invented complete gold proposition.',
 'Chinese/UK are geographical strings, but agreement, film title, screening, China and August11 are not faithfully stated.',
 'Taking orders, actual2012time intervals and shipment are replaced by unrelated1999/Wednesday/project material.',
 'Cravia Child is not Craig Charles; the actual four names and joining relationship are absent.',
 'Year-old/told-like opening fragments remain; actual age, police identity and prohibition on movement are missing.'
]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    torch.set_num_threads(2);run=ROOT/'runs/owt_partial_assessment_20261028_v23_readonly'
    p,r,end=read(run/'plan.json'),read(run/'result.json'),read(run/'supervisor/final_status.json')
    native=read(run/'code_snapshot/root_terminal_native_observation.json')
    training=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v23_final';audit=read(training/'root_partial_state_review.json')
    assert end['status']=='COMPLETE' and end['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    assert read(training/'supervisor/final_status.json')['status']=='FAILED'
    assert audit['new_updates']==7437 and r['optimizer_updates']==r['checkpoint_writes']==0 and r['all_used_RNG_unchanged']
    assert support.sha(run/'plan.json')==end['plan_sha256'] and support.sha(p['source'])==r['source_sha256']==p['source_sha']
    s=torch.load(p['source'],map_location='cpu',weights_only=True)
    assert support.state_sha(s['model'])==r['source_model_sha256']==r['model_after_sha256'];del s
    f=read(Path(p['artifact'])/'features.json');y=read(Path(p['artifact'])/'targets.json')
    raw=read(run/'final_generation.json');train=read(run/'final_train_reconstruction.json')
    summaries={};records={}
    for label,rows in [*raw.items(),('train_normal',train)]:
        items=[]
        for i,row in enumerate(rows):
            expected=f['train'][p['train_reconstruction_indices'][i]] if label=='train_normal' else f['dev'][p['generation_indices'][i]]
            assert expected=={'id':row['id'],'prompt':row['source']}
            g=row['generated'];assert bytes.fromhex(g['raw_hex']).decode('utf-8')==g['text']
            assert row['reconstruction_exact']==(g['well_formed'] and g['text']==row['source'])
            item={'index':i,'source':row['source'],'reconstruction':g['text'],**metrics(row['source'],g['text'])}
            if label in ('normal','train_normal'):
                fields=TRAIN_READINGS[i] if label=='train_normal' else READINGS[i]
                item.update(subject=fields[0],action=fields[1],object=fields[2],polarity=('negative acknowledgement missing' if label=='normal' and i==0 else 'agreement not preserved' if label=='train_normal' and i==12 else fields[3]),relations=fields[4],ROOT_reading=(TRAIN_COMMENTS if label=='train_normal' else DEV_COMMENTS)[i],meaning_grade='meaning_not_preserved',faithful_paraphrase=False,partial_proposition=False)
            items.append(item)
        summary={k:statistics.mean(x[k] for x in items) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
        summary.update(exact=sum(row['reconstruction_exact'] for row in rows),UTF8=sum(row['generated']['utf8_valid'] for row in rows),EOS=sum(row['generated']['eos'] for row in rows))
        if label!='train_normal':
            top=0
            for i,item in enumerate(items):
                values=[chrf(other['source'],item['reconstruction']) for other in items]
                top+=values[i]==max(values) and values.count(max(values))==1
            summary['own_source_unique_top1_of16']=top
        summaries[label]=summary;records[label]=items
    calls={'encode':32+4+4,'denoise':0,'decoder':96+sum(stop_count(rows) for rows in raw.values())+stop_count(train)}
    assert calls==r['actual_calls']
    data=torch.load(run/'final_logits.pt',map_location='cpu',weights_only=True);ids=data['target_ids'];active=ids!=257
    for i,target in enumerate(y['dev']):assert ids[i].tolist()==target['target_ids']+[257]*(129-len(target['target_ids']))
    verified={}
    for mode in raw:
        x=data[mode].double();nll=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1);correct=int((x.argmax(-1)[active]==ids[active]).sum());score=float(nll[active].mean())
        assert correct==r['final'][mode]['correct'] and abs(score-r['final'][mode]['NLL'])<.001
        verified[mode]={'NLL':score,'correct':correct,'symbols':int(active.sum()),'byte_accuracy':correct/int(active.sum())}
    out={'status':'V23_PARTIAL_READONLY_ROOT_RAW_REVIEWED','training_actual_status_retained':'FAILED','training_new_updates':7437,'global_updates':47061,
        'source_sha256':p['source_sha'],'source_model_sha256':r['source_model_sha256'],'source_file_and_model_verified_unchanged':True,
        'worker_used_RNG_unchanged_check':r['all_used_RNG_unchanged'],'optimizer_updates':0,'checkpoint_writes':0,
        'saved_logit_metrics':verified,'state_reconstruction_metrics':summaries,'ROOT_read_dev16_and_train16':True,
        'ROOT_faithful_dev':0,'ROOT_faithful_train':0,'ROOT_partial_dev':0,'ROOT_partial_train':0,'prior_train_flagged_proposition_not_preserved_now':True,
        'actual_calls':calls,'total_calls':sum(calls.values()),'seconds':r['seconds'],'plan_sha256':end['plan_sha256'],
        'all_owned_gone_before_analysis':True,'additional_model_forwards':0,'E2_final_accessed':False,
        'limits':'Observed dev repeatedly reused/nonblind ROOT reading/lexical not semantic truth. Teacher gains are not faithful whole reconstruction. Dataset, extra updates, order, partial cutoff and new-row heuristics are confounded.'}
    with (run/'root_result.json').open('x',encoding='utf-8') as h:json.dump(out,h,ensure_ascii=False,indent=2)
    with (run/'root_state_reconstruction_similarity.json').open('x',encoding='utf-8') as h:json.dump({'summary':summaries,'rows':records},h,ensure_ascii=False,indent=2)
    lines=['# V23 saved partial: SAMEINPUT state-conditioned reconstruction, ROOT all32 reading','',out['limits']]
    for label in ('normal','train_normal'):
        for row in records[label]:
            lines+=['',f"## {label} {row['index']}",f"Source: {row['source']}",f"Reconstruction: {row['reconstruction']}",f"Subject/action/object: {row['subject']} / {row['action']} / {row['object']}",f"Polarity: {row['polarity']}",f"Relations: {row['relations']}",f"ROOT: {row['ROOT_reading']}",f"Grade: {row['meaning_grade']}; chrF={row['chrF2_1to6']:.3f}"]
    with (run/'root_all32_state_reconstruction_reading.md').open('x',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')
    print(json.dumps(out,ensure_ascii=True))

if __name__=='__main__':main()
