"""CPU-only component/penalty calculation from the one saved140-call trace."""
import json,re,sys,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import text_generation_support_v1 as support

def scores(logits,ids):
    x=logits.double();mask=ids!=257;loss=x.logsumexp(-1)-x.gather(-1,ids[...,None]).squeeze(-1)
    return {'nll_sum':float(loss[mask].sum()),'correct':int((x.argmax(-1)[mask]==ids[mask]).sum()),'symbols':int(mask.sum())}

def combine(items):
    total={k:sum(x[k] for x in items) for k in items[0]}
    return {**total,'NLL':total['nll_sum']/total['symbols'],'accuracy':total['correct']/total['symbols']}

def main():
    torch.set_num_threads(2);r=ROOT/'runs/decoder_boundary_readonly_20261028_v1'
    result=json.loads((r/'result.json').read_text());terminal=json.loads((r/'code_snapshot/root_native_terminal_check.json').read_text())
    assert terminal['native_python_count']==0 and result['model_before_sha256']==result['model_after_sha256']
    assert support.sha(r/'traces.pt')==result['traces_sha256']
    d=torch.load(r/'traces.pt',map_location='cpu',weights_only=True);modes=('normal','zeroVP','swapVP','zeroPrompt','swapPrompt','allZero')
    metrics={};deltas={}
    for mode in modes:
        metrics[mode]={name:combine([scores(t[mode+'_'+name],t[name+'_ids']) for t in d['dev']]) for name in ('teacher','own')}
        a=metrics[mode]['teacher']['NLL'];deltas[mode]=a-metrics['normal']['teacher']['NLL']
    direct=combine([scores(t['direct_logits'],t['teacher_ids']) for t in d['dev']])
    agreements={}
    for mode in modes:
        diffs=[];matches=0;count=0
        for t in d['dev']:
            a=t['normal_own'].double();b=t[mode+'_own'].double();p=a.softmax(-1)
            diffs.append((p*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1).flatten())
            matches+=int((a.argmax(-1)==b.argmax(-1)).sum());count+=a.shape[0]*a.shape[1]
        agreements[mode]={'own_prefix_KL_from_normal':float(torch.cat(diffs).mean()),'own_prefix_argmax_agreement':matches/count}
    qp=[];margins=[];ratios=[];ngram_count=0
    for t in d['dev']:
        logits=t['normal_own'].double();probs=logits.softmax(-1)
        for row,ids in enumerate(t['own_ids']):
            raw=bytes(i for i in ids.tolist() if i<256);words=list(re.finditer(rb'[A-Za-z]+',raw));seen=set();negative=[]
            for i,word in enumerate(words):
                if i<3:continue
                key=tuple(w.group().lower() for w in words[i-3:i+1])
                if key in seen:negative.append(word)
                seen.add(key)
            for word in negative:
                first=word.start()-int(word.start()>0 and raw[word.start()-1]==32);end=word.end()
                q=math.exp(sum(float(probs[row,j,int(ids[j])].clamp_min(1e-300).log()) for j in range(first,end)))
                qp.append(q);ratios.append(len(raw)/len(negative)*q/max(1-q,1e-12))
                j=end-1;token=int(ids[j]);v=logits[row,j].clone();selected=float(v[token]);v[token]=-float('inf');margins.append(selected-float(v.max()))
            ngram_count+=len(negative)
    q=torch.tensor(qp,dtype=torch.float64);ratio=torch.tensor(ratios,dtype=torch.float64);margin=torch.tensor(margins)
    train=d['train_reconstruction_only'];reconstruction={k:scores(train[k],train['target_ids']) for k in ('freeVP','cleanCodec_ORACLE_train_only')}
    reconstruction={k:{**v,'NLL':v['nll_sum']/v['symbols'],'accuracy':v['correct']/v['symbols']} for k,v in reconstruction.items()}
    output={'status':'DECODER_BOUNDARY_AND_REPETITION_CALCULATED','actual_model_calls':140,'optimizer_steps':0,
        'actual_run_seconds':result['seconds'],'source_unchanged':True,'dev_rows':16,'train_reconstruction_rows':4,
        'conditioning_metrics':metrics,'teacher_NLL_delta_from_normal':deltas,'own_prefix_distributions':agreements,
        'freeVP_pointwise_codec_readout_descriptive':direct,'train_clean_codec_ORACLE_diagnostic_only':reconstruction,
        'repeated_word4gram_negative_spans':ngram_count,'repeated_word_byte_span_probability_mean':float(q.mean()),
        'repeated_word_byte_span_probability_median':float(q.median()),
        'word_span_unlikelihood_gradient_vs_positive_CE_at_same_bytes':{
            str(l):{'median':float((ratio*l).median()),'mean':float((ratio*l).mean())} for l in (.1,.25,.5,1,2,4)},
        'gradient_ratio_formula':'lambda * active_symbols/negative_span_count * q/(1-q); compares per-logit chosen-byte derivatives on the fixed observed own-prefix; not an actual training update.',
        'counterfactual_final_byte_penalty_only':{str(a):float((margin<a).float().mean()) for a in (.5,1,2,4,8,16)},
        'penalty_warning':'Final-byte scores calibrate margins only; deploying them alone can turn the into misspellings. Train word/span negatives with normal CE; do not punish all repeated UTF8 bytes.',
        'limitations':['small already-observed16prompt panel, not semantic benchmark','swapping is within each four-row batch and is an intervention, not independent causal proof','pointwise byte codec readout is not an explicit semantic score','clean codec train-only oracle is capacity/interface diagnostic, not generation performance','no new free rollout or sampling done in this trace','root directly saw owner alive but worker finished before root direct live observation; owned supervisor CIM identity verified worker'],
        'semantic_history':'Typed supplied-role relation scores preserved. V4 predicted-heads all3correct23/32 had3realizationerrors; novelwording had2/32correct3heads. Current V7 has no explicit V4semanticreader.',
        'E2_final_accessed':False,'additional_CPU_model_forwards':0}
    with (r/'root_result.json').open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=True))
if __name__=='__main__':main()
