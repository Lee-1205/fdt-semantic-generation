"""History first, then source-backed alternatives, then saved-data calculations."""
from collections import defaultdict
import hashlib,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    workspace=ROOT.parent
    history=[ROOT/'docs/FDT_QUERYCOUNT_PARTIAL9_RESULT_20261027_KR.md']+[workspace/'outputs'/name for name in (
        'ASTRA_DOCUMENT_TRAINING_RESULT_20260926_KR.md','ASTRA_ANSWER_SIGNAL_RESULT_20260930_KR.md',
        'FDT_TEXT_CONTEXT_DECODER_RESULT_20261028_KR.md','FDT_TEXT_SPAN_ALIGNMENT_RESULT_20261028_KR.md')]
    for path in history:path.read_text(encoding='utf-8')
    corpus=ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu'
    targets=read(corpus/'targets.json');run=ROOT/'runs/text_language_continuation_20261028_v6'
    report={};perdoc={}
    for phase in ('baseline','final'):
        data=torch.load(run/(phase+'_NLL_logits.pt'),map_location='cpu',weights_only=True)
        logits=data['logits'].double();ids=data['target_ids'];active=ids!=257
        nll=logits.logsumexp(-1)-logits.gather(-1,ids[...,None]).squeeze(-1)
        pred=logits.argmax(-1);ends=active&(ids==256);content=active&~ends
        doc=defaultdict(lambda:[0.,0])
        for i,t in enumerate(targets['dev']):
            d=doc[t['document_id']];d[0]+=float(nll[i,active[i]].sum());d[1]+=int(active[i].sum())
        perdoc[phase]={key:value[0]/value[1] for key,value in doc.items()}
        probabilities=logits.softmax(-1);p_end=probabilities[...,256][ends]
        sorted_prob,indices=probabilities[ends].sort(-1,descending=True)
        before=sorted_prob.cumsum(-1)-sorted_prob
        keep=before<.9
        eos_in_nucleus=((indices==256)&keep).any(-1)
        report[phase]={'NLL':float(nll[active].mean()),'accuracy':float((pred[active]==ids[active]).double().mean()),
            'macro_document_NLL':sum(perdoc[phase].values())/len(doc),
            'content_NLL':float(nll[content].mean()),'EOF_NLL':float(nll[ends].mean()),
            'EOF_TP':int(((pred==256)&ends).sum()),'EOF_FN':int(((pred!=256)&ends).sum()),
            'nonEOF_false_EOS':int(((pred==256)&content).sum()),'nonEOF_positions':int(content.sum()),
            'EOF_probability_median':float(p_end.median()),'EOF_probability_mean':float(p_end.mean()),
            'EOF_in_top_p_point9_under_gold_prefix':int(eos_in_nucleus.sum()),
            'top_p_calculation_limitation':'Gold-prefix distributions only. Not a free-generation sampling result.'}
    improved=sum(perdoc['final'][key]<value for key,value in perdoc['baseline'].items())
    delta=report['final']['NLL']-report['baseline']['NLL']
    train=targets['train'];bydoc=defaultdict(list)
    for i,t in enumerate(train):bydoc[t['document_id']].append(i)
    balanced=[]
    for key in sorted(bydoc):
        end=[i for i in bydoc[key] if train[i]['source_eof']]
        other=sorted([i for i in bydoc[key] if not train[i]['source_eof']],key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())
        assert len(end)==1 and other
        balanced.extend([end[0],other[0]])
    populations={'original_one_epoch':list(range(len(train))),'prospective_document_matched_end_nonend':balanced}
    accounting={}
    for name,indices in populations.items():
        symbols=sum(len(train[i]['target_ids']) for i in indices);eos=sum(train[i]['source_eof'] for i in indices)
        accounting[name]={'rows':len(indices),'documents':len({train[i]['document_id'] for i in indices}),
            'target_symbols':symbols,'target_utf8_bytes':symbols-eos,'EOF':eos,'EOF_fraction':eos/symbols,
            'weighted_EOF_mass_fraction':{str(w):w*eos/(symbols+(w-1)*eos) for w in (1,2,4,8,16)},
            'arithmetic_only_not_actual_gradient_share':True,'new_training_performed':False}
    short=[]
    import re
    for d in read(corpus/'documents.json')['train']:
        for sentence in re.split(r'(?<=[.!?])\s+',d['text']):
            size=len(sentence.encode('utf-8'))
            if 12<=size<=42:short.append((d['source_id'],sentence))
    output={'status':'HISTORY_RESEARCH_CALCULATION_COMPLETE','history_files':{str(p):sha(p) for p in history},
        'primary_sources':{'unlikelihood':'https://arxiv.org/abs/1908.04319','degeneration_nucleus':'https://arxiv.org/abs/1904.09751',
            'DITTO':'https://arxiv.org/abs/2206.02369','self_contrastive':'https://aclanthology.org/2023.findings-acl.431/'},
        'observed_relation_strength':{'primary':1.,'uniform8':1.,'old_role':1823/1824,'scope':'Typed supplied-role/fuzzy/basis observed panels; correlated positions. Preserve success; not equal to free natural text.'},
        'saved_logits':report,'documents_improved':improved,'documents_total':64,
        'byte_accuracy_gain_percentage_points':100*(report['final']['accuracy']-report['baseline']['accuracy']),
        'NLL_change_nats_per_byte_symbol':delta,'final_bits_per_byte_symbol':report['final']['NLL']/math.log(2),
        'uniform_258_symbol_NLL':math.log(258),'backpropagated_unique_target_bytes_this_epoch':383651,
        'scale_note':'Initial English-story curriculum, 512documents/~0.384MB gradient targets, not broad pretraining or convergence proof.',
        'EOS_accounting':accounting,'prospective_short_sentence_candidates':{'rows':len(short),'documents':len({x[0] for x in short}),
            'byte_length_12_to42':True,'segmentation':'Offline punctuation heuristic, not validated full sentence semantics or runtime parser.',
            'DITTO_reproduction_claimed':False},
        'whole_model_failure_conclusion_supported':False,'bounded_trial_target_shortfall_preserved':True,
        'additional_model_forwards':0,'new_GPU_training':False,'E2_final_accessed':False}
    path=workspace/'outputs/FDT_HISTORY_FIRST_REPETITION_CALC_20261028.json'
    with path.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('history_files','primary_sources')},ensure_ascii=True))
if __name__=='__main__':main()
