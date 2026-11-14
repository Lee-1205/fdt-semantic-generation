"""Saved-output English faithfulness review, never inference/training features.

Manual short-sentence judgments and restricted authored-ledger event comparison
are distinct from a general semantic metric. Unparseable paraphrases require
manual review; they must not be declared semantically wrong by this parser.
"""
import decimal,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.codec_commit_stop_v56 import reconstruct_commit_logits

FRAME=re.compile(r'In record (?P<record>[^,]+), (?P<actor>\S+) placed (?P<quantity>\S+) (?P<object>.+?) in the (?P<place>.+?) and left (?P<remaining>\S+) on the (?P<remaining_place>.+?)\. (?P<checker>\S+) checked the count (?P<order>before|after) moving the container to the (?P<destination>.+?)\. The (?P<cabinet>.+?) stayed (?P<state>closed|open); (?P<polarity>neither|either|one|both) person(?:s)? put the remaining objects in that cabinet\.')

def numeric(value):
    try:return str(decimal.Decimal(value.replace(',','')).normalize())
    except decimal.InvalidOperation:return 'UNRESOLVED_LITERAL:'+value

def fields(text):
    m=FRAME.fullmatch(text.strip())
    if not m:return None
    d=m.groupdict()
    for k,v in d.items():d[k]=' '.join(v.split()).casefold()
    for k in ('quantity','remaining','record'):d[k]=numeric(d[k])
    return d

def raw_review(run,p,rows,saved):
    n=ok=0;NLL=0.;decoded_matches=decoded_bytes=0;per=[]
    for i,(row,r) in enumerate(zip(rows,saved)):
        packet=torch.load(run/f'raw_retained_English_{i:02d}.pt',weights_only=True,map_location='cpu');z,y=packet['logits'],packet['targets'];b=row['prompt'].encode()
        expected=torch.full_like(y.flatten(),257);expected[:len(b)]=torch.tensor(list(b));expected[len(b)]=256;assert torch.equal(expected,y.flatten())
        assert packet['Exact_entries']==0 and packet['Exact_policy']==p['exact_policy'] and packet['fuzzy_commit'].count_nonzero()
        d=reconstruct_commit_logits(z);assert d['text']==r['output'] and d['ids']==r['ids'] and row['id']==r['id'] and row['prompt']==r['source']
        count=int((y!=257).sum());correct=int(((z.argmax(-1)==y)&(y!=257)).sum());ce=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'))
        assert correct==r['correct'] and math.isclose(ce,r['NLL_sum'],rel_tol=1e-6,abs_tol=4*torch.finfo(torch.float32).eps*count+1e-5)
        output=d['text'].encode();decoded_matches+=sum(x==yy for x,yy in zip(b,output));decoded_bytes+=max(len(b),len(output));n+=count;ok+=correct;NLL+=ce
        per.append(dict(index=i,id=row['id'],exact=r['exact'],source_bytes=len(b),raw_errors=count-correct,EOS=d['EOS'],UTF8=d['UTF8']))
    return dict(rows=len(rows),NLL=NLL/n,byte_EOS_accuracy=ok/n,decoded_byte_positional_accuracy=decoded_matches/decoded_bytes,
        exact_auxiliary=sum(r['exact'] for r in saved),per_row=per)

def main():
    torch.set_num_threads(4);run=ROOT/'runs/hard_fuzzy_cycle_20261112_v67';p=io.load_json(run/'plan.json');result=io.load_json(run/'result.json')
    assert io.load_json(run/'ROOT_terminal_native.json')['nativePythonCount']==0
    rows=io.load_json(Path(p['data'])/'retained_English.json');saved=io.load_json(run/'retained_English_reconstructions.json');assert len(rows)==len(saved)==36
    metrics=raw_review(run,p,rows,saved);assert metrics['byte_EOS_accuracy']==result['retained_English']['byte_EOS_accuracy']
    short=[];events=[];unparseable=[];meta=[]
    pair_topics=['giving actors and keeping','gate/window state and order','movement time and direction','basket quantities','arrival/event order',
        'negative versus positive visitor','lending actors/recipient','lighter versus heavier','payment and pen quantity','letter location alternatives','repair order and battery action','only-second versus both photographs']
    for i,r in enumerate(saved[:24]):
        assert rows[i]['kind']=='single_sentence'
        if r['exact']:category='preserved';note='All supplied source propositions retained verbatim.'
        elif i in (4,5):
            assert r['output']==r['source'].replace('At 6:45','A  6:45',1)
            category='preserved_with_minor_form_issue';note='Missing t in At; 6:45, Tuesday, Alton, three blue folders, source/destination and moved action unchanged. Human judgment, not general automatic equivalence.'
        elif i==23:
            assert r['output']==r['source']+' ';category='preserved_whitespace_only';note='Trailing space has no proposition-level effect.'
        else:raise AssertionError('new short-output difference requires manual meaning review')
        short.append(dict(index=i,id=r['id'],topic=pair_topics[i//2],judgment=category,actor_relation_polarity_quantity_time_and_completeness='no meaning change observed in manual comparison',note=note,source=r['source'],output=r['output']))
    for i,r in enumerate(saved[24:],24):
        a=r['source'].split('\n\n');b=r['output'].split('\n\n')
        assert len(a)==len(b) and a[0]==b[0]
        for j,(source,output) in enumerate(zip(a[1:],b[1:]),1):
            gold,pred=fields(source),fields(output)
            if gold is None or pred is None:
                unparseable.append(dict(row=i,ordinal=j,source=source,output=output,judgment='unscorable_requires_manual_review'));continue
            changed={k:dict(source=gold[k],output=pred[k]) for k in gold if gold[k]!=pred[k]}
            core={k:v for k,v in changed.items() if k!='record'}
            events.append(dict(row=i,id=r['id'],ordinal=j,source_fields=gold,output_fields=pred,core_preserved=not core,
                changed_core_fields=core,record_reference_preserved='record' not in changed,source=source,output=output))
            if 'record' in changed:meta.append(dict(row=i,ordinal=j,**changed['record']))
    summary=dict(status='ENGLISH_SOURCE_FAITHFULNESS_V67_CPU_REVIEW',priority='English proposition preservation PRIMARY; byte/whole-string equality auxiliary; Korean deferred',
        short_sentences=dict(rows=24,manual_meaning_preserved=24,verbatim_auxiliary=sum(r['exact'] for r in saved[:24]),minor_form_issues=2,whitespace_only=1,
            scope='unblinded ROOT source-output judgments on restricted observed authored English; not general semantic accuracy'),
        long_contexts=dict(rows=12,event_bundles=len(events),core_event_bundles_preserved=sum(e['core_preserved'] for e in events),
            unparseable_requires_manual=len(unparseable),entity_relation_quantity_polarity_or_order_risks=[dict(row=e['row'],ordinal=e['ordinal'],changes=e['changed_core_fields']) for e in events if not e['core_preserved']],
            record_reference_errors=meta,scope='authored ledger grammar only; repeated facts correlated; do not inflate into universal semantic success'),
        auxiliary_raw_and_decoded_metrics=metrics,Exact_entries=0,copy_bytes=0,denoise_calls=0,extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
        limitations='no FActScore/FactCC model executed; regex only offline scoring, never runtime features; paraphrases outside grammar need human review; internal abstract meaning not certified')
    io.atomic_json(run/'root_English_meaning_review.json',summary);io.atomic_json(run/'root_English_meaning_event_cards.json',dict(short=short,events=events,unparseable=unparseable))
    print(json.dumps({k:v for k,v in summary.items() if k!='auxiliary_raw_and_decoded_metrics'}));print(json.dumps({k:v for k,v in metrics.items() if k!='per_row'}))
if __name__=='__main__':main()
