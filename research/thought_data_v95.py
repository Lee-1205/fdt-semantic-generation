"""English input->answer task from existing V5 TRAIN/split, no new facts.

Input features ONLY id/prompt. Fact/role/label/answer metadata stays loss-side.
Authored comparisons/negation, already observed development, not broad NLU.
"""
import collections,hashlib,itertools,json,random
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer_from_finite_worlds(t):
    f=t['source_fact'];outcomes=set()
    for a,b in itertools.product(range(3),repeat=2):
        okay={'greater':a>b,'less':a<b,'not greater':a<=b,'not less':a>=b}[f['operator']]
        if okay:
            q,r=(b,a) if t['reverse'] else (a,b)
            outcomes.add('greater' if q>r else 'less' if q<r else 'equal')
    label=next(iter(outcomes)) if len(outcomes)==1 else 'unknown'
    q,r=(f['b'],f['a']) if t['reverse'] else (f['a'],f['b'])
    answer='It is unknown.' if label=='unknown' else f'{q} is {"larger" if label=="greater" else "smaller"} than {r}.'
    return label,answer


def build(source):
    features=json.loads((source/'features.json').read_text(encoding='utf-8-sig'))
    targets=json.loads((source/'targets.json').read_text(encoding='utf-8-sig'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8-sig'))
    for name,expected in manifest['files'].items():
        if sha(source/name)!=expected:raise ValueError('preserved V5 input SHA')
    grouped=collections.defaultdict(list)
    for index,t in enumerate(targets['train']):
        if t['language']=='en':grouped[(t['source_fact']['key'],t['family'],t['pair_key'])].append(index)
    strata=collections.defaultdict(list)
    for (fact,family,key),indices in grouped.items():
        if len(indices)!=2:raise ValueError('complete reverse pair')
        strata[(fact,family)].append((key,indices))
    chosen=[]
    for (fact,family),pairs in sorted(strata.items()):
        is_negative=targets['train'][pairs[0][1][0]]['source_fact']['operator'].startswith('not ')
        chosen += sorted(pairs)[:1 if is_negative else 2]
    train_indices=sorted(i for _,pair in chosen for i in pair)
    result={}
    for split in ('train','dev_pair','dev_surface','dev_position'):
        indices=train_indices if split=='train' else [i for i,t in enumerate(targets[split]) if t['language']=='en']
        ff=[];tt=[];pairs=collections.defaultdict(list)
        for index in indices:
            f=features[split][index];t=targets[split][index]
            if set(f)!={'id','prompt'} or hashlib.sha256(f['prompt'].encode()).hexdigest()!=f['id']:raise ValueError('source-only feature schema')
            label,answer=answer_from_finite_worlds(t)
            if t['label']!=label or t['answer']!=answer:raise ValueError('independent finite entailment/answer')
            ff.append(dict(f));tt.append(dict(id=f['id'],answer=answer,label=label,
                source_fact=t['source_fact'],reverse=t['reverse'],family=t['family'],layout=t['layout'],pair_key=t['pair_key']))
            pairs[t['pair_key']].append(f['id'])
        if any(len(pair)!=2 for pair in pairs.values()):raise ValueError('no incomplete query-reversal pairs')
        result[split]=dict(features=ff,targets=tt,pairs=list(pairs.values()))
    if len(result['train']['features'])!=384 or any(len(result[k]['features'])!=32 for k in result if k!='train'):
        raise ValueError('fixed384TRAIN+three32DEV contract')
    if collections.Counter(t['label'] for t in result['train']['targets'])!=dict(greater=128,less=128,unknown=128):raise ValueError('three balanced row labels')
    train_groups={t['source_fact']['group'] for t in result['train']['targets']}
    if train_groups & {t['source_fact']['group'] for t in result['dev_pair']['targets']}:raise ValueError('inherited canonical pair split')
    for a,b in itertools.combinations(result,2):
        if {f['id'] for f in result[a]['features']} & {f['id'] for f in result[b]['features']}:raise ValueError('literal split overlap')
    orders=[]
    for epoch in range(4):
        pairs=[list(pair) for pair in result['train']['pairs']];random.Random(95113+epoch).shuffle(pairs);orders.append(pairs)
    return result,orders


def publish(source,out):
    data,orders=build(source);out.mkdir(exist_ok=False)
    write=lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    statistics={}
    for split,d in data.items():
        write(out/(split+'_features.json'),d['features']);write(out/(split+'_targets.json'),d['targets'])
        statistics[split]=dict(rows=len(d['features']),reverse_pairs=len(d['pairs']),
            distinct_fact_keys=len({t['source_fact']['key'] for t in d['targets']}),
            pair_groups=len({t['source_fact']['group'] for t in d['targets']}),
            labels=dict(collections.Counter(t['label'] for t in d['targets'])),
            input_UTF8bytes=sum(len(f['prompt'].encode()) for f in d['features']),
            answer_UTF8bytes=sum(len(t['answer'].encode()) for t in d['targets']),
            max_input_bytes=max(len(f['prompt'].encode()) for f in d['features']))
    write(out/'four_finite_orders.json',orders)
    answers={hashlib.sha256(t['answer'].encode()).hexdigest():t['answer'] for d in data.values() for t in d['targets']}
    write(out/'answers_loss_only.json',answers)
    files={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    manifest=dict(schema='fdt_thought_source_answer_data_v95',status='CPU_DATA_ONLY',source=str(source),
        source_pins={name:sha(source/name) for name in ('manifest.json','features.json','targets.json')},
        builder_SHA=sha(Path(__file__)),statistics=statistics,unique_answer_targets=len(answers),
        feature_keys=['id','prompt'],target_metadata_never_neural_features=True,files=files,
        finite_orders=4,train_rows_per_epoch=384,four_epoch_rows1536=True,
        fresh_rows_or_facts_authored=0,E2final_access=False,actual_model_calls=0,
        limits=['Existing observed A-H comparison/negation only; not broad language/reasoning.',
                'Dev_surface/dev_position intentionally share training facts; dev_pair inherits old held pair groups.',
                'Positive and negative fact view counts differ; balanced row labels do not guarantee equal gradients.',
                'Unknown constant baseline train1/3 and eachDEV1/2; report positive cases separately.'])
    write(out/'manifest.json',manifest);return manifest
