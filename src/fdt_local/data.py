"""Generated typed-premise tasks. Gold types/closure are supervision only."""
import hashlib,json,random
import torch
from torch.nn import functional as F

VERSION='variable-order-data-v0.3'
TRAIN_PHRASES=(('더 크다','크다','보다 크다','is greater than'),
               ('더 작다','작다','보다 작다','is less than'),
               ('크지 않다','더 크지 않다','is not greater than'),
               ('작지 않다','더 작지 않다','is not less than'))
HELDOUT_PHRASES=(('더 큰 값을 갖는다',),('더 작은 값을 갖는다',),
                 ('더 크다고 할 수 없다',),('더 작다고 할 수 없다',))
LABELS=('UNRESOLVED','GREATER','LESS','CONFLICT')


def graph_key(n,facts):
    typed=sorted(set((int(s),int(o),int(kind)) for s,o,kind,_ in facts))
    return hashlib.sha256(json.dumps([n,typed],separators=(',',':')).encode()).hexdigest()


def graph_partition(n,facts):
    bucket=int(graph_key(n,facts)[:12],16)%10
    return 'train' if bucket<8 else 'dev' if bucket==8 else 'test'


def oracle(n,facts,queries):
    reach=[[False]*n for _ in range(n)];distance=[[100]*n for _ in range(n)]
    negatives=[]
    for s,o,k,_ in facts:
        if k<2:
            a,b=(s,o) if k==0 else (o,s);reach[a][b]=True;distance[a][b]=1
        else:negatives.append((s,o,k))
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j]=reach[i][j] or (reach[i][k] and reach[k][j])
                distance[i][j]=min(distance[i][j],distance[i][k]+distance[k][j])
    global_conflict=any(reach[i][i] for i in range(n)) or any(
        reach[s][o] if k==2 else reach[o][s] for s,o,k in negatives)
    labels=[];hops=[]
    for s,o in queries:
        y=3 if global_conflict else 1 if reach[s][o] else 2 if reach[o][s] else 0
        labels.append(y);hops.append(min(distance[s][o],distance[o][s]) if y in (1,2) else 0)
    return labels,hops


def candidate(seed,n_range=(4,6),heldout=False):
    r=random.Random(seed);n=r.randint(*n_range);order=list(range(n));r.shuffle(order)
    mode=r.choices(('chain','sparse','conflict','negative'),weights=(35,30,20,15))[0]
    edges=[]
    if mode in ('chain','conflict'):
        edges=[(order[i],order[i+1]) for i in range(n-1)]
    else:
        for i in range(n):
            for j in range(i+1,n):
                if r.random()<.18:edges.append((order[i],order[j]))
    if mode=='conflict':edges.append((order[-1],order[0]))
    facts=[];lexicon=HELDOUT_PHRASES if heldout else TRAIN_PHRASES
    for s,o in edges:
        k=r.randrange(2)
        if k==1:s,o=o,s
        facts.append((s,o,k,r.choice(lexicon[k])))
    # Negative premises cannot create reverse positive facts.
    for _ in range(r.randint(1,3)):
        s,o=r.sample(range(n),2);k=r.choice((2,3))
        facts.append((s,o,k,r.choice(lexicon[k])))
    if not facts:
        s,o=r.sample(range(n),2);facts=[(s,o,2,r.choice(lexicon[2]))]
    r.shuffle(facts);facts=facts[:12]
    queries=[(s,o) for s in range(n) for o in range(n) if s!=o];r.shuffle(queries);queries=queries[:16]
    labels,hops=oracle(n,facts,queries)
    return {'n':n,'facts':facts,'queries':queries,'labels':labels,'hops':hops,
            'graph_key':graph_key(n,facts),'mode':mode}


def make_split(split,count,seed=301,heldout=False,n_range=(4,6)):
    if split not in ('train','dev','test'):raise ValueError('unknown split')
    out=[];seen=set();attempt=0
    while len(out)<count:
        item=candidate(seed+attempt,n_range,heldout);attempt+=1
        if graph_partition(item['n'],item['facts'])!=split or item['graph_key'] in seen:continue
        seen.add(item['graph_key']);out.append(item)
    return out


def collate(items,c,device='cpu'):
    b=len(items);tokens=torch.full((b,c.facts,c.phrase_bytes),256,dtype=torch.long)
    roles=torch.zeros(b,c.facts,16);fact_mask=torch.zeros(b,c.facts)
    queries=torch.zeros(b,c.slots,16);query_mask=torch.zeros(b,c.slots)
    ground=torch.zeros(b,c.facts,dtype=torch.long);labels=torch.zeros(b,c.slots,dtype=torch.long)
    hops=torch.zeros_like(labels);exposure=0
    for i,item in enumerate(items):
        if item['n']>8 or len(item['facts'])>c.facts:raise ValueError('capacity exceeded')
        for j,(s,o,k,phrase) in enumerate(item['facts']):
            raw=phrase.encode('utf-8')
            if len(raw)>c.phrase_bytes:raise ValueError('phrase bytes exceeded')
            tokens[i,j,:len(raw)]=torch.tensor(list(raw));exposure+=len(raw)
            roles[i,j,s]=1;roles[i,j,8+o]=1;fact_mask[i,j]=1;ground[i,j]=k
        for j,(s,o) in enumerate(item['queries']):
            queries[i,j,s]=1;queries[i,j,8+o]=1;query_mask[i,j]=1
            labels[i,j]=item['labels'][j];hops[i,j]=item['hops'][j]
    # Only these five tensors reach model encode. Supervision travels separately.
    features={k:v.to(device) for k,v in {'tokens':tokens,'roles':roles,'fact_mask':fact_mask,
                                      'queries':queries,'query_mask':query_mask}.items()}
    targets={k:v.to(device) for k,v in {'labels':labels,'ground':ground,'hops':hops}.items()}
    return features,targets,exposure


def manifest(splits):
    result={'version':VERSION,'external_corpus_tokens':0,'splits':{}}
    for name,items in splits.items():
        classes=[0]*4;ground=[0]*4;bytes_=0;hops={}
        for x in items:
            for y in x['labels']:classes[y]+=1
            for _,_,k,p in x['facts']:ground[k]+=1;bytes_+=len(p.encode())
            for h in x['hops']:hops[str(h)]=hops.get(str(h),0)+1
        total=sum(classes)
        result['splits'][name]={'graphs':len(items),'query_labels':classes,
            'class_ratio':[v/total for v in classes],'ground_labels':ground,
            'input_utf8_bytes':bytes_,'hops':hops,
            'graph_digest':hashlib.sha256(''.join(sorted(x['graph_key'] for x in items)).encode()).hexdigest()}
    result['fingerprint']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result
