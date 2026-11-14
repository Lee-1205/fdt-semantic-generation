"""Bounded typed-premise use: candidate -> evidence check -> immutable commit."""
from dataclasses import dataclass
from collections import deque
import json,pathlib
import torch
from data import TRAIN_PHRASES,HELDOUT_PHRASES,LABELS,collate,oracle
from model import Config,FDT100

KINDS=('higher','lower','not_higher','not_lower')


@dataclass(frozen=True)
class Commit:
    subject:str
    object:str
    relation:str
    status:str
    source_ids:tuple
    source_texts:tuple
    confidence:float


def realize(commit):
    if commit.status=='SOURCE_RELATIVE_VERIFIED':
        verb='더 큽니다' if commit.relation=='GREATER' else '더 작습니다'
        return f'{commit.subject}는 {commit.object}보다 {verb}. (제공된 가정의 근거와 일치)'
    if commit.status=='CONFLICT':return '제공된 비교 가정이 서로 모순됩니다.'
    return f'{commit.subject}와 {commit.object}의 관계를 확정할 근거가 부족합니다.'


def path_proof(n,facts,a,b):
    adj=[[] for _ in range(n)]
    for i,(s,o,k,_) in enumerate(facts):
        if k==0:adj[s].append((o,i))
        elif k==1:adj[o].append((s,i))
    todo=deque([(a,())]);seen={a}
    while todo:
        node,path=todo.popleft()
        for nxt,source in adj[node]:
            new=path+(source,)
            if nxt==b:return new
            if nxt not in seen:seen.add(nxt);todo.append((nxt,new))
    return None


def conflict_proof(n,facts):
    for node in range(n):
        proof=path_proof(n,facts,node,node)
        if proof:return proof
    for i,(s,o,k,_) in enumerate(facts):
        if k>=2:
            proof=path_proof(n,facts,s,o) if k==2 else path_proof(n,facts,o,s)
            if proof:return proof+(i,)
    return None


def validate_request(request):
    names=request['entities']
    if not isinstance(names,list) or not 2<=len(names)<=8 or len(set(names))!=len(names):raise ValueError('정확한 개체 이름2–8개가 필요합니다.')
    if any(not isinstance(x,str) or not x or len(x.encode())>128 for x in names):raise ValueError('개체 이름을 확인하세요.')
    if not 1<=len(request['facts'])<=12:raise ValueError('사실1–12개가 필요합니다.')
    facts=[];texts=[]
    for f in request['facts']:
        k=KINDS.index(f['kind']);phrase=f['phrase']
        if phrase not in TRAIN_PHRASES[k]+HELDOUT_PHRASES[k]:raise ValueError('지원 문구와 typed kind가 일치해야 합니다.')
        s,o=names.index(f['subject']),names.index(f['object'])
        if s==o:raise ValueError('자기 비교는 지원하지 않습니다.')
        facts.append((s,o,k,phrase));texts.append(f'{names[s]} | {phrase} | {names[o]}')
    q=request['query'];a,b=names.index(q['subject']),names.index(q['object'])
    if a==b:raise ValueError('서로 다른 대상을 선택하세요.')
    item={'n':len(names),'facts':facts,'queries':[(a,b)],'labels':[0],'hops':[0]}
    return names,item,texts


class Assistant:
    def __init__(self,checkpoint,device='cuda'):
        from train import digest_file,source_digest
        path=pathlib.Path(checkpoint);meta=json.loads(path.with_suffix('.json').read_text())
        if path.name!=meta['file'] or digest_file(path)!=meta['sha256']:raise ValueError('runtime 무결성 오류')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if not saved['training_complete'] or not all(g['passed'] for g in saved['gates']):raise ValueError('학습 단계 검증 미완료')
        if saved['contract']['source']!=source_digest():raise ValueError('학습 소스와 runtime 소스 불일치')
        self.device=device;self.model=FDT100(Config(**saved['config'])).to(device)
        self.model.load_state_dict(saved['model']);self.model.eval();del saved

    @torch.no_grad()
    def ask(self,request,seed=17):
        names,item,texts=validate_request(request);a,b=item['queries'][0]
        feat,_,_=collate([item],self.model.c,self.device)
        generator=torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(device_type=self.device,dtype=torch.float16,enabled=self.device=='cuda'):
            enc=self.model.encode(feat);noise=torch.randn(enc['init'].shape,device=self.device,generator=generator)
            z=self.model.rollout(enc,noise);logits=self.model.readout(z)[0,0];prob=logits.softmax(-1)
        candidate=int(prob.argmax());contradiction=conflict_proof(len(names),item['facts']);proof=None
        if contradiction is not None:status='CONFLICT';proof=contradiction
        elif candidate in (1,2):
            proof=path_proof(len(names),item['facts'],a,b) if candidate==1 else path_proof(len(names),item['facts'],b,a)
            status='SOURCE_RELATIVE_VERIFIED' if proof else 'UNRESOLVED'
        else:status='UNRESOLVED'
        sources=tuple(proof or ())
        commit=Commit(names[a],names[b],LABELS[candidate],status,sources,tuple(texts[i] for i in sources),float(prob[candidate]))
        return {'text':realize(commit),'status':status,'neural_candidate':LABELS[candidate],
                'neural_confidence_not_truth':commit.confidence,'source_ids':list(sources),'source_texts':list(commit.source_texts),
                'internal_natural_language_tokens':0,'persistent_fact_writes':0}
