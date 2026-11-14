"""Bounded typed-premise use: candidate -> evidence check -> immutable commit."""
from dataclasses import dataclass
from collections import deque
import json,pathlib
import torch
from .data import TRAIN_PHRASES,HELDOUT_PHRASES,LABELS,collate,oracle
from .model import Config,FDT100

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

    def __post_init__(self):
        import math
        if any(not isinstance(value,str) for value in (self.subject,self.object,self.relation,self.status)):
            raise TypeError('Commit fields must be immutable strings')
        ids=tuple(self.source_ids);texts=tuple(self.source_texts)
        if any(type(value) is not int for value in ids) or any(not isinstance(value,str) for value in texts):
            raise TypeError('Commit source values must be int/string scalars')
        if len(ids)!=len(texts) or not math.isfinite(float(self.confidence)):
            raise ValueError('invalid Commit provenance/confidence')
        object.__setattr__(self,'source_ids',ids)
        object.__setattr__(self,'source_texts',texts)
        object.__setattr__(self,'confidence',float(self.confidence))


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


