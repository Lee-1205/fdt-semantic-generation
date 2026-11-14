"""Small usable typed-premise console; checkpoint checks remain in runtime.py."""
import argparse,json,sys
from data import TRAIN_PHRASES,HELDOUT_PHRASES
from runtime import Assistant,KINDS

def parse_request(lines,subject,object_):
    names=[];facts=[]
    for line in lines.splitlines():
        if not line.strip():continue
        fields=[x.strip() for x in line.split('|')]
        if len(fields)!=3:raise ValueError('한 줄마다 주체 | 비교 문구 | 대상 형식을 사용하세요.')
        s,phrase,o=fields
        kinds=[i for i in range(4) if phrase in TRAIN_PHRASES[i]+HELDOUT_PHRASES[i]]
        if len(kinds)!=1:raise ValueError('지원 문구: '+', '.join(p for group in TRAIN_PHRASES for p in group))
        for name in (s,o):
            if name not in names:names.append(name)
        facts.append({'subject':s,'object':o,'kind':KINDS[kinds[0]],'phrase':phrase})
    for name in (subject,object_):
        if name not in names:names.append(name)
    return {'entities':names,'facts':facts,'query':{'subject':subject,'object':object_}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();request=json.load(sys.stdin)
    if 'lines' in request:request=parse_request(request['lines'],request['subject'],request['object'])
    result=Assistant(a.checkpoint).ask(request,a.seed)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
