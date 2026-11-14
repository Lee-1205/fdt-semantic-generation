"""Separate training targets for input facts and the requested output meaning."""
import copy,hashlib,json
from pathlib import Path
OPS=('greater','less','not greater','not less')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(features,targets):
    result=copy.deepcopy(targets)
    for panel,rows in features.items():
        for f,y in zip(rows,result[panel]):
            fact=y['source_fact'];raw=f['prompt'].encode('utf-8');phrase=fact['fact'].encode('utf-8')
            if raw.count(phrase)!=1:raise ValueError('one known premise in offline target construction')
            base=raw.index(phrase);a=ord(fact['a'])-65;b=ord(fact['b'])-65
            # Premise entity offsets derive only from the authored source fact.
            pa=base;pb=base+phrase.index(fact['b'].encode('ascii'),1)
            qs,qo,relation,lang=y['head_labels']
            y['input_labels']=[qs,qo,a,b,OPS.index(fact['operator']),lang]
            y['input_spans']=y['span_starts']+[pa,pb]
            y['output_labels']=[qs,qo,{0:1,1:2,2:0}[relation],lang]
            for position,label in zip(y['input_spans'],y['input_labels'][:4]):
                if raw[position]!=65+label:raise ValueError('offline pointer target alignment')
            if y['input_labels'][4]>=2 and y['output_labels'][2]!=0:raise ValueError('negation/equality preserves uncertainty')
    return result

def load(artifact):
    a=Path(artifact)
    f=json.loads((a/'features.json').read_text(encoding='utf-8'));y=json.loads((a/'targets.json').read_text(encoding='utf-8'))
    if any(set(row)!={'id','prompt'} for rows in f.values() for row in rows):raise ValueError('gold in features')
    return f,build(f,y)
