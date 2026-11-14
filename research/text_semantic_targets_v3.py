"""Offline training/scoring metadata only; not neural inputs or runtime parsing."""
import json,re
from pathlib import Path

def targets(row):
    if row['language']=='en':
        m=re.search(r'How does ([A-H]) compare with ([A-H])\?',row['prompt'])
        if m: a,b=m.groups()
        else:
            m=re.search(r'Compare ([A-H]) and ([A-H])\.',row['prompt'])
            if not m:raise ValueError('English query target metadata')
            a,b=m.groups()
    else:
        m=re.search(r'([A-H])와 ([A-H])(?:의 관계는\?|를 비교하라\.)',row['prompt'])
        if not m:raise ValueError('Korean query target metadata')
        a,b=m.groups()
    if a==b:raise ValueError('unexpected self query')
    return [ord(a)-65,ord(b)-65,('greater','less','unknown').index(row['label']),('en','ko').index(row['language'])]

def build_metadata(data,selected):
    panels={'train':[data['train'][i] for i in selected],**{k:data[k] for k in ('train_probe','dev_pairs','dev_wording')}}
    result={name:[targets(row) for row in rows] for name,rows in panels.items()}
    for name,rows in panels.items():
        for row,t in zip(rows,result[name]):
            if row['label']=='unknown':continue
            a,b=chr(65+t[0]),chr(65+t[1])
            # Positive answers independently bind the exact query subject/object.
            if not row['answer'].startswith(a) or b not in row['answer']:raise ValueError('query/answer target mismatch')
    return result
