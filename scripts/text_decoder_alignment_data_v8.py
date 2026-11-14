"""Deterministic train-only pseudo word repetitions, separate from NN features."""
import collections,hashlib,json,re
from pathlib import Path

def build(artifact):
    artifact=Path(artifact);documents=json.loads((artifact/'documents.json').read_text(encoding='utf-8'))
    features=json.loads((artifact/'features.json').read_text(encoding='utf-8'))
    counts=collections.Counter(w.lower() for d in documents['train'] for w in re.findall(r'[A-Za-z]+',d['text']) if 2<=len(w)<=4)
    motifs=[w for w,n in sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:8]]
    rows=[]
    for row in features['train']:
        key=int(hashlib.sha256(row['id'].encode()).hexdigest()[:12],16);word=motifs[key%len(motifs)]
        text=((' '+word)*130).encode()[:129];words=list(re.finditer(rb'[A-Za-z]+',text));seen=set();spans=[]
        for i,w in enumerate(words):
            if i<3:continue
            gram=tuple(x.group() for x in words[i-3:i+1])
            if gram in seen:spans.append([max(0,w.start()-1),w.end()])
            seen.add(gram)
        assert spans and all(0<=a<b<=129 for a,b in spans)
        rows.append({'id':row['id'],'motif':word,'negative_ids':list(text),'negative_spans':spans})
    return {'schema':'fdt_train_only_repetition_v8','source_train_document_count':len(documents['train']),
        'source_train_rows':len(features['train']),'motifs':{w:counts[w] for w in motifs},'rows':rows,
        'raw_features_unchanged':True,'dev_outputs_used_to_build_negatives':False,'E2_final_accessed':False,
        'interpretation':'Pseudo repetitive words from training-only frequency counts; negative loss targets, never inference rules or positive gold continuation.'}
