"""CPU-only question-span/pair metadata; never imported by neural inference.

Preserves the exact V3 pool and evaluation prompts. No new graph/holdout and no
pair-completion expansion. Regex is annotation tooling, not an inference parser.
"""
from __future__ import annotations
import hashlib,json,re,random
from pathlib import Path
from collections import Counter,defaultdict
import torch
from research.text_semantic_targets_v3 import targets
from research.text_generation_data_v1 import check
from research.text_diffusion_v1 import pack_prompts,pack_answers

PARENT_SHA='a5e1c0e774c89a22dde794d2cffd138926ae961fd8cde592bf466d25d63a71b3'
SOURCE_SHA={
 'train.json':'2b2275811c673da932a2b640c309ba464c5773bf025ccc33a4afe9916f8644ee',
 'train_probe.json':'169270bba3fa089f4a78d21fef370c85b2ec3ec807eef872d8e647ca938b372f',
 'dev_pairs.json':'5511afb0832685fe7a0248dd4c150c553fe58d65aa628e00a446d4e7bb95d6c5',
 'dev_wording.json':'cf6e2e91f4f314b64ddc596ce024205989343563b5adfecaef111845c9519b52'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def query_spans(row):
 text=row['prompt']
 patterns=(r'How does ([A-H]) compare with ([A-H])\?',r'Compare ([A-H]) and ([A-H])\.') if row['language']=='en' else (r'([A-H])와 ([A-H])(?:의 관계는\?|를 비교하라\.)',)
 matches=[m for pattern in patterns for m in re.finditer(pattern,text)]
 if len(matches)!=1:raise ValueError('exactly one annotation query required')
 match=matches[0];spans=[]
 for group in (1,2):
  a,b=match.span(group);start=len(text[:a].encode('utf-8'));end=len(text[:b].encode('utf-8'))
  if end-start!=1 or text.encode()[start:end]!=match.group(group).encode():raise ValueError('UTF8 span alignment')
  spans.append((start,end))
 labels=targets(row)
 if [text.encode()[a]-65 for a,b in spans]!=labels[:2]:raise ValueError('span/entity disagreement')
 # Only the two query occurrences are blanked; the premise bytes stay exact.
 a,b=match.span(1);c,d=match.span(2)
 normalized=text[:a]+'{QUERY_SUBJECT}'+text[b:c]+'{QUERY_OBJECT}'+text[d:]
 pair_key=hashlib.sha256((normalized+'|'+''.join(sorted(match.groups()))).encode()).hexdigest()
 swapped=text[:a]+match.group(2)+text[b:c]+match.group(1)+text[d:]
 return {'head_labels':labels,'span_starts':[a for a,b in spans],'span_ends':[b for a,b in spans],
         'pair_key':pair_key,'swapped_prompt_id':hashlib.sha256(swapped.encode()).hexdigest(),
         'answer':row['answer'],'language':row['language'],'class':row['label']}

def build(data,selected):
 check(data)
 if len(selected)!=1024 or len(set(selected))!=1024 or min(selected)<0 or max(selected)>=len(data['train']):raise ValueError('exact inherited unique pool')
 panels={'train':[data['train'][i] for i in selected],**{k:data[k] for k in ('train_probe','dev_pairs','dev_wording')}}
 features={};labels={};pairs={};stats={}
 for name,rows in panels.items():
  features[name]=[{'id':r['id'],'prompt':r['prompt']} for r in rows]
  labels[name]=[query_spans(r) for r in rows]
  groups=defaultdict(list)
  for i,label in enumerate(labels[name]):groups[label['pair_key']].append(i)
  available=[]
  for key,indices in groups.items():
   if len(indices)>2:raise ValueError('ambiguous reverse-pair grouping')
   if len(indices)!=2:continue
   i,j=indices;a,b=labels[name][i],labels[name][j]
   if a['swapped_prompt_id']!=features[name][j]['id'] or b['swapped_prompt_id']!=features[name][i]['id']:raise ValueError('premise changed in paired view')
   if a['head_labels'][:2]!=b['head_labels'][:2][::-1]:raise ValueError('reverse roles')
   opposite={0:1,1:0,2:2}
   if opposite[a['head_labels'][2]]!=b['head_labels'][2]:raise ValueError('reverse relation')
   available.append({'first':i,'second':j,'pair_key':key})
  pairs[name]=available
  tokens=pack_prompts([r['prompt'] for r in rows]);pack_answers([r['answer'] for r in rows])
  candidates=((tokens>=65)&(tokens<=72))
  for i,label in enumerate(labels[name]):
   if any(not candidates[i,position] for position in label['span_starts']):raise ValueError('pointer target outside raw candidate bytes')
  stats[name]={'rows':len(rows),'complete_reverse_pairs':len(available),
   'unpaired_rows':len(rows)-2*len(available),
   'support':dict(Counter(r['language']+'|'+r['label'] for r in rows)),
   'prompt_utf8_bytes':sum(len(r['prompt'].encode()) for r in rows),
   'candidate_byte_positions':int(candidates.sum()),
   'max_prompt_bytes':max(len(r['prompt'].encode()) for r in rows)}
 return {'features':features,'targets':labels,'pairs':pairs},stats

def finite_pair_orders(pairs,row_count,epochs,seed,batch_size=8):
 if row_count%batch_size or batch_size%2:raise ValueError('fixed full even batches required')
 used=[]
 for pair in pairs:used.extend((pair['first'],pair['second']))
 if len(set(used))!=len(used) or any(i<0 or i>=row_count for i in used):raise ValueError('overlapping/invalid pair membership')
 paired=[(p['first'],p['second']) for p in pairs];unpaired=sorted(set(range(row_count))-set(used))
 rng=random.Random(seed);order=[]
 for _ in range(epochs):
  groups=paired.copy();single=unpaired.copy();rng.shuffle(groups);rng.shuffle(single)
  epoch=[i for group in groups for i in group]+single
  batches=[epoch[i:i+batch_size] for i in range(0,row_count,batch_size)];rng.shuffle(batches)
  order.extend(i for batch in batches for i in batch)
 return order

def local_pairs(indices,pairs):
 lookup={row:i for i,row in enumerate(indices)}
 if len(lookup)!=len(indices):raise ValueError('repeated row in one batch')
 return [(lookup[p['first']],lookup[p['second']]) for p in pairs if p['first'] in lookup and p['second'] in lookup]

def write_new(root,directory):
 root=Path(root);directory=Path(directory)
 if directory.exists():raise FileExistsError('artifact overwrite forbidden')
 data_dir=root/'research/artifacts/text_generation_20261028_v1'
 parent=root/'runs/text_semantic_alignment_20261028_v3/checkpoints/state_0512.pt'
 if sha(parent)!=PARENT_SHA:raise ValueError('parent identity')
 for name,h in SOURCE_SHA.items():
  if sha(data_dir/name)!=h:raise ValueError('source data changed '+name)
 source={Path(name).stem:json.loads((data_dir/name).read_text(encoding='utf-8')) for name in SOURCE_SHA}
 payload=torch.load(parent,map_location='cpu',weights_only=True)
 selected=payload['selected_train_indices'];del payload
 artifact,stats=build(source,selected)
 directory.mkdir(parents=True,exist_ok=False)
 for name,value in artifact.items():(directory/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (directory/'selected_train_indices.json').write_text(json.dumps(selected)+'\n',encoding='utf-8')
 for phase,seed in (('reader',31),('realizer',32)):
  order=finite_pair_orders(artifact['pairs']['train'],1024,4,seed)
  (directory/(phase+'_order.json')).write_text(json.dumps(order)+'\n',encoding='utf-8')
 manifest={'schema':'fdt_text_span_data_v4','source_files':SOURCE_SHA,'parent_sha256':PARENT_SHA,
  'builder_sha256':sha(__file__),'statistics':stats,'new_rows':0,'pool_preserved':True,
  'feature_keys':['id','prompt'],'gold_span_runtime_inputs':False,'E2_final_accessed':False,
  'model_forward_calls':0,'optimizer_calls':0,'evaluation_expanded':False,
  'files':{p.name:sha(p) for p in sorted(directory.glob('*.json'))}}
 payload=json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
 manifest['self_excluding_payload_sha256']=hashlib.sha256(payload).hexdigest()
 path=directory/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return {'manifest_file_sha256':sha(path),'manifest':manifest}
