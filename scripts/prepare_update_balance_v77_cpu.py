"""Prepare fixed finite balanced exposure order and prospective grammar revision.

No new TRAIN prompts are synthesized. 1024 unique TRAIN rows are reused with
explicit repeat counts: signed once,unsigned six times,old OWT three times.
"""
import collections,copy,hashlib,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
OUT=ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1'
EVAL=ROOT/'research/artifacts/joint_quantity_eval_20261113_v77_cpu_v1'

def main():
 assert not OUT.exists() and not EVAL.exists()
 selection=io.load_json(ROOT/'configs/research_base_selection_v76_review_20261113.json')
 assert selection['selected_version']=='V75' and selection['selected_SHA']=='f01e851284df0bd971498ee709b81158abb4ce157bf7e19578c146614411c124'
 src=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1'
 rows=io.load_json(src/'train_features.json');targets=io.load_json(src/'train_targets.json')
 pools=collections.defaultdict(list)
 for i,t in enumerate(targets):pools[t.get('mode','OWT')].append(i)
 assert {k:len(v) for k,v in pools.items()}==dict(signed=768,unsigned=128,OWT=128)
 rng=random.Random(77113);queues={}
 for kind,repeats in [('signed',1),('unsigned',6),('OWT',3)]:
  seq=[]
  for epoch in range(repeats):
   part=list(pools[kind]);rng.shuffle(part);seq.extend(part)
  queues[kind]=iter(seq)
 orders=[]
 for cycle in range(384):
  for kind in ('unsigned','signed','OWT','unsigned','signed'):orders.append([next(queues[kind])])
 for q in queues.values():
  try:next(q);raise AssertionError('extra exposures')
  except StopIteration:pass
 multiplicity=collections.Counter(o[0] for o in orders)
 assert len(orders)==1920 and all(multiplicity[i]=={'signed':1,'unsigned':6,'OWT':3}[targets[i].get('mode','OWT')] for i in range(1024))
 headbytes=sum(len(rows[o[0]]['prompt'].encode()) for o in orders)
 critical=sum(sum(s['end']-s['start'] for s in targets[o[0]]['critical_spans']) for o in orders)
 envelope=torch.load(selection['selected_runtime'],map_location='cpu',weights_only=True)
 weights=envelope['runtime']['symbol_weights'];parent_report=envelope['runtime']['objective_report']
 assert weights.shape==(258,) and weights[256]==1 and weights[257]==0 and torch.isfinite(weights).all()
 OUT.mkdir()
 io.atomic_json(OUT/'orders.json',orders)
 io.atomic_json(OUT/'symbol_weight_source.json',dict(source=selection['selected_runtime'],source_SHA=selection['selected_SHA'],
  weights=weights.tolist(),parent_objective_report=parent_report,policy='frozen V75 TRAIN-derived symbol weights; no held labels used'))
 io.atomic_json(OUT/'manifest.json',dict(schema='update_balance_v77_CPU_policy',unique_TRAIN_rows=1024,
  teacher_unique_source_rows=1024,head_exposure_rows=1920,head_kind_rows=dict(unsigned=768,signed=768,OWT=384),
  per_source_repeats=dict(unsigned=6,signed=1,OWT=3),finite_cycle_order=['unsigned','signed','OWT','unsigned','signed'],cycles=384,
  teacher_source_UTF8_bytes=sum(len(r['prompt'].encode()) for r in rows),head_source_UTF8_bytes=headbytes,head_critical_bytes=critical,
  original_OWT_physical_unique_rows=128,new_TRAIN_prompts=0,source_data=str(src),source_manifest_SHA=hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest(),
  source_features_SHA=hashlib.sha256((src/'train_features.json').read_bytes()).hexdigest(),source_targets_SHA=hashlib.sha256((src/'train_targets.json').read_bytes()).hexdigest(),
  fresh_optimizer_max_attempts=1920,old_budget_reuse=False,actual_GPU_model_training=False,E2_registry_access=0,downloads=0,
  limitations='Finite repeated source exposures, not1920newdocuments/oneepochof1024. Actual row frequency40/40/20%, not guarantee of parameter gradient mass; clipping/Adam intervene. Same TRAIN prompts as observed V76, V75 chosen parent.'))
 # Correct only a known grammar defect. Preserve every original64 source and
 # score in its old immutable artifact/run.32prompts remain exactly unchanged.
 ed=ROOT/'research/artifacts/joint_quantity_eval_20261113_v76_cpu_v1';oldf=io.load_json(ed/'features.json');oldt=io.load_json(ed/'targets.json')
 ef=[];et=[];changed=0
 names=['Alaric','Blythe','Cyrene','Denzel']
 for f,t in zip(oldf,oldt):
  text=f['prompt'].replace("'s a ","'s ");changed+=text!=f['prompt']
  ident='eval77:'+hashlib.sha256(text.encode()).hexdigest()[:24];label=copy.deepcopy(t)
  label.update(id=ident,original_V76_id=t['id'],original_V76_prompt_SHA=hashlib.sha256(f['prompt'].encode()).hexdigest(),
   prompt_changed_from_V76=text!=f['prompt'],clause=text[t['clause_start']:],EOS_position=len(text.encode())%16)
  for key,value in [('A',t['primary_text']),('B',t['secondary_text'])]:
   expected=value.removesuffix(' copper washers')
   matches=list(re.finditer(r'(?<![A-Za-z0-9_])'+re.escape(expected)+r'(?![A-Za-z0-9_])',text));assert len(matches)==1
   label[key+'_span']=dict(start=matches[0].start(),end=matches[0].end(),text=expected)
  ns=[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b',text)]
  label['critical_spans']=complete_expression_spans(text,ns)
  ef.append(dict(id=ident,prompt=text));et.append(label)
 assert changed==32 and not any("'s a " in f['prompt'] for f in ef)
 assert not any(f['prompt'] in r['prompt'] or r['prompt'] in f['prompt'] for f in ef for r in rows)
 EVAL.mkdir()
 for name,v in [('features',ef),('targets',et)]:io.atomic_json(EVAL/(name+'.json'),v)
 io.atomic_json(EVAL/'manifest.json',dict(schema='joint_quantity_v77_grammar_revision',rows=64,changed_surface_rows=32,unchanged_surface_rows=32,
  original64source_and_scores_preserved=True,same64facts_and_names_previously_observed=True,new_blind_benchmark=False,NN_calls=0,
  source_UTF8_bytes=sum(len(f['prompt'].encode()) for f in ef),parent_manifest_SHA=hashlib.sha256((ed/'manifest.json').read_bytes()).hexdigest(),
  limitation='Known grammar revision after V76reading;32sameprompts+32editedprompts/samefacts/names;bothretention/generalizationlimitsreported,not64newblindcases.'))
 for dest in (OUT,EVAL):io.atomic_json(dest/'builder_SHA.json',dict(file=str(Path(__file__)),SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print(json.dumps(dict(status='V77_BALANCED_EXPOSURE_AND_GRAMMAR_CPU_DATA_WRITTEN',head_rows=1920,head_bytes=headbytes,head_critical_bytes=critical,changed_eval=32,CUDA_initialized=torch.cuda.is_initialized())))
 assert not torch.cuda.is_initialized()
if __name__=='__main__':main()
