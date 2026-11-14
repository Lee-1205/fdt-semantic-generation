"""ROOT sampler multiplicity/update cycles, frozen profile and grammar revision."""
import collections,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
D=ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1';E=ROOT/'research/artifacts/joint_quantity_eval_20261113_v77_cpu_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 torch.set_num_threads(2);m=io.load_json(D/'manifest.json');src=Path(m['source_data']);rows=io.load_json(src/'train_features.json');targets=io.load_json(src/'train_targets.json');orders=io.load_json(D/'orders.json')
 for f,key in [('manifest.json','source_manifest_SHA'),('train_features.json','source_features_SHA'),('train_targets.json','source_targets_SHA')]:assert sha(src/f)==m[key]
 assert len(rows)==1024 and len(orders)==1920 and all(len(o)==1 and 0<=o[0]<1024 for o in orders)
 repeats=collections.Counter(o[0] for o in orders);counts=collections.Counter()
 kinds=lambda i:targets[i].get('mode','OWT')
 for i in range(1024):assert repeats[i]=={'signed':1,'unsigned':6,'OWT':3}[kinds(i)]
 for start in range(0,1920,5):
  assert [kinds(o[0]) for o in orders[start:start+5]]==['unsigned','signed','OWT','unsigned','signed']
 for o in orders:counts[kinds(o[0])]+=1
 assert counts==dict(unsigned=768,signed=768,OWT=384)
 for cursor in (0,1,4,5,127,128,1023,1024,1919,1920):
  state=json.loads(json.dumps(dict(cursor=cursor,orders=orders)))
  assert state['orders'][state['cursor']:]==orders[cursor:]
 assert sum(len(rows[o[0]]['prompt'].encode()) for o in orders)==m['head_source_UTF8_bytes']==2720189
 assert sum(sum(s['end']-s['start'] for s in targets[o[0]]['critical_spans']) for o in orders)==m['head_critical_bytes']==86108
 profile=io.load_json(D/'symbol_weight_source.json');assert sha(Path(profile['source']))==profile['source_SHA']
 parent=torch.load(profile['source'],map_location='cpu',weights_only=True)
 assert torch.equal(torch.tensor(profile['weights']),parent['runtime']['symbol_weights']) and profile['parent_objective_report']==parent['runtime']['objective_report']
 old=ROOT/'research/artifacts/joint_quantity_eval_20261113_v76_cpu_v1';a=io.load_json(old/'features.json');at=io.load_json(old/'targets.json');b=io.load_json(E/'features.json');bt=io.load_json(E/'targets.json')
 assert len(a)==len(b)==64
 edited=0
 for previous,current,original,t in zip(a,b,at,bt):
  assert set(current)=={'id','prompt'} and current['id']==t['id']
  assert current['prompt']==previous['prompt'].replace("'s a ","'s ")
  edited+=previous['prompt']!=current['prompt']
  assert "'s a " not in current['prompt'] and current['prompt'].encode()[t['clause_start']:].decode()==t['clause']
  # Confirm the revised possession is grammatical for all four known domains.
  if t['layout']=='end':
   assert re.search(r"distinguished (?:Alaric|Blythe|Cyrene|Denzel)'s (?:mass change|temperature change|displacement change|delivery) of ",t['clause'])
  for key in ('actor','peer','unit','sign','A','B','primary_text','secondary_text','quartet_id','member'):assert original[key]==t[key]
  for s in t['critical_spans']:assert current['prompt'].encode()[s['start']:s['end']].decode()==s['text']
  for k in ('A','B'):
   s=t[k+'_span'];assert current['prompt'].encode()[s['start']:s['end']].decode()==s['text']
  assert len(current['prompt'].encode())%16==t['EOS_position']
 assert edited==32
 assert not any(f['prompt'] in r['prompt'] or r['prompt'] in f['prompt'] for f in b for r in rows)
 proof=dict(status='ROOT_V77_SAMPLER_PROFILE_AND_GRAMMAR_CPU_PASS',unique_TRAIN1024=True,head_exposures1920_not_newdocs=True,
  exact_repeats_and_384interleaved_40_40_20_cycles_verified=True,finite_JSON_nextcursor_checked=True,
  frozen_V75_symbol_weights_parent_identical=True,eval_samefacts64_edited32_unchanged32_independently_checked=True,
  all_original_V76_data_scores_preserved=True,features_id_prompt_ONLY=True,new_eval_not_TRAIN=True,
  teacher_unique_rows=1024,head_UTF8_bytes=2720189,head_critical_bytes=86108,
  actual_model_forwards=0,CUDA_initialized=torch.cuda.is_initialized(),
  old_trial_unused_calls_time_reused=False,new_GPU_or_actual_worker=False,
  files_SHA={str(f):sha(f) for root in (D,E) for f in root.glob('*.json')})
 assert not proof['CUDA_initialized'];dest=D/'root_data_profile_CPU_review.json';assert not dest.exists();io.atomic_json(dest,proof)
 print(json.dumps({k:v for k,v in proof.items() if k!='files_SHA'}))
if __name__=='__main__':main()
