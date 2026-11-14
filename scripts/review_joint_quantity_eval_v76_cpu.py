"""Independent prospective eval clause/quantity/quartet checks; no NN."""
import collections,hashlib,itertools,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import learning_io_v2 as io
D=ROOT/'research/artifacts/joint_quantity_eval_20261113_v76_cpu_v1'
END=r"Review (?P<episode>\d+) distinguished (?P<actor>\w+)'s (?P<kind>a mass change|a temperature change|a displacement change|a delivery) of (?P<A>[+-]?\d+(?:\.75)? (?:kg|degrees C|km|copper washers)) from the independent (?P<other>stock weight|chamber temperature|route length|remaining stock) that (?P<peer>\w+) verified as (?P<B>\+?\d+(?:\.25)? (?:kg|degrees C|km|copper washers))\."
MID=r"The independent (?P<other>stock weight|chamber temperature|route length|remaining stock), verified by (?P<peer>\w+), was (?P<B>\+?\d+(?:\.25)? (?:kg|degrees C|km|copper washers)) during review (?P<episode>\d+); (?P<actor>\w+) separately documented (?P<kind>a mass change|a temperature change|a displacement change|a delivery) of (?P<A>[+-]?\d+(?:\.75)? (?:kg|degrees C|km|copper washers))\. Neither figure replaces the other."
def main():
 rows=io.load_json(D/'features.json');targets=io.load_json(D/'targets.json');m=io.load_json(D/'manifest.json');groups=collections.defaultdict(list)
 assert len(rows)==len(targets)==64
 for row,t in zip(rows,targets):
  assert set(row)=={'id','prompt'} and row['id']==t['id'];raw=row['prompt'].encode();assert len(raw)%16==t['EOS_position']
  clause=raw[t['clause_start']:].decode();assert clause==t['clause'];g=re.fullmatch(END if t['layout']=='end' else MID,clause);assert g
  assert g['actor']==t['actor'] and g['peer']==t['peer'] and g['A']==t['primary_text'] and g['B']==t['secondary_text']
  unit=t['unit'];A=f"{t['A']} copper washers" if unit=='items' else f"{t['sign']}{t['A']}.75 {unit}";B=f"{t['B']} copper washers" if unit=='items' else f"+{t['B']}.25 {unit}"
  assert A==g['A'] and B==g['B']
  expected_kind={'kg':('a mass change','stock weight'),'degrees C':('a temperature change','chamber temperature'),'km':('a displacement change','route length'),'items':('a delivery','remaining stock')}[unit]
  assert (g['kind'],g['other'])==expected_kind
  for k in ('A','B'):
   s=t[k+'_span'];expected=t['primary_text' if k=='A' else 'secondary_text'].removesuffix(' copper washers')
   assert raw[s['start']:s['end']].decode()==s['text']==expected
  for s in t['critical_spans']:assert raw[s['start']:s['end']].decode()==s['text']
  groups[t['quartet_id']].append((row,t))
 for quartet in groups.values():
  assert len(quartet)==4 and {tuple(t['member']) for _,t in quartet}==set(itertools.product(range(2),repeat=2))
  q={tuple(t['member']):t for _,t in quartet};normal=[]
  for row,t in quartet:
   x=row['prompt']
   for key in sorted(('A','B'),key=lambda k:t[k+'_span']['start'],reverse=True):
    span=t[key+'_span'];x=x[:span['start']]+'<'+key+'>'+x[span['end']:]
   normal.append(x)
  assert len(set(normal))==1
  for i,j in itertools.product(range(2),repeat=2):assert q[i,j]['A']==q[0,0]['A']+i and q[i,j]['B']==q[0,0]['B']+j
 train=io.load_json(ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1/train_features.json')
 assert not any(r['prompt'] in t['prompt'] or t['prompt'] in r['prompt'] for r in rows for t in train)
 assert not any(re.search(r'\b'+n+r'\b',t['prompt']) for t in train for n in m['names_previously_observed_but_never_currentTRAIN'])
 assert len(groups)==16 and {t['EOS_position'] for t in targets}==set(range(16))
 proof=dict(status='ROOT_V76_PROSPECTIVE_EVAL64_CPU_PASS',all64_whole_clauses_independently_reparsed=True,all16_quartets_independent_A_B=True,
   all_UTF8_offsets_values_domains_names_checked=True,actual_prompts_names_notTRAIN=True,
   new_strings_but_previously_observed_held_names_not_blind=True,NN_calls=0,E2_registry_access=0,
   files_SHA={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.glob('*.json')})
 target=D/'root_eval_CPU_review.json';assert not target.exists();io.atomic_json(target,proof);print(json.dumps({k:v for k,v in proof.items() if k!='files_SHA'}))
if __name__=='__main__':main()
