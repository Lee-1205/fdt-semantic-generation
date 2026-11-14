"""Source-bound value/identity/manual cards from saved outputs; no NN."""
import collections,json,re,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_extraction_v73_meaning_cpu import short_fields,long_fields
from review_long_extrapolation_v69_saved_cpu import PATTERN
from review_context_v74_meaning_cpu import kg_values,primary_values
from review_critical_retention_v75_meaning_cpu import primary75
from review_v75_regression_severity_cpu import stock_tiers,oldkg_tiers

def field_values(rows,targets,records):
 counts=collections.defaultdict(collections.Counter);cards=[];complete=0;group={}
 for i,(row,t,r) in enumerate(zip(rows,targets,records)):
  out=r['output'].encode();okay=True;states={}
  for k in ('A','B'):
   s=t[k+'_span'];expected=s['text'];actual=out[s['start']:s['end']].decode(errors='replace')
   unit='' if t['unit']=='items' else ' '+re.escape(t['unit'])
   match=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?)'+unit,actual)
   gold=Decimal(expected if t['unit']=='items' else expected[:-(len(t['unit'])+1)])
   tier='same_value_unit_clear' if match and Decimal(match[1])==gold else 'clear_value_change' if match else 'malformed_missing_or_symbol'
   cell=k+':'+t['unit'];counts[cell]['fields']+=1;counts[cell][tier]+=1;okay &= tier=='same_value_unit_clear';states[k]=tier
   if tier!='same_value_unit_clear':cards.append(dict(row=i,field=k,source=expected,output=actual,tier=tier,unit=t['unit'],layout=t['layout'],quartet=t['quartet_id'],member=t['member']))
  complete+=okay;group[tuple((t['quartet_id'],*t['member']))]=states
 pairtotal=pairunchanged=paircorrect=0
 for key in sorted({k[0] for k in group}):
  for axis in (0,1):
   other='B' if axis==0 else 'A'
   for value in (0,1):
    ab0=(0,value) if axis==0 else (value,0);ab1=(1,value) if axis==0 else (value,1)
    if (key,*ab0) in group and (key,*ab1) in group:
     pairtotal+=1
     # Both unchanged facts have the correct source value; matching two wrong
     # strings is not counted as successful fact independence.
     paircorrect+=group[(key,)+ab0][other]==group[(key,)+ab1][other]=='same_value_unit_clear'
 return dict(cells=dict(counts),all_two_value_fields_clear_rows=complete,risks=cards,unchanged_other_fact_pair_checks=pairtotal,unchanged_other_fact_both_correct=paircorrect)

def paired_long(rows,targets,before,after):
 cats={'quantities':('quantity','reserve'),'names':('actor','checker','negative_actor'),'episode':('episode',),'items':('item','reserve_item'),'order':('order',),'locations':('location',),'negative':('negative',)}
 result={}
 for cat,fields in cats.items():
  bad=[{},{}]
  for stage,records in enumerate((before,after)):
   for ri,(row,t,r) in enumerate(zip(rows,targets,records)):
    raw=row['prompt'].encode();out=r['output'].encode()
    for j,e in enumerate(t['events']):
     a,b=e['byte_start'],e['byte_end'];m=PATTERN.fullmatch(raw[a:b].decode());assert m
     for k in fields:
      if k not in m.groupdict() or m[k] is None:continue
      s,z=m.span(k);text=out[a+s:a+z].decode(errors='replace')
      if m[k].strip().casefold()!=text.strip().casefold():bad[stage][ri,j,k]=dict(row=ri,event=j+1,field=k,source=m[k],output=text)
  b,a=map(set,bad);result[cat]=dict(before=len(b),after=len(a),corrected=len(b-a),newly_wrong=len(a-b),still_wrong=len(a&b),
    new_errors=[bad[1][key] for key in sorted(a-b)])
 return result

def main():
 run=ROOT/'runs/update_retention_20261113_v80';assert io.load_json(run/'root_saved_review.json')['status']=='ROOT_V80_SAVED_REVIEW_PASS'
 p=io.load_json(run/'plan.json');base=Path(p['baseline_run']);ed=Path(p['eval_data']);old=Path(p['old_eval_data'])
 rows=io.load_json(ed/'features.json');targets=[dict(t,origin='same_observed_revised64_no_new_blind_cases') for t in io.load_json(ed/'targets.json')]
 before=io.load_json(run/'new_baseline_reconstructions.json');after=io.load_json(run/'new_English_reconstructions.json')
 nsb,bc=short_fields(rows,targets,before);nsa,ac=short_fields(rows,targets,after)
 nb=field_values(rows,targets,before);na=field_values(rows,targets,after)
 ows=io.load_json(old/'short_features.json');ot=io.load_json(old/'short_targets.json');ob=io.load_json(base/'old_untrained128_reconstructions.json');oa=io.load_json(run/'old_untrained128_reconstructions.json')
 lr=io.load_json(old/'long_features.json')[:2];lt=io.load_json(old/'long_targets.json')[:2];lb=io.load_json(base/'long_reconstructions.json');la=io.load_json(run/'long_reconstructions.json')
 lbs,lbc,_=long_fields(lr,lt,lb);las,lac,regions=long_fields(lr,lt,la);paired=paired_long(lr,lt,lb,la)
 panels={}
 for name,data,b_label in [('observed_V74_DEV',Path(p['observed_V74_data']),'observed_V74_DEV'),('observed_V75_DEV',Path(p['observed_V75_data']),'new_English')]:
  sr=io.load_json(data/'dev_features.json');st=[dict(t,origin=name) for t in io.load_json(data/'dev_targets.json')]
  b=io.load_json(base/(b_label+'_reconstructions.json'));a=io.load_json(run/(name+'_reconstructions.json'))
  vb,vcb=short_fields(sr,st,b);va,vca=short_fields(sr,st,a)
  entry=dict(before=vb,after=va,stock_before=kg_values(sr,st,b),stock_after=kg_values(sr,st,a))
  entry.update(primary_before=primary_values(sr,st,b) if name=='observed_V74_DEV' else primary75(sr,st,b),
   primary_after=primary_values(sr,st,a) if name=='observed_V74_DEV' else primary75(sr,st,a))
  if name=='observed_V75_DEV':entry.update(stock_tiers_before=stock_tiers(sr,st,b),stock_tiers_after=stock_tiers(sr,st,a))
  panels[name]=entry
  io.atomic_json(run/(name+'_reading_cards.json'),dict(before=vcb,after=vca))
 osb,_=short_fields(ows,ot,ob);osa,oc=short_fields(ows,ot,oa)
 v76run=ROOT/'runs/joint_quantity_20261113_v76'
 v76comparison=paired_long(lr,lt,io.load_json(v76run/'long_reconstructions.json'),la)
 v77comparison=paired_long(lr,lt,io.load_json(ROOT/'runs/update_balance_20261113_v77/long_reconstructions.json'),la)
 v78comparison=paired_long(lr,lt,io.load_json(ROOT/'runs/competing_margin_20261113_v78/long_reconstructions.json'),la)
 v79comparison=paired_long(lr,lt,io.load_json(ROOT/'runs/last_linear_20261113_v79/long_reconstructions.json'),la)
 report=dict(status='ROOT_V80_SOURCE_BOUND_MEANING_READY',new64_before=nsb,new64_after=nsa,new64_two_values_before=nb,new64_two_values_after=na,
  old128_before=osb,old128_after=osa,old_positive_kg_before=kg_values(ows,ot,ob),old_positive_kg_after=kg_values(ows,ot,oa),
  old_positive_kg_tiers_before=oldkg_tiers(ows,ot,ob),old_positive_kg_tiers_after=oldkg_tiers(ows,ot,oa),
    observed_panels=panels,long_before=lbs,long_after=las,long_paired=paired,V76_to_V80_long_paired=v76comparison,V77_to_V80_long_paired=v77comparison,V78_to_V80_long_paired=v78comparison,V79_to_V80_long_paired=v79comparison,extra_NN_calls=0,
    limitations='Source-bound numeric/identity checks and restricted authored event grammar, not abstract latent-semantic accuracy; all outputs unfiltered; revised64 has32unchanged+32edited prompts/same64facts/names alreadyobserved, not64newblindcases.')
 io.atomic_json(run/'root_meaning_review.json',report)
 io.atomic_json(run/'root_manual_reading_cards.json',dict(new_before=bc,new_after=ac,old128_after=oc,long_before=lbc,long_after=lac,long_regions=regions))
 print(json.dumps(dict(new_two_values_before={k:v for k,v in nb.items() if k!='risks'},new_two_values_after={k:v for k,v in na.items() if k!='risks'},
  new_field_changes_before=nsb['changed_source_fields'],new_field_changes_after=nsa['changed_source_fields'],
  old_positive72before=report['old_positive_kg_tiers_before'],old_positive72after=report['old_positive_kg_tiers_after'],
  observed={k:dict(stock_before=v['stock_before']['counts'],stock_after=v['stock_after']['counts'],primary_before=v['primary_before']['cells'],primary_after=v['primary_after']['cells']) for k,v in panels.items()},
  long_paired=paired,long_polarity=[r['polarity'] for r in las['rows']])))
if __name__=='__main__':main()
