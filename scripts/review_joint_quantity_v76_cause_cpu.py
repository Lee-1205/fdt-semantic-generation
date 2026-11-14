"""Saved-history/data composition review and explicit research-base selection."""
import collections,datetime,json,pathlib,re,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
 torch.set_num_threads(2);run=ROOT/'runs/joint_quantity_20261113_v76';p=io.load_json(run/'plan.json');m=io.load_json(run/'root_meaning_review.json');saved=io.load_json(run/'root_saved_review.json')
 assert saved['status']=='ROOT_V76_SAVED_REVIEW_PASS'
 stages={}
 for label,r in [('V75',pathlib.Path(p['baseline_run'])),('V76',run)]:
  q=io.load_json(r/'plan.json');rows=io.load_json(pathlib.Path(q['data'])/'train_features.json');ts=io.load_json(pathlib.Path(q['data'])/'train_targets.json');counts=collections.defaultdict(collections.Counter)
  for f,t in zip(rows,ts):counts[t.get('mode','OWT')].update(rows=1,UTF8bytes=len(f['prompt'].encode()),plus=f['prompt'].count('+'),minus=f['prompt'].count('-'))
  state=torch.load(r/'model/runtime.pt',map_location='cpu',weights_only=True)['runtime'];hist=state['history']
  stages[label]=dict(TRAIN_by_kind=dict(counts),wrong_CE_fraction_actual=sum(.25*h['wrong_critical_CE'] for h in hist)/sum(h['loss'] for h in hist),
   data_derived_base_class_weights=q['objective_report']['class_weights'],per_row_normalized_objective=True,
   wrong_critical_amplification_max=max(h.get('wrong_critical_position_amplification',0) for h in hist) if label=='V76' else None)
 longnew=m['long_paired']['quantities']['new_errors'];clear=[x for x in longnew if re.fullmatch(r'[+-]?\d+',x['output'])];mal=[x for x in longnew if x not in clear]
 assert len(clear)==13 and len(mal)==7
 evalrows=io.load_json(pathlib.Path(p['eval_data'])/'features.json');awkward=[i for i,r in enumerate(evalrows) if "'s a " in r['prompt']]
 assert len(awkward)==32
 proof=dict(status='ROOT_V76_CAUSE_SCOPE_AND_SOURCE_READING_REVIEW_COMPLETE',training_composition=stages,
  causal_interpretation='Signed rows40% to75%, unsigned40% to12.5%, sameOWT rowshare20% to12.5%; each window normalized, so byte share is not update share. Data-derived digit class weight also declined. Joint changes support interference/distribution hypothesis, not isolated proof.',
  capped_loss_implementation_verified_but_insufficient=True,
  new_long_numeric_positions=20,new_long_clear_different_values=clear,new_long_malformed_lost_values=mal,
  long_error_positions_not20independent_documents=True,long_total_quantity_slots=5436,long_events=2718,
  original_eval_awkward_possessive_article_rows=awkward,all64_original_scores_retained_no_posthoc_exclusion=True,
  eval_quality_caveat='32 END wrappers contain possessive plus an article; not natural-English benchmark. This cannot explain separate observed-long numerical regression.',
  ROOT_read_new22_changed_clauses=True,ROOT_read_observed_V75_all14_changed_clauses=True,
  ROOT_read_all63_changed_long_event_contexts=True,ROOT_read_all34_unique_long_field_patterns=True,ROOT_read6long_regions=True,
  manual_scope_not_all524kbytes_individually_semantically_scored=True,
  no_output_repair=True,extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized())
 assert not proof['CUDA_initialized'];io.atomic_json(run/'root_cause_and_reading_review.json',proof)
 selection=dict(time=datetime.datetime.now().astimezone().isoformat(),decision='KEEP_V75_AS_NEXT_RESEARCH_INITIALIZATION_DO_NOT_PROMOTE_V76',
  selected_version='V75',selected_runtime=p['cycle_source'],selected_SHA=p['cycle_sha'],mandatory_standalone_base=p['storage_parent'],mandatory_base_SHA=p['storage_parent_sha'],
  rejected_as_default='V76',V76_final_SHA=io.load_json(run/'latest.json')['sha256'],all_V76_states_preserved=True,
  reason='Observed long quantity errors16 to36 with20new and0fixed; naturalOWT decoded108 to122;new64 two-value correctness58/64 unchanged despite stock/unit gains.',
  deployed_model_registry_changed=False,weight_files_modified=False,automatic_extra_training=False,all_unused_budgets_closed=True)
 assert support.sha(selection['selected_runtime'])==selection['selected_SHA']
 config=ROOT/'configs/research_base_selection_v76_review_20261113.json';assert not config.exists();io.atomic_json(config,selection);io.atomic_json(run/'root_research_base_selection.json',selection)
 print(json.dumps(dict(status=proof['status'],training=stages,new_long_clear=13,new_long_malformed=7,awkward_eval_rows=32,selection=selection)))
if __name__=='__main__':main()
