"""Create new V29 driver files; never alter an earlier sealed version."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def new(name,text):
    with (R/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)

w=(R/'scripts/run_owt_fuzzy_reconstruction_20261029_v28.py').read_text(encoding='utf-8')
w=w.replace('v28','v29').replace('V28','V29').replace('actual V27','actual V28')
w=w.replace('from research.owt_prefix_hold_v29 import PrefixHoldFuzzyAE as FuzzySentenceAE,OBJECTIVE','from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE as FuzzySentenceAE,OBJECTIVE,TRAINABILITY')
w=w.replace('from owt_fuzzy_engine_v29 import ContinuationEngine,cumulative_count','from owt_fuzzy_engine_v29 import ContinuationEngine,cumulative_count,OWNERSHIP')
w=w.replace('owt_resource_budget_v24','owt_resource_budget_v29')
w=w.replace("'attempts':8192","'attempts':4096").replace("'max_calls':28000","'max_calls':15000")
w=w.replace('be85ab3b6bdd21b70ce9942d40a704a0a04a157245e3336446bde401f5da4648','bc0fe9d59cf39d59290b64377d40e53e6f214c083eaaa3206b2bcf84961c4246')
w=w.replace('d427da55242ad8e8fa6e388cfc12778d0429b34f437a80ca20f3c0113ff0991c','b9bb06b30fc791508ade5c8df42612d61778a79ce078c64e955618ddd13b8eb4')
w=w.replace('626462998312ab4634db9ecaff5eb5ec676cbb94ec17cc5cfe87eba283b52fb1','a61b9ddc88efab59947f0458a78d9423b27c04d0fa60f79e39fd080065f7575c')
w=w.replace("'training':24576","'training':12288").replace("'total_max':26776","'total_max':14488")
w=w.replace("p.get('epochs')!=2","p.get('epochs')!=1").replace("p.get('source_global_attempts')!=79844 or p.get('source_global_updates')!=79817","p.get('source_global_attempts')!=88036 or p.get('source_global_updates')!=88005")
w=w.replace("    if p.get('resource_policy')!=RESOURCE_POLICY:","    if p.get('trainability')!=TRAINABILITY or p.get('ownership_policy')!=OWNERSHIP:raise ValueError('fixed ownership migration')\n    if p.get('resource_policy')!=RESOURCE_POLICY:")
w=w.replace("'new_rows':65536","'new_rows':32768").replace("'target_UTF8_bytes':2*sum","'target_UTF8_bytes':sum").replace("'old_document_rows':32768,'new_document_rows':32768","'old_document_rows':16384,'new_document_rows':16384").replace("'old_target_UTF8_bytes':2*sum","'old_target_UTF8_bytes':sum").replace("'new_target_UTF8_bytes':2*sum","'new_target_UTF8_bytes':sum")
w=w.replace('range(361,363)','range(371,372)').replace('(8192,8190,65536)','(8192,8188,65536)').replace("source.get('curriculum_probability')","source.get('constant_mask_probability')").replace('completed source ramp must end atp50','completed source p50 hold')
w=w.replace("engine=ContinuationEngine(model,f['train'],y['train'],orders,budget,plan_sha,device)","engine=ContinuationEngine(model,f['train'],y['train'],orders,budget,plan_sha,device,source_parameter_groups=source['optimizer_parameter_names'])")
w=w.replace("cumulative_count(source,'updates')!=79817 or cumulative_count(source,'attempts')!=79844","cumulative_count(source,'updates')!=88005 or cumulative_count(source,'attempts')!=88036")
w=w.replace("'all_algorithm_components_sha256':digest","'ownership_migration':digest").replace("'source_local_updates':8190,'source_global_updates':79817","'source_local_updates':8188,'source_global_updates':88005")
w=w.replace('while engine.attempts<8192:','while engine.attempts<4096:').replace('engine.attempts in (4096,)','engine.attempts in (2048,)')
w=w.replace("'global_attempts':79844+engine.attempts,'global_updates':79817+engine.updates","'global_attempts':88036+engine.attempts,'global_updates':88005+engine.updates").replace("'epochs':2","'epochs':1")
start=w.index("            'interpretation':'")
end=w.index("}\n        io.atomic_json(output/'result.json'",start)
w=w[:start]+"            'trainability':TRAINABILITY,'ownership_migration':engine.migration,\n            'interpretation':'Same original fuzzy input-meaning-input reconstruction, p50 objective and inference. Existing language blocks5-6 newly trainable; old63 Adam states/scaler/allRNG preserved, new18 tensors append fresh Adam group at baseLR3e-6. One new finite epoch on unchanged32768spans; original reasoning remains frozen. Saved V28 baseline, no extra initial forwards. Reconstruction from own input state only, teacher scores auxiliary.'"+w[end:]
w=w.replace('all actual V27 weights','all actual V28 weights')
new('scripts/run_owt_fuzzy_reconstruction_20261029_v29.py',w)

o=(R/'scripts/supervise_owt_fuzzy_reconstruction_20261029_v28.py').read_text(encoding='utf-8').replace('v28','v29').replace('V28','V29').replace('owt_resource_budget_v24','owt_resource_budget_v29')
o=o.replace("'attempts':8192","'attempts':4096").replace("'max_calls':28000","'max_calls':15000").replace("plan.get('epochs')!=2","plan.get('epochs')!=1").replace("!=26776","!=14488")
literal="TRAINABILITY={'language_tail_blocks':4,'language_embedding_trainable':False,'same_total_parameters':True,'new_parameters':0,'reasoning_diffusion_training':False,'inference_changed':False}\nOWNERSHIP={'old_groups':2,'append_groups':1,'new_group_base_lr':3e-6,'new_group_weight_decay':.01,'old_moments_preserved':True,'new_states_initially_empty':True}\n"
o=o.replace("ROOT=Path(__file__).resolve().parents[1]",literal+"\nROOT=Path(__file__).resolve().parents[1]")
o=o.replace("    if plan.get('resource_policy')!=RESOURCE_POLICY:","    if plan.get('trainability')!=TRAINABILITY or plan.get('ownership_policy')!=OWNERSHIP:raise ValueError('ownership policy')\n    if plan.get('resource_policy')!=RESOURCE_POLICY:")
new('scripts/supervise_owt_fuzzy_reconstruction_20261029_v29.py',o)

p=(R/'scripts/prepare_owt_fuzzy_reconstruction_20261029_v28.py').read_text(encoding='utf-8')
p=p.replace('v28','v29').replace('V28','V29').replace('v27','v28').replace('V27','V28').replace('owt_resource_budget_v24','owt_resource_budget_v29')
p=p.replace('SCHEMA,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM','SCHEMA,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,LIMITS,COST')
p=p.replace('from research.owt_prefix_hold_v29 import OBJECTIVE','from research.owt_extraction_tail_v29 import OBJECTIVE,TRAINABILITY\nfrom owt_fuzzy_engine_v29 import OWNERSHIP')
p=p.replace("!=79817","!=88005").replace("!=8190","!=8188").replace('source_global_attempts=79844,source_global_updates=79817','source_global_attempts=88036,source_global_updates=88005')
p=p.replace('baseline_run=str(baseline),','limits=LIMITS,cost_contract=COST,epochs=1,trainability=TRAINABILITY,ownership_policy=OWNERSHIP,baseline_run=str(baseline),')
start=p.index('        import_policy=');end=p.index("    names=",start)
p=p[:start]+"        import_policy='Actual V28 fullstate model/allRNG/scaler and old63 Adam subcomponents exact; append18 existing tensors with empty fresh Adam group; source oldfrozen and phase frozen scopes separately retained.',\n        reference_policy='Saved V28 assessment at88005updates; same held256 and fixedtrain16, no initial GPU baseline repeat.',\n        interpretation='One finite epoch; thaw existing language blocks5-6 at3e-6, original fuzzy mechanism/decoder/inference/p50 unchanged. No output-meaning reasoning training; no old budget reuse.')\n    p['training_exposure']={k:v//2 for k,v in p['training_exposure'].items()}\n"+p[end:]
p=p.replace('research/owt_prefix_hold_v29.py','research/owt_extraction_tail_v29.py').replace('tests/test_owt_prefix_hold_v29.py','tests/test_owt_extraction_tail_runtime_v29.py')
p=p.replace("names=('scripts/owt_resource_budget_v29.py',","names=('scripts/build_owt_v29_driver.py','scripts/owt_resource_budget_v29.py',")
new('scripts/prepare_owt_fuzzy_reconstruction_20261029_v29.py',p)
