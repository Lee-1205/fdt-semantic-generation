"""New finite continuation after first extraction-tail epoch; old closed code preserved."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def new(name,s):
    with (R/name).open('x',encoding='utf-8',newline='\n') as f:f.write(s)

s=(R/'scripts/run_owt_fuzzy_reconstruction_20261029_v29.py').read_text(encoding='utf-8')
s=s.replace('actual V28','actual V29').replace('owt_fuzzy_engine_v29','owt_fuzzy_engine_v30').replace('reconstruction_v29','reconstruction_v30').replace('RECONSTRUCTION_V29','RECONSTRUCTION_V30')
s=s.replace("'attempts':4096","'attempts':8192").replace("'max_calls':15000","'max_calls':28000").replace("'training':12288","'training':24576").replace("'total_max':14488","'total_max':26776").replace("p.get('epochs')!=1","p.get('epochs')!=2")
s=s.replace('bc0fe9d59cf39d59290b64377d40e53e6f214c083eaaa3206b2bcf84961c4246','97165ab6e7fcf9e6b75174b5b8eff80801c7f95925e49c3c8923b70cbd0d22ae').replace('b9bb06b30fc791508ade5c8df42612d61778a79ce078c64e955618ddd13b8eb4','b004c9dc1c510c3ed75cd03dab896faff4e615cbd79b23f4a23260bd87b523d3').replace('a61b9ddc88efab59947f0458a78d9423b27c04d0fa60f79e39fd080065f7575c','7babea2452ea480fba1e917ee9286b9b27cc31a64cf1a2d04cea07187ffc59e0')
s=s.replace('!=88036','!=92132').replace('!=88005','!=92099').replace('88036+engine.attempts','92132+engine.attempts').replace('88005+engine.updates','92099+engine.updates')
s=s.replace("'new_rows':32768","'new_rows':65536").replace("'target_UTF8_bytes':sum","'target_UTF8_bytes':2*sum").replace("'old_document_rows':16384,'new_document_rows':16384","'old_document_rows':32768,'new_document_rows':32768").replace("'old_target_UTF8_bytes':sum","'old_target_UTF8_bytes':2*sum").replace("'new_target_UTF8_bytes':sum","'new_target_UTF8_bytes':2*sum")
s=s.replace('range(371,372)','range(381,383)').replace('(8192,8188,65536)','(4096,4094,32768)')
s=s.replace("'ownership_migration':digest","'exact_continuation_import':digest").replace("'source_local_attempts':8192,'source_local_updates':8188,'source_global_updates':88005","'source_local_attempts':4096,'source_local_updates':4094,'source_global_updates':92099")
s=s.replace('while engine.attempts<4096:','while engine.attempts<8192:').replace('engine.attempts in (2048,)','engine.attempts in (4096,)').replace("'epochs':1","'epochs':2")
s=s.replace("'ownership_migration':engine.migration","'original_V29_first_thaw_migration':engine.migration")
s=s.replace("'interpretation':'Same original fuzzy input-meaning-input reconstruction, p50 objective and inference. Existing language blocks5-6 newly trainable; old63 Adam states/scaler/allRNG preserved, new18 tensors append fresh Adam group at baseLR3e-6. One new finite epoch on unchanged32768spans; original reasoning remains frozen. Saved V28 baseline, no extra initial forwards. Reconstruction from own input state only, teacher scores auxiliary.'", "'interpretation':'Two finite epochs continuing first four-language-block adaptation. All81 populated Adam states/three groups and full model/scaler/allRNG/frozen scope retained; no fresh Adam/architecture/loss/inference/corpus change. Original fuzzy input meaning to SAME input reconstruction only; reasoning remains frozen. Saved V29 baseline with no initial GPU repeat. Teacher scores auxiliary, not meaning preservation percentage.'")
new('scripts/run_owt_fuzzy_reconstruction_20261029_v30.py',s)

o=(R/'scripts/supervise_owt_fuzzy_reconstruction_20261029_v29.py').read_text(encoding='utf-8').replace('reconstruction_v29','reconstruction_v30').replace('RECONSTRUCTION_V29','RECONSTRUCTION_V30')
o=o.replace("'attempts':4096","'attempts':8192").replace("'max_calls':15000","'max_calls':28000").replace("plan.get('epochs')!=1","plan.get('epochs')!=2").replace('!=14488','!=26776')
new('scripts/supervise_owt_fuzzy_reconstruction_20261029_v30.py',o)

p=(R/'scripts/prepare_owt_fuzzy_reconstruction_20261029_v29.py').read_text(encoding='utf-8')
p=p.replace('from the actual V28','from the actual V29').replace('reconstruction_20261029_v29','reconstruction_20261029_v30').replace('owt_fuzzy_engine_v29','owt_fuzzy_engine_v30')
p=p.replace("old=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v28'","old=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v29'")
p=p.replace('state_8192.pt','state_4096.pt').replace('!=88005','!=92099').replace('!=8188','!=4094').replace('source_global_attempts=88036,source_global_updates=88005','source_global_attempts=92132,source_global_updates=92099').replace('epochs=1','epochs=2')
p=p.replace("import_policy='Actual V28 fullstate model/allRNG/scaler and old63 Adam subcomponents exact; append18 existing tensors with empty fresh Adam group; source oldfrozen and phase frozen scopes separately retained.'","import_policy='Actual V29 all81 populated Adam states/3groups/model/scaler/allRNG/frozen scope exact; no new optimizer state or architecture change.'")
p=p.replace('Saved V28 assessment at88005updates','Saved V29 assessment at92099updates').replace('One finite epoch; thaw existing language blocks5-6 at3e-6','Two finite epochs adapting same four existing language tail blocks with all81 moments retained')
p=p.replace('v//2','v*2').replace('scripts/build_owt_v29_driver.py','scripts/build_owt_v30_driver.py').replace('scripts/owt_fuzzy_engine_v29.py','scripts/owt_fuzzy_engine_v30.py').replace('tests/test_owt_extraction_tail_runtime_v29.py','tests/test_owt_extraction_tail_continuation_v30.py')
new('scripts/prepare_owt_fuzzy_reconstruction_20261029_v30.py',p)
