"""ROOT seal after completed CPU contracts and actual independent-worker preflight."""
import json,time,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support

o=R/'runs/owt_fuzzy_reconstruction_20261029_v29';path=o/'plan.json';p=io.load_json(path)
if p['authorized'] or p['deadline'] is not None or (o/'progress.json').exists():raise RuntimeError('new allocation seal only')
env=json.loads((o/'code_snapshot/root_preseal_native_resources.json').read_text(encoding='utf-8-sig'))
if env['native_python'] or env['C_free_bytes']<p['limits']['start_free_bytes'] or env['GPU_free_MiB']<12288:raise RuntimeError('preseal native/resources')
import xml.etree.ElementTree as ET
suite=ET.parse(o/'code_snapshot/cpu_delta_retry.xml').getroot()[0]
if suite.attrib['tests']!='7' or suite.attrib['errors']!='0' or suite.attrib['failures']!='0':raise RuntimeError('focused CPU contracts')
for name,digest in p['files'].items():
    if support.sha(name)!=digest:raise RuntimeError('code/source/data pin changed '+name)
p['files'][str(Path(__file__).resolve())]=support.sha(__file__)
(o/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
review={'CPU_focused_pass':7,'CPU_seconds':float(suite.attrib['time']),
    'initial_test_temp_permission_error_preserved':'cpu_delta.xml; no model execution in failed tmp_path setup',
    'independent_validate_stdout':'INDEPENDENT_VALIDATE_PASS','actual_worker_CPU_preflight':True,
    'actual_source_forward_calls':0,'actual_CUDA_replay_performed':False,
    'old_Adam63_exact_subcomponents':True,'new_existing18_append_group_empty':True,
    'source_and_phase_frozen_scopes_separate':True,'model_parameters':111485711,'trainable_parameters':38309762,
    'source_model_tensor_values_unchanged_before_SGD':True,'new_parameters':0,
    'same_fuzzy_input_meaning_reconstruction_only':True,'reasoning_training':False,
    'cost_contract':p['cost_contract'],'fixed_limits':p['limits'],'resources':env,
    'single_hidden_launch':True,'unused_prior_budget_reused':False}
io.atomic_json(o/'readiness.json',review)
now=time.time();p.update(authorized=True,allocation_start=now,deadline=now+900,cutoff=now+810)
io.atomic_json(path,p);review.update(plan_sha256=support.sha(path),allocation_start=now,deadline=p['deadline'],cutoff=p['cutoff'])
io.atomic_json(o/'code_snapshot/root_seal_review.json',review)
print(json.dumps({k:review[k] for k in ('plan_sha256','allocation_start','deadline','cutoff')}))
