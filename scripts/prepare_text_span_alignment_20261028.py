"""Create an unauthorized execution draft from the completed CPU policy review."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from text_generation_support_v1 import sha
from run_text_span_alignment_20261028 import SCHEMA,LIMITS,PARENT_SHA,CACHE_SHA,MANIFEST_SHA

def create():
 output=ROOT/'runs/text_span_alignment_20261028_v4';output.mkdir(exist_ok=False)
 artifact=ROOT/'research/artifacts/text_span_alignment_20261028_v4_cpu_v2'
 parent=ROOT/'runs/text_semantic_alignment_20261028_v3/checkpoints/state_0512.pt'
 cache=ROOT/'runs/text_context_decoder_20261028_v2/frozen_context_cache.pt'
 if sha(parent)!=PARENT_SHA or sha(cache)!=CACHE_SHA or sha(artifact/'manifest.json')!=MANIFEST_SHA:raise ValueError('reviewed source SHA')
 paths=[ROOT/p for p in ('research/text_span_reader_v4.py','research/text_span_data_v4.py',
  'research/text_context_decoder_v2.py','research/text_diffusion_v1.py','research/text_generation_data_v1.py',
  'research/text_semantic_targets_v3.py','scripts/text_span_engine_v4.py','scripts/run_text_span_alignment_20261028.py',
  'scripts/supervise_text_span_alignment_20261028.py','scripts/text_generation_support_v1.py',
  'scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py',
  'tests/test_text_span_runtime_v4.py','configs/text_span_alignment_20261028_v4_cpu_policy.json')]
 paths+=list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(artifact.glob('*.json')))+[parent,cache]
 plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'artifact':str(artifact),
  'parent':str(parent),'parent_sha':PARENT_SHA,'cache':str(cache),'cache_sha':CACHE_SHA,'manifest_sha':MANIFEST_SHA,
  'files':{str(p):sha(p) for p in paths},'limits':LIMITS,'phase_limits':{'reader':512,'realizer':512},
  'transition_boundary':256,'allocation_start':None,'deadline':None,'cutoff':None,
  'E2_final_accessed':False,'evaluation_teacher_allowed':False,'max_actual_calls_expected':2520,
  'purpose':'Learn raw question byte pointers then train a faithful neural realizer on consistent teacher/matched-prediction conditions',
  'phase_contract':'reader-only512, realizer-only512; fixed factor embeddings and VP; one continuous scaler/allRNG with phase-specific fresh optimizers',
  'row_exposure_caps':{'reader':4096,'realizer':4096,'transition_reader':2048},
  'checkpoint_contract':'reader0/256/512 plus realizer global512/768/1024, both optimizer states, scaler/allRNG/orders/cursors/history/exposure/calls; emergency current fullstate; no deletion',
  'stops':['nonfinite','frozen/ownership','3AMP skips','loss>100','fixed time/calls/finiteorders','disk/output','OSprocess'],
  'evaluation':'all80 observed rows predicted-only; no teacher/oracle filtering; baseline/headcheck/final/zero-VP',
  'goals':{'each_dev_exact':.75,'each_dev_positive_exact':.60,'each_dev_well_formed':.90},
  'interpretation':'Finite A-H bilingual synthetic candidate, not general NER/LM or isolated diffusion causal proof; no old budget reused'}
 io.atomic_json(output/'plan.json',plan);snapshot=output/'code_snapshot';snapshot.mkdir()
 for p in paths:
  if p.suffix=='.py' and p.parent.name in ('scripts','research','tests'):(snapshot/p.name).write_bytes(p.read_bytes())
 print(json.dumps({'output':str(output),'draft_plan_sha256':sha(output/'plan.json'),'authorized':False}))
if __name__=='__main__':create()
