"""Repeated-order/profile fullstate contracts on a smoke model, CPU only."""
import copy,json,random,re,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from research.fuzzy_factual_objective_v70 import training_weights
from research import linked_runtime_v75 as linked
from update_balance_engine_v77 import RetentionEngine
import text_generation_support_v1 as support

def main():
 started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(77);random.seed(77);np.random.seed(77)
 assert shutil.disk_usage(ROOT).free>=100*1024**3+32*1024**2
 rows=[dict(id='a',prompt='Cruz stored +7 kg!!'),dict(id='b',prompt='Cyrus logged -2.50 degrees C at 03:17.'),dict(id='c',prompt='A plain sentence without a numeral. '*2)]
 labels={r['id']:complete_expression_spans(r['prompt'],[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'Cruz|Cyrus',r['prompt'])]) for r in rows}
 orders=[[0],[1],[2],[0],[1]];directory=ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1/engine_fixture_v1';directory.mkdir(exist_ok=False);config=Config.smoke()
 _,source_report=training_weights(rows);source_report.update(class_weights=[1.,1.,1.,1.],normalization=1.)
 weights=[1.]*258;weights[257]=0.
 profile=dict(weights=weights,parent_objective_report=source_report,source_SHA='fixture_parent_TRAIN_only')
 def model():
  e=FuzzyCodecV64(config,input_capacity=256);e.enable_position_adapter();e.enable_long_capacity();e.enable_local_banks()
  return FuzzyRoundTripV64(e,OrderedFuzzyReadout(config.z_dim,config.value_dim,hidden=128))
 initial=copy.deepcopy(model().state_dict());parent=directory/'parent.pt';torch.save(dict(model=initial),parent)
 reference=dict(path=str(parent.resolve()),SHA=linked.file_sha(parent),model_key='model',model_SHA=support.state_sha(initial))
 class Budget:
  data_sha='fixture77';parent_reference={'source':'fixture77'};critical_targets=labels;frozen_symbol_profile=profile
  def __init__(self,label):
   self.output=directory/label;self.output.mkdir();self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0);self.block_work=0;self.fail=False
  def check(self,write_bytes=0,emergency=False):
   if time.monotonic()-started>60:raise TimeoutError('CPU fixture60sec')
   if shutil.disk_usage(ROOT).free<100*1024**3+write_bytes:raise OSError('100GiB plus pending write')
   if sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())+write_bytes>32*1024**2:raise OSError('32MiB fixture cap')
  def charge(self,kind):
   if self.fail:self.fail=False;raise InterruptedError('fixture STOP')
   self.calls[kind]+=1
 def create(label):
  with torch.random.fork_rng(devices=[]):m=model();m.load_state_dict(initial,strict=True)
  return RetentionEngine(m,rows,orders,Budget(label),'fixture_plan77','cpu')
 a=create('a');assert torch.equal(a.symbol_weights,torch.tensor(weights))
 assert a.objective_report['class_weights']==[1.,1.,1.,1.] and a.objective_report['current_unique_pool_class_weights_NOT_used']!=[1.,1.,1.,1.]
 a.cache_next();partial=copy.deepcopy(a.export());a.cache_next();a.cache_next();complete=copy.deepcopy(a.export())
 b=create('b');b.restore(partial);b.cache_next();b.cache_next()
 def normalized(s):
  s=copy.deepcopy(s)
  for r in s['cache_refs']:r['targets']=Path(r['targets']).name
  return s
 assert support.state_sha(normalized(b.export()))==support.state_sha(normalized(complete))
 transition=create('transition');transition.restore(complete);assert support.state_sha(transition.export())==support.state_sha(complete)
 a.step();middle=copy.deepcopy(a.export());env=linked.pack(middle,initial,reference,a.names,support.state_sha)
 a.budget.check(write_bytes=16*1024**2);meta=linked.save_verified(directory/'populated_runtime.pt',env,ROOT,support.state_sha);stored=linked.load_verified(meta['path'],meta['SHA'],ROOT,support.state_sha)
 a.step(True);a.step();a.step();a.step();expected=a.export()
 restored=create('restored');restored.restore(stored);restored.step(True);restored.step();restored.step();restored.step()
 assert support.state_sha(restored.export())==support.state_sha(expected)
 assert a.attempts==a.cursor==5 and a.updates==4 and a.skips==1 and a.scaler.is_enabled() and len(a.optimizer.state)==7
 assert all(int(v['step'])==4 for v in a.optimizer.state.values()) and a.budget.calls==dict(encode=8,readout=8,stream_chunk=0,denoise=0)
 assert a.exposure['encoder_rows']==3 and a.exposure['head_rows']==a.exposure['head_encoder_rows']==5
 assert a.exposure['head_UTF8_bytes']==a.exposure['head_encoder_UTF8_bytes']==sum(len(rows[o[0]]['prompt'].encode()) for o in orders)
 assert not list(directory.rglob('*.features.pt'));a.verify_frozen()
 for stage in ('teacher','head'):
  stopped=create('stop_'+stage)
  if stage=='head':stopped.restore(complete)
  stopped.budget.fail=True
  try:
   stopped.cache_next() if stage=='teacher' else stopped.step();raise AssertionError('expected STOP')
  except InterruptedError:pass
  saved=copy.deepcopy(stopped.export());again=create('restored_stop_'+stage);again.restore(saved)
  assert support.state_sha(again.export())==support.state_sha(saved)
  try:
   again.cache_next() if stage=='teacher' else again.step();raise AssertionError('automatic replay forbidden')
  except RuntimeError:pass
 for key in ('embedding_sha','symbol_weights'):
  bad=copy.deepcopy(middle)
  if key=='embedding_sha':bad[key]='wrong'
  else:bad[key][0]+=1
  try:restored.restore(bad);raise AssertionError('identity/profile mismatch not rejected')
  except ValueError:pass
 assert not torch.cuda.is_initialized()
 proof=dict(status='V77_REPEATED_ORDER_FROZEN_PROFILE_ENGINE_CPU_PASS',partial_teacher_boundary_exact=True,
  teacher_unique3_head_repeated5_source_exposure_exact=True,populated7Adam_enabledCPUscaler_ALLRNG_parentlinked_exact_nextupdate=True,
  repeated_indices_not_deduplicated_or_rewound=True,AMPskip_consumes_repeated_attempt_cursor_and_exposure=True,
  frozen_profile_exact_and_not_current_pool_weights=True,wrong_weight_source_epoch_restore_rejected=True,
  teacher_head_inflightSTOP_exact_restore_NOreplay=True,only7readout_gradients_upstream_frozen=True,
  actual_large_model_forwards=0,CUDA_initialized=False,actual_fullsize_worker_ready=False,
  fixture_calls=dict(encode=8,readout=8),seconds=time.monotonic()-started,fixture_output_bytes=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file()),
  files_SHA={str(f):support.sha(f) for f in [ROOT/'scripts/update_balance_engine_v77.py',ROOT/'research/update_balance_objective_v77.py',ROOT/'research/critical_retention_objective_v76.py',Path(__file__)]})
 target=directory.parent/'root_engine_CPU_contract.json';assert not target.exists();target.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))
if __name__=='__main__':main()
