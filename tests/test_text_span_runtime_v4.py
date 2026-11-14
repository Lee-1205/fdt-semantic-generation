import copy,json,sys,time,random
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_span_reader_v4 import SpanSemanticGenerator
from research.text_span_data_v4 import query_spans
from research.text_generation_data_v1 import render
from research.text_diffusion_v1 import pack_prompts
from text_span_engine_v4 import PhaseEngine
import run_text_span_alignment_20261028 as worker
import supervise_text_span_alignment_20261028 as owner
import text_generation_support_v1 as support
torch.set_num_threads(2)

class FakeBudget:
 def __init__(self):self.calls={'reader':0,'decoder':0,'encode':0,'denoise':0};self.stopped=False
 def check(self):
  if self.stopped:raise InterruptedError('CPU fixture STOP')
 def charge(self,kind):
  self.check()
  if sum(self.calls.values())>=2800:raise RuntimeError('actual cap')
  self.calls[kind]+=1

def fixture():
 torch.manual_seed(91);random.seed(91);np.random.seed(91)
 features=[];targets=[]
 for i in range(8):
  p,a,label=render('A','H','greater' if i%2==0 else 'less',bool(i%2),'en' if i<4 else 'ko',0)
  row={'prompt':p,'answer':a,'language':'en' if i<4 else 'ko','label':label}
  features.append({'id':str(i),'prompt':p});targets.append(query_spans(row))
 tokens=pack_prompts([r['prompt'] for r in features]);mask=torch.cat((tokens!=256,torch.ones(8,1,dtype=torch.bool)),1)
 cache={'train':{'prompt':torch.randn(8,161,32),'mask':mask,'latent':torch.randn(8,40,32)}}
 model=SpanSemanticGenerator(Config.smoke());frozen={'CPU_FIXTURE_ONLY':torch.ones(3)}
 orders={'reader':list(range(8))*512,'realizer':list(range(8))*512};pairs={'train':[]}
 eng=PhaseEngine(model,frozen,{'train':features},{'train':targets},pairs,cache,orders,FakeBudget(),'CPU_FIXTURE_PLAN','cpu')
 return eng

def position_fixture(engine,phase,attempt):
 # Synthetic CPU control position, not evidence of512 real model updates.
 if phase=='reader':engine.attempts_by_phase['reader']=attempt;engine.updates_by_phase['reader']=attempt;engine.cursors['reader']=attempt*8
 else:
  engine.attempts_by_phase['realizer']=attempt;engine.updates_by_phase['realizer']=attempt;engine.cursors['realizer']=attempt*8
 engine.history=[]
 for p in ('reader','realizer'):
  for n in range(1,engine.attempts_by_phase[p]+1):
   engine.history.append({'global_attempt':len(engine.history)+1,'phase':p,'phase_attempt':n,'CPU_fixture_control_record':True})

def test_reader_fullstate_restore_exact_next_step_and_parameter_ownership(tmp_path):
 e=fixture();e.step();assert e.optimizers['reader'].state and e.scaler.state_dict()['scale']==1024
 p=copy.deepcopy(e.export());meta=support.save_full(tmp_path,p)
 loaded=torch.load(tmp_path/'checkpoints'/meta['file'],weights_only=True)
 new=fixture();new.restore(loaded)
 assert support.state_sha(new.export())==support.state_sha(p) and new.next_indices()==e.next_indices()
 assert all(not x.requires_grad for x in new.model.decoder.parameters())
 assert not new.model.reader.types.requires_grad
 assert all(not x.requires_grad for x in new.model.reader.values.parameters())

def test_boundary_keeps_populated_reader_optimizer_scaler_all_rng_and_clears_gradients(tmp_path):
 e=fixture();e.step();position_fixture(e,'reader',512)
 before_rng=torch.get_rng_state().clone();scale=copy.deepcopy(e.scaler.state_dict());reader_state=support.state_sha(e.optimizers['reader'].state_dict())
 e.finish_reader()
 assert torch.equal(before_rng,torch.get_rng_state()) and e.scaler.state_dict()==scale
 assert support.state_sha(e.optimizers['reader'].state_dict())==reader_state and not e.optimizers['realizer'].state
 assert all(p.grad is None and not p.requires_grad for p in e.model.reader.parameters())
 payload=copy.deepcopy(e.export());support.save_full(tmp_path,payload)
 new=fixture();new.restore(payload);assert support.state_sha(new.export())==support.state_sha(payload)
 new.step();assert new.attempts==513 and new.cursors['realizer']==8 and new.optimizers['realizer'].state
 assert support.state_sha(new.optimizers['reader'].state_dict())==reader_state
 new.verify_frozen()

def test_realizer_teacher_to_consistent_prediction_boundary_restores(tmp_path):
 e=fixture();e.step();position_fixture(e,'reader',512);e.finish_reader();e.step();position_fixture(e,'realizer',256)
 p=copy.deepcopy(e.export());new=fixture();new.restore(p)
 before=new.budget.calls['reader'];new.step()
 assert new.attempts==769 and new.attempts_by_phase['realizer']==257 and new.budget.calls['reader']==before+1
 assert new.history[-1]['teacher_condition_rows']+new.history[-1]['predicted_condition_rows']==8
 assert new.exposure['transition_reader_rows']==8
 new.verify_frozen();support.save_full(tmp_path,new.export())

def test_failed_attempt_is_durable_and_consumes_cursor_without_fake_update():
 e=fixture()
 with patch.object(e.model,'reader_loss',side_effect=FloatingPointError('CPU numeric fixture')):
  with pytest.raises(FloatingPointError):e.step()
 assert e.attempts==1 and e.updates==0 and e.cursor==8 and e.history[-1]['outcome']=='FAILED_ATTEMPT'
 p=copy.deepcopy(e.export());new=fixture();new.restore(p)
 assert new.cursor==8 and new.attempts==1 and new.updates==0
 new.budget.stopped=True
 with pytest.raises(InterruptedError):new.step()
 assert new.attempts==1

def test_phase_cap_restore_rejects_changed_order_and_missing_scaler():
 e=fixture();e.step();p=copy.deepcopy(e.export());bad=copy.deepcopy(p);bad['scaler']={}
 with pytest.raises(ValueError):fixture().restore(bad)
 bad=copy.deepcopy(p);bad['orders']['reader'][0]=7
 with pytest.raises(ValueError):fixture().restore(bad)
 position_fixture(e,'reader',512)
 with pytest.raises(StopIteration):e.next_indices()

def test_enabled_scaler_amp_skips_consume_attempts_rows_and_stop_at_three():
 e=fixture();before=support.state_sha(e.model.state_dict())
 hook=e.model.reader.pointer_query.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
 try:
  e.step();e.step()
  with pytest.raises(FloatingPointError):e.step()
 finally:hook.remove()
 assert e.attempts==3 and e.updates==0 and e.cursor==24 and e.skips==3
 assert all(r['outcome']=='AMP_SKIP' for r in e.history)
 assert e.scaler.state_dict()['scale']==128 and not e.optimizers['reader'].state
 assert before==support.state_sha(e.model.state_dict()) and e.exposure['reader_rows']==24
 p=copy.deepcopy(e.export());restored=fixture();restored.restore(p)
 assert restored.skips==3 and restored.scaler.state_dict()['scale']==128

def test_all_evaluation_rows_use_predictions_only_and_headcheck_has_no_decoder():
 e=fixture();features={};targets={};cache={}
 for name,count in (('train_probe',16),('dev_pairs',32),('dev_wording',32)):
  features[name]=e.features['train']*(count//8);targets[name]=e.targets['train']*(count//8)
  cache[name]={k:v.repeat(count//8,*([1]*(v.ndim-1))) for k,v in e.cache['train'].items()}
 with patch.object(e.model,'teacher_context_for_training',side_effect=AssertionError('teacher in eval')),patch.object(e.model,'consistent_context_for_training',side_effect=AssertionError('teacher transition in eval')):
  a=worker.evaluate(e,features,targets,cache,e.budget)
  changed=copy.deepcopy(targets)
  for rows in changed.values():
   for row in rows:row['head_labels']=[7,6,2,1];row['answer']='Different gold.'
  b=worker.evaluate(e,features,changed,cache,e.budget)
 assert all([r['generated'] for r in a[k]['raw']]==[r['generated'] for r in b[k]['raw']] for k in a)
 previous=e.budget.calls['decoder']
 with patch.object(e.model.decoder,'generate',side_effect=AssertionError('generation in headcheck')):h=worker.evaluate(e,features,targets,cache,e.budget,heads_only=True)
 assert e.budget.calls['decoder']==previous and sum(len(v['raw']) for v in h.values())==80

class Child:
    pid=567;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0

@pytest.mark.parametrize('crash',[False,True])
def test_actual_v2_plan_schema_full_post_popen_cleanup(tmp_path,crash):
    pp=ROOT/'runs/text_span_alignment_20261028_v4/plan.json';plan=json.loads(pp.read_text(encoding='utf-8'))
    plan.update(output=str(tmp_path),authorized=True,allocation_start=time.time());plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    testplan=tmp_path/'plan.json';testplan.write_text(json.dumps(plan));owner.validate(testplan,tmp_path)
    worker.validate(testplan,tmp_path,ROOT/'scripts/run_text_span_alignment_20261028.py')
    patches=[]
    def popen(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def bad(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('V2postPopen')
                return original(path,value)
            monkey=patch.object(owner.io,'atomic_json',bad);monkey.start();patches.append(monkey)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'TEXT_SPAN_ALIGNMENT_V4_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=popen),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':True}) as cleanup:
            rc=owner.supervise(testplan,tmp_path,ROOT/'scripts/run_text_span_alignment_20261028.py')
        state=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert rc==(2 if crash else 0) and state['status']==('STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()

def test_expired_v2_rejects_before_popen(tmp_path):
    source=ROOT/'runs/text_span_alignment_20261028_v4/plan.json';plan=json.loads(source.read_text())
    plan.update(output=str(tmp_path),authorized=True,allocation_start=time.time()-1000)
    plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
    with patch.object(owner.subprocess,'Popen') as p:
        with pytest.raises(TimeoutError):owner.supervise(pp,tmp_path,ROOT/'scripts/run_text_span_alignment_20261028.py')
        p.assert_not_called()

@pytest.mark.skipif(sys.platform!='win32',reason='real Windows')
def test_v2_real_windows_complete_owner_cpu_only(tmp_path):
    out=tmp_path/'native';out.mkdir();stub=tmp_path/'stub.py'
    stub.write_text("import argparse,os,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()\nif a.preflight:print('CPU_PREFLIGHT_PASS');raise SystemExit(0)\no=Path(a.output);(o/'progress.json').write_text(json.dumps({'phase':'cpu','attempts':0,'native_pid':os.getpid()}));time.sleep(2);(o/'result.json').write_text(json.dumps({'status':'TEXT_SPAN_ALIGNMENT_V4_COMPLETE','model_calls':0}))\n")
    source=ROOT/'runs/text_span_alignment_20261028_v4/plan.json';plan=json.loads(source.read_text());plan.update(output=str(out),authorized=True,allocation_start=time.time());plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    plan['files']={str(Path(owner.__file__).resolve()):support.sha(owner.__file__),str(stub.resolve()):support.sha(stub)}
    pp=out/'plan.json';pp.write_text(json.dumps(plan))
    with patch.object(owner,'resources',return_value={'cpu_fixture':True}):rc=owner.supervise(pp,out,stub)
    result=json.loads((out/'supervisor/final_status.json').read_text());assert rc==0 and result['exit']['all_owned_processes_gone']
