import copy,hashlib,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_owt_reconstruction_continuation_v21 import fixture as make_source
from test_owt_prefix_reconstruction_v19 import Budget
from test_decoder_alignment_runtime_v8 import Child
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE,OBJECTIVE
from owt_fuzzy_engine_v22 import ContinuationEngine,COMPONENTS,cumulative_count
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261028_v22 as worker
import supervise_owt_fuzzy_reconstruction_20261028_v22 as owner
torch.set_num_threads(2)

def fixture():
    source,_,_=make_source();source.step();source.step();state=copy.deepcopy(source.export())
    model=PrefixRobustFuzzyAE(source.model.c)
    engine=ContinuationEngine(model,source.features,source.targets,worker.finite_orders(23,[301,302,303,304]),Budget(),'CPU22','cpu')
    digest=engine.import_parent(state,'CPU21','CPU_SOURCE21');return engine,state,digest

def test_actual_V21_schema_all_components_ancestry_and_source_immutability():
    e,s,digest=fixture();out=e.export()
    assert s['schema']=='fdt_owt_fuzzy_AE_v21_fullstate' and out['schema']=='fdt_owt_fuzzy_AE_v22_fullstate'
    assert digest==support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:out[k] for k in COMPONENTS})
    assert cumulative_count(s,'updates')==20 and cumulative_count(out,'updates')==20
    assert e.ancestry['ancestry']==s['ancestry'] and e.attempts==e.updates==e.cursor==0
    assert sum(e.budget.allocation_calls.values())==0 and out['training_objective']==OBJECTIVE
    before=support.state_sha(s);record=e.step();assert support.state_sha(s)==before
    assert e.budget.allocation_calls=={'encode':1,'denoise':0,'decoder':2}
    assert abs(record['loss']-.5*(record['clean_CE']+record['corrupted_prefix_CE']))<1e-5
    e.verify_frozen()

def test_new_fullstate_mask_rng_exact_next_last7_and_separate_budget(tmp_path):
    e,s,_=fixture();e.step();e.step();saved=support.save_full(tmp_path,copy.deepcopy(e.export()))
    snap=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    expected=e.step();final=copy.deepcopy(e.export());other,_,_=fixture();other.restore(snap)
    assert other.step()==expected and support.state_sha(other.export())==support.state_sha(final)
    while other.attempts<len(other.orders):other.step()
    assert other.cursor==92 and other.updates==12 and cumulative_count(other.export(),'updates')==32
    assert other.budget.allocation_calls=={'encode':12,'denoise':0,'decoder':24}
    assert other.budget.calls['encode']==s['actual_calls']['encode']+12
    assert other.exposure['eligible_prefix_bytes']==other.exposure['target_UTF8_bytes']

def test_same_data_finite_exposure_and_fixed_train_reconstruction_indices():
    artifact=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v20'
    f=json.loads((artifact/'features.json').read_text(encoding='utf-8'))
    y=json.loads((artifact/'targets.json').read_text(encoding='utf-8'))
    orders=worker.finite_orders(16384,range(301,305));assert len(orders)==8192
    for epoch in range(4):assert sorted(i for b in orders[epoch*2048:(epoch+1)*2048] for i in b)==list(range(16384))
    assert worker.COST['training']==24576 and worker.COST['total_max']==26776 and worker.LIMITS['max_calls']==28000
    assert 4*sum(len(x['prompt'].encode()) for x in f['train'])==5837512
    ids=worker.TRAIN_RECONSTRUCTION_INDICES
    assert ids==[0,4,8,12,16,20,24,28,8192,8196,8200,8204,8208,8212,8216,8220]
    assert len({y['train'][i]['document_id'] for i in ids})==16
    assert all(set(f['train'][i])=={'id','prompt'} for i in ids)


def test_fixed_training_reconstruction_takes_only_features_and_same_input_state():
    import inspect
    calls=Budget()
    class Model:
        def eval(self):return self
        def extract(self,tokens):return {'same_input_tokens':tokens.clone()}
        def reconstruct(self,meaning,charge,mode):
            assert mode=='normal' and meaning['same_input_tokens'].shape==(4,128)
            charge('decoder')
            return torch.tensor([[120,256]+[257]*127]*4)
    rows=[{'id':str(i),'prompt':'Known input sentence '+str(i)} for i in range(16)]
    assert 'targets' not in inspect.signature(worker.reconstruct_training_sentences).parameters
    out=worker.reconstruct_training_sentences(Model(),rows,calls,'cpu')
    assert [x['source'] for x in out]==[r['prompt'] for r in rows]
    assert len(out)==16 and calls.allocation_calls=={'encode':4,'denoise':0,'decoder':4}
    assert all(r['generated']['text']=='x' and r['generated']['eos'] for r in out)
    with pytest.raises(ValueError):worker.reconstruct_training_sentences(Model(),[dict(r,target='FORBIDDEN') for r in rows],Budget(),'cpu')


@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_V22_plan_postPopen_owner_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261028_v22/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');owner.validate(path,tmp_path)
    worker.validate(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v22.py');patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V22_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v22.py')
        result=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and result['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
