import copy,hashlib,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_owt_prefix_reconstruction_v19 import fixture as make_source,Budget
from test_decoder_alignment_runtime_v8 import Child
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE,OBJECTIVE
from owt_fuzzy_engine_v20 import ContinuationEngine,COMPONENTS,cumulative_count
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261028_v20 as worker
import supervise_owt_fuzzy_reconstruction_20261028_v20 as owner
torch.set_num_threads(2)

def fixture():
    source,_,_=make_source();source.step();source.step();state=copy.deepcopy(source.export())
    model=PrefixRobustFuzzyAE(source.model.c)
    engine=ContinuationEngine(model,source.features,source.targets,worker.finite_orders(23,[281,282]),Budget(),'CPU20','cpu')
    digest=engine.import_parent(state,'CPU19','CPU_SOURCE19');return engine,state,digest

def test_actual_V19_schema_all_components_ancestry_and_source_immutability():
    e,s,digest=fixture();out=e.export()
    assert s['schema']=='fdt_owt_fuzzy_AE_v19_fullstate' and out['schema']=='fdt_owt_fuzzy_AE_v20_fullstate'
    assert digest==support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:out[k] for k in COMPONENTS})
    assert cumulative_count(s,'updates')==16 and cumulative_count(out,'updates')==16
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
    assert other.cursor==46 and other.updates==6 and cumulative_count(other.export(),'updates')==22
    assert other.budget.allocation_calls=={'encode':6,'denoise':0,'decoder':12}
    assert other.budget.calls['encode']==s['actual_calls']['encode']+6
    assert other.exposure['eligible_prefix_bytes']==other.exposure['target_UTF8_bytes']

def test_expanded_documents_old_prefix_held_source_hashes_and_finite_exposure():
    artifact=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v20';old=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v15'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    docs=load(artifact/'documents.json');features=load(artifact/'features.json');targets=load(artifact/'targets.json')
    for name,newrows in (('documents.json',docs),('features.json',features),('targets.json',targets)):
        before=load(old/name);assert newrows['dev']==before['dev'] and newrows['train'][:len(before['train'])]==before['train']
    for panel,rows in docs.items():
        for row in rows:
            assert hashlib.sha256(row['text'].encode()).hexdigest()==row['raw_sha256']
            h=hashlib.sha256(' '.join(row['text'].split()).encode()).hexdigest();assert h==row['normalized_sha256']
            assert (int(h[:8],16)%9==0)==(panel=='dev')
    assert len({x['normalized_sha256'] for rows in docs.values() for x in rows})==4160
    assert len({x['id'] for rows in features.values() for x in rows})==16640
    assert len(docs['train'])==4096 and len(features['train'])==16384 and len(targets['train'])==16384
    orders=worker.finite_orders(16384,range(281,283));assert len(orders)==4096
    for epoch in range(2):assert sorted(i for b in orders[epoch*2048:(epoch+1)*2048] for i in b)==list(range(16384))
    assert worker.COST['training']==3*4096 and worker.COST['total_max']==13968 and worker.LIMITS['max_calls']==14800
    assert 2*sum(len(x['prompt'].encode()) for x in features['train'])==2918756

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_V20_plan_postPopen_owner_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261028_v20/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');owner.validate(path,tmp_path)
    worker.validate(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v20.py');patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V20_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v20.py')
        result=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and result['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
