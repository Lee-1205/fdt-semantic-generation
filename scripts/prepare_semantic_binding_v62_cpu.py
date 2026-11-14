"""New orders and NEW worker/owner; no changes to immutable V61 run files."""
import ast,hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def emit(name,text):
    path=ROOT/name
    if path.exists():raise FileExistsError(path)
    ast.parse(text);path.write_text(text,encoding='utf8')


def main():
    olddata=ROOT/'research/artifacts/semantic_binding_20261111_v61_cpu'
    data=ROOT/'research/artifacts/semantic_binding_20261112_v62_cpu'
    if data.exists():raise FileExistsError('immutable V62 curriculum')
    groups=json.loads((olddata/'train_groups.json').read_text(encoding='utf-8-sig'));orders=[]
    for epoch in range(8):
        shuffled=list(groups);random.Random(62211+epoch).shuffle(shuffled)
        orders.extend([sum(shuffled[i:i+4],[]) for i in range(0,len(shuffled),4)])
    assert len(orders)==2048
    for epoch in range(8):
        batches=orders[epoch*256:(epoch+1)*256]
        assert sorted(sum(batches,[]))==list(range(4096))
        assert all(all(not(set(b)&set(g)) or set(b).issuperset(g) for g in groups) for b in batches)
    data.mkdir();(data/'orders.json').write_text(json.dumps(orders)+'\n',encoding='utf8')
    refs={}
    for name in ('train_features','train_targets','train_groups','dev_features','dev_targets','dev_groups','root_data_review'):
        path=olddata/(name+'.json');refs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(schema='semantic_binding_v62_reused_data_CPU',data=str(olddata),epochs=8,attempts=2048,
        TRAIN_rows=4096,groups=1024,unique_authored_meanings=2008,row_selections=32768,cached_source_UTF8_bytes=2396648,
        source_encode_calls=0,reused_frozen_features=True,new_rows=0,held_changes=0,refs=refs,
        objective='memory_only_contrast_span2_inside05',inference_architecture_unchanged=True,
        limitation='continued restricted adaptive TRAIN/observed DEV; no general language or long-document semantic claim')
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    engine=(ROOT/'scripts/semantic_binding_engine_v61.py').read_text(encoding='utf8')
    engine=engine.replace('from research.semantic_binding_v61 import training_loss','from research.semantic_binding_objective_v62 import training_loss,OBJECTIVE')
    engine=engine.replace('semantic_binding_v61_fullstate','semantic_binding_v62_fullstate').replace('lr=3e-4','lr=1e-4')
    engine=engine.replace('ontology=[\'give\',\'open\',\'contain\'],graph_contract=', 'objective_contract=OBJECTIVE,ontology=[\'give\',\'open\',\'contain\'],graph_contract=')
    engine=engine.replace("if (state['schema']!=SCHEMA", "if (state['objective_contract']!=OBJECTIVE or state['schema']!=SCHEMA")
    emit('scripts/semantic_binding_engine_v62.py',engine)
    owner=(ROOT/'scripts/supervise_semantic_binding_v61.py').read_text(encoding='utf8')
    owner=owner.replace('v61','v62').replace('1500','2200').replace("p['attempts']!=1024","p['attempts']!=2048")
    owner=owner.replace("p['epochs']!=4","p['epochs']!=8").replace('5*1024**3','1*1024**3')
    owner=owner.replace('run_semantic_binding_20261111_v62.py','run_semantic_binding_20261112_v62.py')
    emit('scripts/supervise_semantic_binding_v62.py',owner)
    test=(ROOT/'scripts/test_semantic_binding_v61_cpu.py').read_text(encoding='utf8')
    test=test.replace('semantic_binding_engine_v61','semantic_binding_engine_v62').replace('semantic_binding_20261111_v61_cpu','semantic_binding_20261112_v62_cpu')
    test=test.replace("    report=dict(","    from research.semantic_binding_objective_v62 import memory_contrast\n    identity=torch.tensor([0,0])\n    value=memory_contrast(reference.role_values,identity)\n    assert torch.autograd.grad(value,factors['relation_logits'],allow_unused=True,retain_graph=True)[0] is None\n    labels=reference.relation_probabilities.detach().clone().flip(-1)\n    altered=replace(reference,relation_probabilities=labels)\n    assert torch.equal(value,memory_contrast(altered.role_values,identity))\n    probe=torch.randn(4,4,128,requires_grad=True)\n    memory_contrast(probe,torch.tensor([0,0,1,1])).backward()\n    assert probe.grad.abs().sum()>0\n    report=dict(contrast_excludes_factor_heads=True,contrast_reaches_actual_memory=True,")
    emit('scripts/test_semantic_binding_v62_cpu.py',test)
    print(json.dumps(manifest))

if __name__=='__main__':main()
