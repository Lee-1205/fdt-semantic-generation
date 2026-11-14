"""Actual V62 worker schema, fixed fresh clock, immutable cache references."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_semantic_binding_20261112_v62 import validate,SOURCE_SHA,BINDER_SHA,OBJECTIVE


def main(seal):
    curriculum=ROOT/'research/artifacts/semantic_binding_20261112_v62_cpu';review=curriculum/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text(encoding='utf8'));output=Path(p['output']);validate(unsealed,output)
        if output.exists():raise FileExistsError('new output only')
        proof=review/'cpu_preflight_owned/final_status.json';r=json.loads(proof.read_text(encoding='utf-8-sig'));report=r['report']
        if (r['status']!='CPU_PREFLIGHT_PASS' or not r['exit']['all_owned_processes_gone'] or r['owner_imported_Torch']
            or not report['fullsize_engine_restore'] or report['parent_Adam_used_for_training'] or not report['new_optimizer_fresh']
            or not report['all_V61_binder_tensors_exact'] or not report['inference_architecture_unchanged']
            or report['new_semantic_parameters']!=504885 or report['engine_schema']!='fdt_semantic_binding_v62_fullstate'
            or report['objective_contract']!=OBJECTIVE or not report['baseline_reused_not_new_assessment']):raise ValueError('actual V62 owned CPU proof')
        p['files'][str(proof)]=support.sha(proof);now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in ('research/semantic_binding_objective_v62.py','research/semantic_binding_v61.py','scripts/semantic_binding_engine_v62.py',
            'scripts/run_semantic_binding_20261112_v62.py','scripts/supervise_semantic_binding_v62.py','scripts/prepare_semantic_binding_plan_v62.py'):
            source=ROOT/name;(snapshots/source.name).write_bytes(source.read_bytes())
        result=dict(plan_sha=support.sha(path),purpose=OBJECTIVE,new_finite2048=True,cost2092=True,baseline_is_reused=True,source_encoder_calls=0)
        (output/'root_prelaunch_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result));return
    if unsealed.exists():raise FileExistsError('immutable prospective contract')
    oldrun=ROOT/'runs/semantic_binding_20261111_v61';old=json.loads((oldrun/'plan.json').read_text(encoding='utf8'))
    output=ROOT/'runs/semantic_binding_20261112_v62';data=Path(old['data']);binder=oldrun/'model/checkpoints/state_1024.pt'
    source,head=Path(old['source']),Path(old['head_source'])
    pins={name:digest for name,digest in old['files'].items() if name.endswith('.py')}
    names=('research/semantic_binding_objective_v62.py','scripts/semantic_binding_engine_v62.py','scripts/run_semantic_binding_20261112_v62.py',
        'scripts/supervise_semantic_binding_v62.py','scripts/test_semantic_binding_v62_cpu.py','scripts/prepare_semantic_binding_v62_cpu.py')
    files=[ROOT/name for name in names]+[Path(__file__),source,head,binder,oldrun/'root_saved_review.json',curriculum/'orders.json',curriculum/'manifest.json',curriculum/'root_CPU_contract.json']
    refs={}
    for label in ('train','dev'):
        for kind in ('features','targets'):
            path=oldrun/(label+'_'+kind+'.pt') if label=='train' else Path(old['reused_dev_'+kind])
            refs['reused_'+label+'_'+kind]=str(path);files.append(path)
    baseline={name:dict(path=str(oldrun/file),sha256=support.sha(oldrun/file)) for name,file in (
        ('metrics','final_metrics.json'),('raw','raw_final.pt'),('predictions','final_predictions.json'))}
    files.extend(Path(ref['path']) for ref in baseline.values())
    manifest=json.loads((curriculum/'manifest.json').read_text(encoding='utf8'))
    files.extend(Path(name) for name in manifest['refs'])
    for file in files:pins[str(file)]=support.sha(file)
    for name,digest in pins.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_semantic_binding_v62',authorized=False,root=str(ROOT),output=str(output),data=str(data),orders=str(curriculum/'orders.json'),
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=support.sha(head),binder_source=str(binder),binder_sha=BINDER_SHA,
        files=pins,**refs,baseline_reference=baseline,objective_contract=OBJECTIVE,
        capacity=16384,hidden=128,attempts=2048,epochs=8,batch=16,groups_per_batch=4,amp_dtype='bfloat16',
        wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=2200,max_output_bytes=1024**3,cost=2092,
        cost_detail=dict(training_bind=2048,final_bind=16,three_read_controls=12,event_interchange_readout=16,source_encode=0,fresh_baseline=0),
        purpose='align actual bound memory without factor probability shortcut; increase joint entity span/interior emphasis over a new finite exposure',
        optimizer='fresh AdamW1e-4 WD.01 clip1 GradScaler1024 BF16 with V61NNunchanged; preserve old V61 Adam/RNG closed',
        objective='2 joint span NLL + relation CE + open polarity BCE +2 quantityCE +.5 interiorBCE +.2 MEMORY-ONLY SupCon(temp.2)',
        exposure=dict(TRAIN_rows=4096,groups=1024,unique_authored_meanings=2008,row_selections=32768,cached_source_UTF8_bytes=2396648,new_rows=0,OWT_rows=0),
        baseline_contract='reuse unchanged V61 final raw/metrics under identical architecture/tensors/cache; NOT a fresh independent baseline assessment',
        semantic_ontology=['give','open','contain'],role_fields=['primary_entity','recipient_optional','theme'],quantity_range=[1,24],
        recovery='model/frozenprior/newAdam/scaler/allRNG/phase/orders/history/cursor/call/exposure/cache refs/baseline reference and objective signature',
        max_checkpoint_writes=6,source_optimizer_updates=0,language_decoder_updates=0,denoise_calls=0,source_copy_bytes=0,
        limitations='restricted adaptive already observed DEV; objectiveweights plus further finite learning both change; not isolated causal contribution, general semantics, long-document semantics or language realization',
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    review.mkdir();unsealed.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');validate(unsealed,output)
    print(json.dumps(dict(actual_worker_schema_PASS=True,cost=p['cost_detail'],total2092=True,max_calls=2200,output_cap=p['max_output_bytes'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
