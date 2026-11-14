"""Pin actual V61 scope; one new clock only after actual worker/owner CPU proof."""
import argparse,ast,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_semantic_binding_20261111_v61 import validate,SOURCE_SHA,BINDER_SHA


def main(seal):
    data=ROOT/'research/artifacts/semantic_binding_20261111_v61_cpu';review=data/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text(encoding='utf-8-sig'));output=Path(p['output']);validate(unsealed,output)
        if output.exists():raise FileExistsError('closed output cannot reopen')
        proof=review/'cpu_preflight_owned/final_status.json';r=json.loads(proof.read_text(encoding='utf-8-sig'));report=r['report']
        if (r['status']!='CPU_PREFLIGHT_PASS' or not r['exit']['all_owned_processes_gone'] or r['owner_imported_Torch']
            or not report['fullsize_engine_restore'] or report['parent_Adam_used_for_training'] or not report['new_optimizer_fresh']
            or not report['all_V60_binder_tensors_exact'] or not report['new_residual_and_interior_zero']
            or report['new_semantic_parameters']!=504885 or report['engine_schema']!='fdt_semantic_binding_v61_fullstate'
            or report['graph_contract']!='coupled_spans_fourrole_memory_readout'):raise ValueError('actual V61 proof required')
        p['files'][str(proof)]=support.sha(proof);now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in ('research/semantic_binding_v61.py','research/semantic_binding_data_v61.py','scripts/semantic_binding_engine_v61.py',
            'scripts/run_semantic_binding_20261111_v61.py','scripts/supervise_semantic_binding_v61.py','scripts/prepare_semantic_binding_plan_v61.py'):
            source=ROOT/name;(snapshots/source.name).write_bytes(source.read_bytes())
        result=dict(plan_sha=support.sha(path),new_scope='joint entity intervals + shared interior + contextual sequence routing',
            cost1340=True,fresh_optimizer=True,no_language_or_denoising=True)
        (output/'root_prelaunch_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result));return
    if unsealed.exists():raise FileExistsError('immutable prospective plan')
    old=json.loads((ROOT/'runs/semantic_binding_20261111_v60/plan.json').read_text(encoding='utf-8-sig'))
    source=Path(old['source']);head=Path(old['head_source']);binder=ROOT/'runs/semantic_binding_20261111_v60/model/checkpoints/state_0576.pt'
    output=ROOT/'runs/semantic_binding_20261111_v61'
    pins={name:digest for name,digest in old['files'].items() if Path(name).suffix=='.py'}
    names=('research/semantic_binding_v61.py','research/semantic_binding_data_v61.py','scripts/semantic_binding_engine_v61.py',
        'scripts/run_semantic_binding_20261111_v61.py','scripts/supervise_semantic_binding_v61.py','scripts/test_semantic_binding_v61_cpu.py',
        'scripts/test_joint_intervals_v61_cpu.py','scripts/review_semantic_binding_data_v61_cpu.py','scripts/prepare_semantic_binding_v61_code_cpu.py')
    files=[ROOT/n for n in names]+[Path(__file__),source,head,binder,ROOT/'runs/semantic_binding_20261111_v60/root_saved_review.json']
    for name in ('reused_dev_features','reused_dev_targets'):files.append(Path(old[name]))
    for name in ('manifest','train_features','train_targets','train_groups','orders','dev_features','dev_targets','dev_groups',
        'root_joint_math_contract','root_CPU_contract','root_data_review','OWT_train_reconstruction_windows'):files.append(data/(name+'.json'))
    for file in files:pins[str(file)]=support.sha(file)
    for name,digest in pins.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    p=dict(schema='fdt_semantic_binding_v61',authorized=False,root=str(ROOT),output=str(output),data=str(data),
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=support.sha(head),binder_source=str(binder),binder_sha=BINDER_SHA,
        reused_dev_features=old['reused_dev_features'],reused_dev_targets=old['reused_dev_targets'],files=pins,
        capacity=16384,hidden=128,attempts=1024,epochs=4,batch=16,groups_per_batch=4,amp_dtype='bfloat16',
        wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=1500,max_output_bytes=5*1024**3,cost=1340,
        cost_detail=dict(frozen_source_batches=256,semantic_training=1024,baseline_and_final=32,three_read_controls=12,event_interchange_readout=16),
        purpose='fix entity-boundary overextension with coupled ordered intervals, shared entity interior and bidirectional context; factors still consume bound three-memory state only',
        exposure=dict(authored_TRAIN_rows=4096,authored_groups=1024,authored_unique_meanings=2008,row_selections=16384,
            cached_TRAIN_input_UTF8_bytes=manifest['TRAIN_input_bytes']*4,original_source_encoder_batches=256,OWT_windows_used=0),
        architecture='warm V60 all tensors + zero residual BiGRU and shared/entity interior; no span-length hard cap or gold candidate filtering; O(L) interval partition',
        feature_contract='id+prompt source extraction; frozen K_C memberships/values + K_E neural keys/values + K_P prior; source-only UTF8 boundary metadata',
        optimizer='fresh AdamW3e-4 WD.01 clip1 GradScaler1024 BF16; GRU float32 packed; V60 oldAdam preserved closed',
        objective='joint ordered-interval NLL + relation CE + open-polarity BCE +2 quantityCE +.1 SupCon +.25 shared/role interior BCE',
        evaluation='unchanged already observed DEV256; all predicted-only; restricted operational semantic extraction, not natural language generation',
        recovery='all new NN/frozen prior/Adam/scaler/allRNG/orders/history/phase/cache refs/extraction journal/cursor/exposure/call budget',
        semantic_ontology=['give','open','contain'],role_fields=['primary_entity','recipient_optional','theme'],quantity_range=[1,24],
        max_checkpoint_writes=6,source_optimizer_updates=0,language_decoder_updates=0,denoise_calls=0,source_copy_bytes=0,
        limitations='adaptive simultaneous model+TRAIN coverage change, not isolated causal attribution or blind evaluation; lengths26..145B do not establish long-document semantics; original16384Bcodec preserved',
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    review.mkdir();unsealed.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');validate(unsealed,output)
    print(json.dumps(dict(actual_worker_schema_PASS=True,cost=p['cost_detail'],total=p['cost'],
        cache_bytes_per_copy=manifest['expected_packed_cache_tensor_bytes'],output_cap_bytes=p['max_output_bytes'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
