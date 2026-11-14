"""Pin actual semantic-only scope. --seal after CPU source and recovery proof."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_semantic_binding_20261111_v60 import validate,SOURCE_SHA,BINDER_SHA

def main(seal):
    data=ROOT/'research/artifacts/semantic_binding_20261111_v60_cpu';review=data/'execution_review_v2';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text(encoding='utf-8-sig'));output=Path(p['output']);validate(unsealed,output)
        if output.exists():raise FileExistsError('closed budget cannot reopen')
        proof=review/'cpu_preflight_owned/final_status.json';r=json.loads(proof.read_text(encoding='utf-8-sig'))
        if (r['status']!='CPU_PREFLIGHT_PASS' or not r['exit']['all_owned_processes_gone'] or r['owner_imported_Torch']
            or not r['report']['fullsize_engine_restore'] or r['report']['parent_Adam_used_for_training'] or not r['report']['new_optimizer_fresh']
            or r['report']['engine_schema']!='fdt_semantic_binding_v60_fullstate'
            or r['report']['graph_contract']!='fourrole_memory_only_fact_readout'):raise ValueError('actual owned CPU proof')
        p['files'][str(proof)]=support.sha(proof);now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for file in ('research/semantic_binding_v60.py','scripts/semantic_binding_engine_v60.py','scripts/run_semantic_binding_20261111_v60.py',
            'scripts/supervise_semantic_binding_v60.py','scripts/prepare_semantic_binding_plan_v60.py','research/semantic_binding_data_v60.py'):
            source=ROOT/file;(snapshots/source.name).write_bytes(source.read_bytes())
        result=dict(plan_sha=support.sha(path),new_semantic_encoder_only=True,calls816=True,fresh_optimizer=True,source_and_language_frozen=True)
        (output/'root_prelaunch_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));return
    if unsealed.exists():raise FileExistsError('immutable prospective plan')
    oldrun=ROOT/'runs/broad_ordered_20261111_v57';old=json.loads((oldrun/'plan.json').read_text(encoding='utf-8-sig'))
    source=Path(old['source']);head=oldrun/'model/checkpoints/state_8192.pt';output=ROOT/'runs/semantic_binding_20261111_v60'
    binder=ROOT/'runs/semantic_binding_20261111_v59/model/checkpoints/state_0576.pt'
    reused_dev=ROOT/'runs/semantic_binding_20261111_v59/dev_features.pt';reused_targets=ROOT/'runs/semantic_binding_20261111_v59/dev_targets.pt'
    pins={k:v for k,v in old['files'].items() if Path(k).suffix=='.py'}
    names=('research/semantic_binding_v60.py','research/semantic_binding_data_v60.py','scripts/semantic_binding_engine_v60.py',
        'scripts/run_semantic_binding_20261111_v60.py','scripts/supervise_semantic_binding_v60.py',
        'scripts/test_semantic_binding_v60_cpu.py','scripts/review_semantic_binding_data_v60_cpu.py')
    files=[ROOT/name for name in names]+[Path(__file__),source,head,oldrun/'root_saved_review.json',binder,reused_dev,reused_targets,
        ROOT/'research/semantic_binding_v59.py',ROOT/'research/semantic_binding_data_v59.py',ROOT/'scripts/semantic_binding_engine_v59.py',
        ROOT/'runs/semantic_binding_20261111_v59/root_saved_review.json']
    files.extend(data/(name+'.json') for name in ('manifest','train_features','train_targets','train_groups','dev_features','dev_targets','dev_groups',
        'orders','root_CPU_contract','root_data_review','OWT_train_reconstruction_windows','preparation_incident_01','preparation_incident_02'))
    for file in files:pins[str(file)]=support.sha(file)
    for name,digest in pins.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_semantic_binding_v60',authorized=False,root=str(ROOT),output=str(output),data=str(data),
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=support.sha(head),files=pins,
        binder_source=str(binder),binder_sha=BINDER_SHA,reused_dev_features=str(reused_dev),reused_dev_targets=str(reused_targets),
        capacity=16384,hidden=128,attempts=576,epochs=4,batch=20,groups_per_batch=2,amp_dtype='bfloat16',
        wall_seconds=900,start=None,cutoff=None,deadline=None,max_calls=950,max_output_bytes=3*1024**3,cost=816,
        cost_detail={'frozen_source_batches':180,'semantic_training':576,'baseline_and_final':32,'three_read_controls':12,'event_interchange_readout':16},
        purpose='fix pre-bank factor bypass: all facts read bound three-memory state ONLY, learned fourth event/cue role; one-event extraction, no surface-code success claim',
        exposure=dict(authored_TRAIN_rows=2880,authored_fact_groups=288,row_selections=11520,cached_TRAIN_input_UTF8_bytes=manifest['TRAIN_input_bytes']*4,
            original_source_encoder_batches=180,OWT_reconstruction_windows_prepared_not_trained=733),
        feature_contract='frozen K_C token memberships/values and K_E neural keys/values + original frozen K_P prior; no rawH/goldrole/span/regexNNfeatures',
        semantic_ontology=['give','open','contain'],role_fields=['primary_entity','recipient_optional','theme'],quantity_range=[1,24],
        optimizer='warm V59 tensors +6 zero contextual gains; changed graph fresh AdamW3e-4 WD.01 clip1 GradScaler1024 BF16; all old optimizer trajectories remain closed/preserved',
        objective='UTF8 span CE + relation CE + open-polarity BCE + 2 contain-quantity CE + .1 supervised contrastive(temperature.2)',
        evaluation='held names/new template families and exact literal/factor facts; source/gold selection used only in CPU postscore; no generated-language claim',
        recovery='all new binder tensors/frozen prior/Adam/scaler/allRNG/source/cache refs/phase/extraction journal/orders/cursor/history/call exposure',
        max_checkpoint_writes=6,source_optimizer_updates=0,language_decoder_updates=0,denoise_calls=0,source_copy_bytes=0,
        limitations='restricted adaptive authored curriculum,64TRAIN/4DEV duplicate prompts; DEV files unchanged/held, OWT windows unlabelled and NOT used in this semantic trial',
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    review.mkdir(exist_ok=False)
    unsealed.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(unsealed,output)
    print(json.dumps(dict(actual_schema_PASS=True,cost=p['cost_detail'],total816=True)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
