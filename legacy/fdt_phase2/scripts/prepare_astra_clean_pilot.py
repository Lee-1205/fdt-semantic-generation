"""Seal a fresh paired plan on repaired data; carry all prior budget charges."""
from pathlib import Path
import argparse,json,hashlib,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from astra_bounded_pilot import digest,restrict_row
from astra_evidence_io import save
from train_fdt_v4_1_retention import RowPool,TokenBudgetSourcePlanner

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def remaining_limits(stop):
    used=int(stop['reserved_supervised_all_attempts'])
    if stop['B']!='NOT_STARTED' or not 0<=used<250000:raise ValueError('unexpected prior arm accounting')
    seconds=14400-float(stop['charged_execution_seconds_with_shutdown_allowances'])
    if seconds<=0:raise ValueError('no GPU budget remains')
    return 250000-used,seconds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(3)
    if a.out.exists():raise FileExistsError(a.out)
    stop=read(a.prior/'DATA_SEMANTICS_STOP.json');limit,seconds=remaining_limits(stop)
    old=read(a.prior/'data_plan_resume.json');candidate=read(a.candidate/'candidate_manifest.json')
    preflight=read(a.candidate/'candidate_preflight.json')
    if preflight['status']!='CPU_PREFLIGHT_PASS' or preflight['exact_token_overlap_matches']:raise ValueError('candidate preflight')
    if candidate['source_pilot_stop_sha256']!=digest(a.prior/'DATA_SEMANTICS_STOP.json'):raise ValueError('stop lineage drift')
    manifest=read(Path(old['parent']).parent/'run_manifest.json')
    fractions=manifest['source_token_fractions'];planner=TokenBudgetSourcePlanner(fractions)
    pools={};shards={}
    for i,(name,folder) in enumerate(candidate['dataset_dirs'].items()):
        folder=Path(folder);metadata=read(folder/'manifest.json')
        pool=RowPool(folder,'train',20260918+i*1009)
        if len(pool.paths)!=1:raise ValueError('single shard contract')
        if digest(pool.paths[0]).lower()!=metadata['train']['shard_sha256'].lower():raise ValueError('candidate shard drift')
        pools[name]=pool;shards[name]={'path':str(pool.paths[0]),'sha256':digest(pool.paths[0]),
            'manifest_sha256':digest(folder/'manifest.json'),'initial_state':pool.state()}
    rows=[];active=supervised=0;seen=set()
    while supervised<limit:
        name=planner.choose_source();p=pools[name]
        if p.rows_consumed>=p.total_rows:raise ValueError('source epoch would exceed one')
        index=int(p.order[p.cursor]);row=p.next_batch(1)
        selected,stats=restrict_row(row,limit-supervised,1200000-active)
        if selected is None:raise ValueError('active limit before endpoint')
        if (name,index) in seen:raise ValueError('replayed row')
        seen.add((name,index));planner.record(name,stats['active'])
        rows.append({'source':name,'row_id':index,**stats,
            'label_mask_sha256':hashlib.sha256(selected['labels'].numpy().tobytes()).hexdigest()})
        active+=stats['active'];supervised+=stats['supervised']
    def content_hash(pool,index):
        return hashlib.sha256(pool.payload['input_ids'][index][pool.payload['attention_mask'][index].bool()].to(torch.int32).numpy().tobytes()).hexdigest()
    training_content={content_hash(pools[r['source']],r['row_id']) for r in rows}
    validation={};sources={}
    for name,p in pools.items():
        selected=[];excluded=set(training_content)
        for i in p.order[p.cursor:]:
            index=int(i);key=content_hash(p,index)
            if key in excluded:continue
            selected.append({'row_id':index,'active_input_sha256':key});excluded.add(key)
            if len(selected)==16:break
        if len(selected)!=16:raise ValueError('insufficient distinct validation rows')
        validation[name]={'kind':'repaired_data_unconsumed_content_disjoint','global_training_history':'UNKNOWN','rows':selected}
        selected_rows=[r for r in rows if r['source']==name]
        sources[name]={'rows':len(selected_rows),'active':sum(r['active'] for r in selected_rows),
            'supervised':sum(r['supervised'] for r in selected_rows),'epoch_fraction':len(selected_rows)/p.total_rows,
            'replay_count':0,'final_pool_state':p.state()}
    plan={k:old[k] for k in ['parent','parent_sha256','fixed52','tokenizer_dir','tokenizer_sha256','config']}
    plan.update(seed=20260918,active=active,supervised=supervised,rows=rows,shards=shards,validation=validation,sources=sources,
        source_token_fractions=fractions,candidate_manifest_sha256=digest(a.candidate/'candidate_manifest.json'),
        candidate_preflight_sha256=digest(a.candidate/'candidate_preflight.json'),
        comparison='fresh repaired-data A/B; never merge with the aborted original-data trial',
        planner_restart='fresh zeroed fair queue on new candidate row indices; identical in A and B',
        global_lineage_overlap='UNKNOWN',holdout_reuse='same preregistered 192; parent outputs already inspected, not a new blind test')
    a.out.mkdir(parents=True)
    save(a.out/'data_plan.json',plan);save(a.out/'holdout.json',read(a.prior/'holdout.json'))
    save(a.out/'budget_accounting.json',{'prior_dir':str(a.prior.resolve()),
        'prior_stop_sha256':digest(a.prior/'DATA_SEMANTICS_STOP.json'),
        'prior_charged_tokens':stop['reserved_supervised_all_attempts'],
        'prior_charged_seconds':stop['charged_execution_seconds_with_shutdown_allowances'],
        'remaining_gpu_seconds':seconds,'per_arm_supervised_limit':limit,
        'maximum_combined_charged_tokens':stop['reserved_supervised_all_attempts']+2*limit,
        'authorized_total_seconds':14400,'automatic_promotion':False})
    print(json.dumps({'supervised':supervised,'active':active,'rows':len(rows),'sources':sources,'seconds':seconds}))

if __name__=='__main__':main()
