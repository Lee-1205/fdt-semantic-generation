"""Shared budget/recovery and a sealed data plan for the authorized two-arm pilot."""
import hashlib
import json
import os
import random
import uuid
from pathlib import Path
import torch

SUPERVISED_LIMIT=250_000
ACTIVE_LIMIT=1_200_000
TOTAL_SECONDS=14_400
PARENT_NAME='fdt_v4_1_424m_alpha020_retention_smoke_500k_20260907_r1'
PARENT_SHA='50e084717a0cce99ea29b9e695e402c15e0a0807f3c64db795eab6d26093436d'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def restrict_row(row,remaining_supervised,remaining_active,pad=0):
    active=int(row['attention_mask'].sum())
    if remaining_supervised<=0 or active>remaining_active:return None,None
    result={k:v.clone() for k,v in row.items()}
    labels=result['labels']
    valid=row['attention_mask'].bool() & labels.ne(-100) & labels.ne(pad)
    valid[:,0]=False
    positions=valid.nonzero()
    count=min(len(positions),int(remaining_supervised))
    if count==0:raise ValueError('unsupervised row in sealed pilot stream')
    keep=torch.zeros_like(valid)
    selected=positions[:count];keep[selected[:,0],selected[:,1]]=True
    labels.masked_fill_(~keep,-100)
    return result,{'active':active,'supervised':count}


def budget_stop(state,total_elapsed_seconds):
    if state['supervised']>=SUPERVISED_LIMIT:return 'SUPERVISED_BUDGET'
    if state['active']>=ACTIVE_LIMIT:return 'ACTIVE_BUDGET'
    if total_elapsed_seconds>=TOTAL_SECONDS:return 'TIME_BUDGET'
    return None


def write_recovery(path,model,optimizer,state,identity):
    """Only the designated rolling recovery is replaced; milestones are immutable."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
    payload={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
             'progress':dict(state),'identity':identity,'torch_rng':torch.get_rng_state(),
             'python_rng':random.getstate(),
             'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    with temporary.open('xb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def read_recovery(path,model,optimizer,identity):
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload['identity']!=identity:raise RuntimeError('recovery identity mismatch')
    model.load_state_dict(payload['model_state_dict'],strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    torch.set_rng_state(payload['torch_rng']);random.setstate(payload['python_rng'])
    if payload['cuda_rng'] is not None:torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return dict(payload['progress'])


def make_data_plan(source,out):
    from train_fdt_v4_1_retention import RowPool,TokenBudgetSourcePlanner
    from astra_evidence_io import save
    from astra_pilot_holdout import make_holdout
    from fdt_rlm.tokenization import load_tokenizer
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest_path=source/'runs'/PARENT_NAME/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    parent=manifest_path.parent/'latest.pt'
    if digest(parent)!=PARENT_SHA:raise RuntimeError('parent changed')
    planner=TokenBudgetSourcePlanner(manifest['source_token_fractions'])
    planner.restore(manifest['source_states']['__token_planner__'])
    pools={};shards={};validation={}
    tokenizer_dir=source/'tokenizers/fdt_v3_bpe_24k'
    tokenizer=load_tokenizer(str(tokenizer_dir))
    holdout=make_holdout()
    spans={r['id']:tokenizer.encode(r['evidence_span']) for r in holdout if r.get('evidence_span')}
    fullprompts={r['id']:tokenizer.encode(r['prompt']) for r in holdout}
    leakage=[]
    for name in planner.fractions:
        folder=Path(manifest['dataset_dirs'][name]);s=manifest['source_states'][name]
        pool=RowPool(folder,'train',s['seed']);pool.restore(s)
        if len(pool.paths)!=1:raise RuntimeError('pilot plan requires the verified single-shard views')
        pools[name]=pool
        shards[name]={'path':str(pool.paths[0]),'sha256':digest(pool.paths[0]),'initial_state':s}
        # Byte comparison is on canonical int32 token IDs; pad blocks separate rows.
        corpus=pool.payload['input_ids'].to(torch.int32).numpy().tobytes()
        for kind,checks in [('whole_prompt',fullprompts),('answer_bearing_span',spans)]:
            for key,ids in checks.items():
                needle=torch.tensor(ids,dtype=torch.int32).numpy().tobytes()
                offset=corpus.find(needle)
                while offset>=0 and offset%4:offset=corpus.find(needle,offset+1)
                if offset>=0:leakage.append({'source':name,'case':key,'kind':kind})
    if leakage:raise RuntimeError(f'holdout token overlap found: {leakage}')
    rows=[];active=supervised=0;seen=set()
    while supervised<SUPERVISED_LIMIT:
        name=planner.choose_source();pool=pools[name]
        if pool.rows_consumed>=pool.total_rows:raise RuntimeError('source would replay within first epoch')
        row_id=int(pool.order[pool.cursor]);row=pool.next_batch(1)
        selected,stats=restrict_row(row,SUPERVISED_LIMIT-supervised,ACTIVE_LIMIT-active)
        if selected is None:break
        if (name,row_id) in seen:raise RuntimeError('duplicate selected row')
        seen.add((name,row_id));planner.record(name,stats['active'])
        rows.append({'source':name,'row_id':row_id,**stats,
                     'label_mask_sha256':hashlib.sha256(selected['labels'].numpy().tobytes()).hexdigest()})
        active+=stats['active'];supervised+=stats['supervised']
    if supervised!=SUPERVISED_LIMIT:raise RuntimeError('active budget prevents planned supervised endpoint')
    summary={n:{'rows':sum(r['source']==n for r in rows),
        'active':sum(r['active'] for r in rows if r['source']==n),
        'supervised':sum(r['supervised'] for r in rows if r['source']==n),
        'final_pool_state':pools[n].state()} for n in pools}
    # These views have no validation split. Reserve 16 later, not-yet-consumed
    # rows per source, excluding content duplicates of the parent+pilot prefix.
    # This is a local continuation holdout, not a claim of novel pretraining data.
    for name,pool in pools.items():
        def row_hash(index):
            ids=pool.payload['input_ids'][index]
            mask=pool.payload['attention_mask'][index].bool()
            return hashlib.sha256(ids[mask].to(torch.int32).numpy().tobytes()).hexdigest()
        seen_content={row_hash(int(i)) for i in pool.order[:pool.cursor]}
        chosen=[]
        for i in pool.order[pool.cursor:]:
            index=int(i);key=row_hash(index)
            if key in seen_content:continue
            chosen.append({'row_id':index,'active_input_sha256':key});seen_content.add(key)
            if len(chosen)==16:break
        if len(chosen)!=16:raise RuntimeError(f'no 16 disjoint source holdout rows: {name}')
        validation[name]={'kind':'unconsumed_same_view_content_deduplicated',
                          'global_training_history':'UNKNOWN','rows':chosen}
    fixed=source/'prepared_data/fdt_v4_1_fixed52_dual_validation_20260903_r1/shards/validation'
    fixed_paths=sorted(fixed.glob('*.pt'))
    if len(fixed_paths)!=1:raise RuntimeError('fixed52 payload missing')
    plan={'parent':str(parent),'parent_sha256':PARENT_SHA,'manifest_sha256':digest(manifest_path),
          'shards':shards,'validation':validation,'fixed52':{'path':str(fixed_paths[0]),'sha256':digest(fixed_paths[0])},
          'tokenizer_dir':str(tokenizer_dir),'tokenizer_sha256':digest(tokenizer_dir/'tokenizer.json'),
          'seed':20260917,'active':active,'supervised':supervised,'rows':rows,'sources':summary,
          'leakage':{'checked':'whole prompt and answer-bearing token span against all five current train shards',
                     'matches':leakage,'global_lineage_overlap':'UNKNOWN'},
          'config':{'alpha':.2,'legacy_scale':1.,'lr':1e-7,'weight_decay':.01,'grad_clip':.7,
                    'batch_rows':8,'forward_batch_rows':4,'eos_weight':2.,'freeze_anchor':True,
                    'exact_objective':0.,'recovery_objective':0.,'unlikelihood_objective':0.}}
    save(out/'data_plan.json',plan);save(out/'holdout.json',holdout)
    print(json.dumps({'supervised':supervised,'active':active,'rows':len(rows),'sources':summary}),flush=True)
