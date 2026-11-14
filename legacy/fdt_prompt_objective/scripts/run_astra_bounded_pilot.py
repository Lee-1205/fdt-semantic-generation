"""Authorized 250K-supervised-token-per-arm normalization pilot, fail closed.

Uses the existing trainer loss and routing checks. All results are immutable;
only each new arm's explicitly designated recovery.pt is replaced atomically.
The persisted four-hour deadline includes baseline evaluation and downtime.
"""
from pathlib import Path
import argparse
import ast
import dataclasses
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
import uuid
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from astra_bounded_pilot import (digest,restrict_row,write_recovery,read_recovery,
    budget_stop,TOTAL_SECONDS)
from astra_evidence_io import save,completed,ensure_run_identity
from astra_data_contract import require_semantic_clearance
from fdt_rlm.models import build_model
from fdt_rlm.config import ModelConfig
from fdt_rlm.tokenization import load_tokenizer
from train_fdt_v4_1_retention import (validate_main_architecture,base_lm_step_denominator,
    chunked_weighted_lm_loss,chunked_prompt_lm_loss,routing_diagnostics,merge_active_anchor_masks,
    dead_anchor_fraction_from_masks,move_batch)


class PilotStop(RuntimeError):pass


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def guard(deadline):
    # Leave one minute for the current small operation and recovery I/O.
    if time.time()>=deadline-60:raise PilotStop('TIME_BUDGET')


def load_model(path):
    p=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    m=build_model(ModelConfig(**p['model_config']))
    m.load_state_dict(p['model_state_dict'],strict=True)
    return m.to('cuda',dtype=torch.float32).eval()


def configure_training(model,plan):
    validate_main_architecture(model.config)
    if model.config.rope_transition_alpha!=.2:raise RuntimeError('alpha changed')
    for name,p in model.named_parameters():
        if '.anchor.' in name or '.anchor_norm.' in name or name.startswith('exact_pointer.'):
            p.requires_grad_(False)
    model.set_gradient_checkpointing(True)
    c=plan['config'];parameters=[p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters,lr=c['lr'],weight_decay=c['weight_decay'])


def tensor_state_hash(model,optimizer):
    h=hashlib.sha256()
    def walk(obj):
        if isinstance(obj,torch.Tensor):
            v=obj.detach().cpu().contiguous()
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(obj,dict):
            for k in sorted(obj,key=str):h.update(str(k).encode());walk(obj[k])
        elif isinstance(obj,(list,tuple)):
            for v in obj:walk(v)
        else:h.update(repr(obj).encode())
    walk(model.state_dict());walk(optimizer.state_dict())
    return h.hexdigest()


def save_model(path,model,state,identity):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
    with temp.open('xb') as f:
        torch.save({'model_config':dataclasses.asdict(model.config),
            'model_state_dict':model.state_dict(),'progress':dict(state),'identity':identity},f)
        f.flush();os.fsync(f.fileno())
    # Windows rename does not replace an existing destination.
    temp.rename(path)
    save(path.with_suffix('.sha256.json'),{'path':str(path),'sha256':digest(path)})


@torch.inference_mode()
def evaluate_nll(model,rows,deadline,*,all_tokens=False):
    model.eval();total=0.;count=0;row_means=[]
    for row in rows:
        guard(deadline)
        mask=row['attention_mask'].bool();ids=row['input_ids'][mask].long().cuda()[None]
        targets=(row['input_ids'] if all_tokens else row['labels'])[mask].long().cuda()[None,1:]
        valid=targets.ne(-100)&targets.ne(model.config.pad_token_id)
        logits=model(ids,attention_mask=torch.ones_like(ids))['logits'][:,:-1].float()
        losses=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
            targets.masked_fill(~valid,0).reshape(-1),reduction='none').reshape_as(targets)
        n=int(valid.sum());value=float(losses[valid].sum())
        if n==0 or not torch.isfinite(losses[valid]).all():raise PilotStop('INVALID_VALIDATION_LOSS')
        row_means.append(value/n);total+=value;count+=n
    return {'rows':len(row_means),'tokens':count,'row_mean_nll':sum(row_means)/len(row_means),
            'token_weighted_nll':total/count,'eos_weight':1.}


def row_list(payload,indices):
    return [{k:payload[k][int(i)] for k in ('input_ids','labels','attention_mask')} for i in indices]


def retention(model,fixed,source_rows,deadline):
    return {'fixed52':evaluate_nll(model,fixed,deadline,all_tokens=True),
            'sources':{n:evaluate_nll(model,rows,deadline) for n,rows in source_rows.items()}}


def retention_fail(current,baselines):
    return any(current['fixed52'][key]>1.01*b['fixed52'][key]
        for b in baselines for key in ('row_mean_nll','token_weighted_nll'))


def reserve_attempt(folder,entries,state,supervised_limit=250000):
    """Charge retries too, before any update. A crash cannot reset the budget."""
    ledger=folder/'attempts';ledger.mkdir(exist_ok=True)
    old=[read(p) for p in ledger.glob('*.json')]
    active=sum(r['active'] for r in old)+sum(e['active'] for e in entries)
    supervised=sum(r['supervised'] for r in old)+sum(e['supervised'] for e in entries)
    if active>1_200_000:raise PilotStop('RESERVED_ACTIVE_BUDGET')
    if supervised>supervised_limit:raise PilotStop('RESERVED_SUPERVISED_BUDGET')
    save(ledger/(uuid.uuid4().hex+'.json'),{'starting_step':state['step'],
        'starting_cursor':state['cursor'],'active':sum(e['active'] for e in entries),
        'supervised':sum(e['supervised'] for e in entries)})
    return {'active':active,'supervised':supervised}


def cooperative_pause(folder,state):
    """A request is acknowledged in recovery, only at a committed boundary.

    Keep the request as evidence. A later explicit invocation can resume from
    that recovery without deleting the request or replaying the acknowledgement.
    """
    request=folder.parent/'STOP_REQUESTED.json'
    if request.exists():
        key=digest(request)
        if state.get('acknowledged_stop_request')!=key:
            state['acknowledged_stop_request']=key
            raise PilotStop('COOPERATIVE_PAUSE')


def pending_gate(model,opt,state,folder,identity,fixed,source_rows,deadline,baselines):
    milestone=state.get('pending_milestone')
    if milestone is None:return
    checkpoint=folder/f'milestone_{milestone}.pt'
    if not checkpoint.exists():save_model(checkpoint,model,
        {k:v for k,v in state.items() if k!='route_masks'},identity)
    elif not completed(checkpoint.with_suffix('.sha256.json')):
        raise PilotStop('INCOMPLETE_MILESTONE_HASH')
    elif digest(checkpoint)!=read(checkpoint.with_suffix('.sha256.json'))['sha256']:
        raise PilotStop('MILESTONE_HASH_DRIFT')
    path=folder/f'retention_{milestone}.json'
    if not completed(path):save(path,retention(model,fixed,source_rows,deadline))
    result=read(path)
    if retention_fail(result,baselines):
        confirmation=folder/f'retention_{milestone}_confirmation.json'
        if not completed(confirmation):save(confirmation,retention(model,fixed,source_rows,deadline))
        if retention_fail(read(confirmation),baselines):raise PilotStop('FIXED52_RETENTION')
    for n,values in result['sources'].items():
        bad=any(values[k]>1.03*baselines[0]['sources'][n][k]
            for k in ('row_mean_nll','token_weighted_nll'))
        state['source_regressions'][n]=state['source_regressions'][n]+1 if bad else 0
    if max(state['source_regressions'].values())>=2:raise PilotStop('SOURCE_RETENTION')
    state['pending_milestone']=None
    write_recovery(folder/'recovery.pt',model,opt,state,identity)


@torch.inference_mode()
def generate_holdout(model,tokenizer,holdout,folder,deadline):
    model.eval();folder.mkdir(parents=True,exist_ok=True)
    for case in holdout:
        path=folder/(case['id']+'.json')
        if completed(path):continue
        guard(deadline);prompt_ids=tokenizer.encode(case['prompt'])
        ids=torch.tensor([prompt_ids],device='cuda');tokens=[]
        if len(prompt_ids)+64>2048:raise PilotStop('CONTEXT_CONTRACT')
        for _ in range(64):
            guard(deadline)
            logits=model(ids,attention_mask=torch.ones_like(ids))['logits'][:,-1].float()
            if not torch.isfinite(logits).all():raise PilotStop('NONFINITE_GENERATION')
            t=int(logits.argmax(-1));tokens.append(t)
            if t==model.config.eos_token_id:break
            ids=torch.cat((ids,torch.tensor([[t]],device='cuda')),dim=1)
        # No expected value, source label or scoring result enters model decoding.
        save(path,{'id':case['id'],'prompt_ids':prompt_ids,'output_ids':tokens,
            'text':tokenizer.decode(tokens,skip_special_tokens=True),
            'eos':tokens[-1]==model.config.eos_token_id,
            'budget':64,'algorithm':'greedy','exact':False,'backend':'full_recompute'})


def load_rows(plan):
    payloads={n:torch.load(s['path'],map_location='cpu',weights_only=False,mmap=True)
              for n,s in plan['shards'].items()}
    sources={n:row_list(payloads[n],[r['row_id'] for r in v['rows']])
             for n,v in plan['validation'].items()}
    fixed=torch.load(plan['fixed52']['path'],map_location='cpu',weights_only=False,mmap=True)
    if fixed['input_ids'].shape[0]!=52:raise RuntimeError('fixed52 count changed')
    return payloads,sources,row_list(fixed,range(52))


def batch_groups(entries,payloads):
    rows=[]
    for entry in entries:
        p=payloads[entry['source']];i=entry['row_id']
        row={k:p[k][i:i+1] for k in ('input_ids','labels','attention_mask')}
        selected,stats=restrict_row(row,entry['supervised'],entry['active'])
        if stats!={k:entry[k] for k in ('active','supervised')}:raise RuntimeError('row accounting drift')
        if hashlib.sha256(selected['labels'].numpy().tobytes()).hexdigest()!=entry['label_mask_sha256']:
            raise RuntimeError('label mask drift')
        rows.append(selected)
    return [{k:torch.cat([r[k] for r in rows[i:i+4]]) for k in rows[0]}
            for i in range(0,len(rows),4)]


def update(model,opt,entries,payloads,normalization,route_masks,route_steps,prompt_lm_weight=0.0):
    model.train();opt.zero_grad(set_to_none=True);groups=batch_groups(entries,payloads)
    denominator=base_lm_step_denominator(groups,normalization,0,2,2.)
    if not 0.0 <= float(prompt_lm_weight) <= 1.0:
        raise ValueError('invalid prompt_lm_weight')
    accounting=[];scalar=0.;entropy=[];offset=0
    for group in groups:
        batch=move_batch(group,torch.device('cuda'));records=[];prompt_records=[]
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            output=model(batch['input_ids'],attention_mask=batch['attention_mask'],return_logits=False)
            loss=chunked_weighted_lm_loss(model,output['hidden'],batch['labels'],batch['attention_mask'],
                0,2,2.,256,True,accounting=records,
                reduction='row_sum' if normalization=='row_mean' else 'weighted_sum')/denominator
            if prompt_lm_weight:
                loss=loss+float(prompt_lm_weight)*chunked_prompt_lm_loss(
                    model,output['hidden'],batch['input_ids'],batch['labels'],
                    batch['attention_mask'],0,256,True,accounting=prompt_records)/len(entries)
            metrics=routing_diagnostics(output)
        if metrics['dead_anchor_fraction'] is None or metrics['entropy_normalized'] is None:
            raise PilotStop('MISSING_ROUTING_DIAGNOSTICS')
        entropy.append(metrics['entropy_normalized'])
        route_masks=merge_active_anchor_masks(route_masks,metrics['active_anchor_masks'])
        if not torch.isfinite(loss):raise PilotStop('NONFINITE_LOSS')
        scalar+=float(loss.detach());loss.backward()
        if prompt_lm_weight and len(prompt_records)!=len(records):
            raise PilotStop('PROMPT_ACCOUNTING_MISMATCH')
        for index,(entry,record) in enumerate(zip(entries[offset:],records)):
            term=record['normalized_source_loss'] if normalization=='row_mean' else record['weighted_loss_sum']
            record.update(source=entry['source'],row_id=entry['row_id'],
                          effective_scalar_contribution=term/denominator)
            if prompt_records:
                record.update(prompt_records[index])
                record['prompt_effective_scalar_contribution']=(
                    prompt_records[index]['prompt_normalized_loss']*float(prompt_lm_weight)/len(entries))
            accounting.append(record)
        offset+=len(records)
        del output,loss,batch
    route_steps+=1;dead=dead_anchor_fraction_from_masks(route_masks)
    if route_steps>=25:
        if dead is None or dead>.01:raise PilotStop('DEAD_ANCHOR_LIMIT')
        route_masks=None;route_steps=0
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.7,
                                      error_if_nonfinite=True)
    opt.step();opt.zero_grad(set_to_none=True)
    return {'loss':scalar,'gradient_norm_before_clip':float(norm),'rows':accounting,
        'step_denominator':denominator,'routing_entropy':sum(entropy)/len(entropy),
        'window_dead_anchor_fraction':dead},route_masks,route_steps


def train_arm(arm,normalization,plan,folder,identity,deadline,payloads,source_rows,fixed,baselines):
    folder.mkdir(parents=True,exist_ok=True)
    if completed(folder/'final_status.json'):return read(folder/'final_status.json')
    train_config=plan.get('config',{})
    seed=plan['seed'];torch.manual_seed(seed);random.seed(seed);torch.cuda.manual_seed_all(seed)
    model=load_model(plan['parent']);opt=configure_training(model,plan)
    recovery=folder/'recovery.pt';state={'step':0,'cursor':0,'supervised':0,'active':0,
        'route_masks':None,'route_steps':0,'source_regressions':{n:0 for n in source_rows}}
    if recovery.exists():state=read_recovery(recovery,model,opt,identity)
    elif list(folder.glob('step_*.json')) or list((folder/'attempts').glob('*.json')):
        raise RuntimeError('prior attempt without recovery; do not restart')
    if not completed(folder/'initialization.json'):
        save(folder/'initialization.json',{'parent_sha256':plan['parent_sha256'],'seed':seed,
            'normalization':normalization,'fresh_optimizer':True,
            'trainable_parameters':{n:p.numel() for n,p in model.named_parameters() if p.requires_grad},
        'initial_state_hash':tensor_state_hash(model,opt),'lr':train_config.get('lr',1e-7),
        'weight_decay':train_config.get('weight_decay',.01),
        'prompt_lm_weight':float(train_config.get('prompt_lm_weight',0.0))})
    stop=None
    try:
        pending_gate(model,opt,state,folder,identity,fixed,source_rows,deadline,baselines)
        while state['cursor']<len(plan['rows']):
            cooperative_pause(folder,state)
            guard(deadline)
            if budget_stop(state,TOTAL_SECONDS-(deadline-time.time())):break
            entries=plan['rows'][state['cursor']:state['cursor']+8]
            if state['active']+sum(e['active'] for e in entries)>1_200_000:raise PilotStop('ACTIVE_BUDGET')
            if state['supervised']+sum(e['supervised'] for e in entries)>250_000:raise PilotStop('SUPERVISED_BUDGET')
            old_sup=state['supervised']
            reserve_attempt(folder,entries,state,plan['supervised'])
            metrics,masks,steps=update(model,opt,entries,payloads,normalization,
                state['route_masks'],state['route_steps'],
                float(train_config.get('prompt_lm_weight',0.0)))
            state.update(step=state['step']+1,cursor=state['cursor']+len(entries),
                supervised=old_sup+sum(e['supervised'] for e in entries),
                active=state['active']+sum(e['active'] for e in entries),route_masks=masks,route_steps=steps)
            record={'step':state['step'],'supervised':state['supervised'],'active':state['active'],**metrics}
            step_path=folder/f"step_{state['step']:06d}.json"
            # Resume may replay unsaved progress; require exact equality, never overwrite it.
            if completed(step_path):
                if read(step_path)!=record:raise PilotStop('RESUME_REPLAY_MISMATCH')
            else:save(step_path,record)
            recovery_interval=int(plan.get('recovery_interval_supervised',25000))
            if recovery_interval<=0:raise PilotStop('INVALID_RECOVERY_INTERVAL')
            recovery_due=(old_sup//recovery_interval != state['supervised']//recovery_interval
                          or state['step']==1 or
                          time.time()-state.get('last_recovery_unix',time.time())>=3000)
            milestone=next((n for n in plan.get('milestones',[50000,125000,plan['supervised']])
                            if old_sup<n<=state['supervised']),None)
            state['pending_milestone']=milestone
            if recovery_due or milestone:
                state['last_recovery_unix']=time.time()
                write_recovery(recovery,model,opt,state,identity)
            if state['step']==1 and not completed(folder/'actual_restore_verified.json'):
                before=tensor_state_hash(model,opt)
                with torch.no_grad():
                    next(p for p in model.parameters() if p.requires_grad).reshape(-1)[0].add_(1.)
                    next(iter(opt.state.values()))['exp_avg'].reshape(-1)[0].add_(1.)
                restored=read_recovery(recovery,model,opt,identity)
                after=tensor_state_hash(model,opt)
                if before!=after or restored['cursor']!=state['cursor']:raise PilotStop('RESTORE_MISMATCH')
                save(folder/'actual_restore_verified.json',{'before':before,'after':after,
                    'model_and_populated_adam_state':'bitwise roundtrip','next_update_equivalence':'tiny fixture only',
                    'supervised':state['supervised']})
                state=restored
            if state['step']%25==0:print(arm,state['step'],state['supervised'],flush=True)
            if milestone:
                pending_gate(model,opt,state,folder,identity,fixed,source_rows,deadline,baselines)
        if stop is None:stop=budget_stop(state,0) or 'PLAN_COMPLETE'
    except PilotStop as exc:stop=str(exc)
    except BaseException as exc:
        # A failed step is not a committed boundary. Preserve the previous recovery.
        save(folder/f'failure_{uuid.uuid4().hex}.json',{'type':type(exc).__name__,'message':str(exc),
            'last_completed_step_in_memory':state['step'],'recovery_path':str(recovery)})
        raise
    # Save only if the stop happened at a fully committed boundary. All normal
    # safety exits inside update occur before opt.step; recovery remains authoritative.
    clean_stop=stop in {'SUPERVISED_BUDGET','ACTIVE_BUDGET','TIME_BUDGET','PLAN_COMPLETE','FIXED52_RETENTION','SOURCE_RETENTION',
                       'RESERVED_ACTIVE_BUDGET','RESERVED_SUPERVISED_BUDGET','COOPERATIVE_PAUSE'}
    if clean_stop:
        write_recovery(recovery,model,opt,state,identity)
        if stop!='COOPERATIVE_PAUSE' and not (folder/'final.pt').exists():save_model(folder/'final.pt',model,
            {k:v for k,v in state.items() if k!='route_masks'},identity)
    result={'arm':arm,'normalization':normalization,'stop':stop,'optimizer_steps':state['step'],
        'supervised':state['supervised'],'active':state['active'],'clean_boundary':clean_stop,
        'reserved_supervised_including_failed_attempts':sum(read(p)['supervised'] for p in (folder/'attempts').glob('*.json')),
        'cursor':state['cursor'],
        'paused':stop=='COOPERATIVE_PAUSE',
        'final_checkpoint':str(folder/'final.pt') if clean_stop and stop!='COOPERATIVE_PAUSE' else None}
    status_path=folder/(f'pause_{uuid.uuid4().hex}.json' if result['paused'] else 'final_status.json')
    save(status_path,result);print(json.dumps(result),flush=True)
    del opt,model;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-dir',type=Path,required=True)
    ap.add_argument('--mode',choices=['preflight','train'],required=True)
    args=ap.parse_args();require_semantic_clearance(args.plan_dir)
    plan=read(args.plan_dir/'data_plan_resume.json');holdout=read(args.plan_dir/'holdout.json')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if shutil.disk_usage(args.plan_dir).free<35*1024**3:raise RuntimeError('need 35 GiB headroom')
    paths=[plan['parent'],plan['fixed52']['path']]+[v['path'] for v in plan['shards'].values()]
    expected=[plan['parent_sha256'],plan['fixed52']['sha256']]+[v['sha256'] for v in plan['shards'].values()]
    for p,want in zip(paths,expected):
        if digest(p)!=want:raise RuntimeError('input hash drift: '+p)
    if digest(Path(plan['tokenizer_dir'])/'tokenizer.json')!=plan['tokenizer_sha256']:raise RuntimeError('tokenizer drift')
    identity={'plan_sha256':digest(args.plan_dir/'data_plan_resume.json'),
        'holdout_sha256':digest(args.plan_dir/'holdout.json'),
        'code_sha256':{p.relative_to(ROOT).as_posix():digest(p) for d in ['src','scripts'] for p in sorted((ROOT/d).rglob('*.py'))},
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'tf32':False,
        'generation':{'budget':64,'dtype':'float32','greedy':True,'exact':False},
        'limits':{'per_arm_supervised':250000,'per_arm_active':1200000,'total_seconds':TOTAL_SECONDS}}
    ledger=read(args.plan_dir/'budget_resume_accounting.json')
    identity['prior_budget_accounting_sha256']=digest(args.plan_dir/'budget_resume_accounting.json')
    out=args.plan_dir/'execution_v3';ensure_run_identity(out,identity)
    clock=args.plan_dir/'pilot_clock_v3.json'
    if not completed(clock):
        start=time.time()
        save(clock,{'start_unix':start,'deadline_unix':start+ledger['remaining_gpu_seconds'],
            'prior_charged_seconds':ledger['prior_charged_seconds'],
            'previous_clock_preserved':True,'accounting_sha256':identity['prior_budget_accounting_sha256']})
    deadline=read(clock)['deadline_unix'];guard(deadline)
    payloads,source_rows,fixed=load_rows(plan)
    parentpath=Path(plan['parent']);v20=parentpath.parents[1]/'fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2/latest.pt'
    if digest(v20)!='7ee3f88d319928dd2d3f2542290f55ffcd036dcbb32a8ab22437c511e5890179':raise RuntimeError('V20 drift')
    reuse=read(args.plan_dir/'inference_reuse_verification.json')
    old_identity=read(args.plan_dir/'execution_v2/run_identity.json')
    if digest(args.plan_dir/'execution_v2/run_identity.json')!=reuse['old_identity_sha256']:
        raise RuntimeError('prior inference identity drift')
    for path,value in old_identity['code_sha256'].items():
        if path.startswith('src/') and identity['code_sha256'][path]!=value:
            raise RuntimeError('cannot reuse inference after model code change')
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    funcs={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
           for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for name,value in reuse['unchanged_functions'].items():
        if funcs[name]!=value:raise RuntimeError('inference function drift: '+name)
    for record in reuse['files']:
        if digest(record['path'])!=record['sha256']:raise RuntimeError('prior inference result drift')
    if digest(args.plan_dir/'holdout.json')!=reuse['holdout_sha256']:raise RuntimeError('holdout changed')
    if not completed(out/'inference_provenance.json'):save(out/'inference_provenance.json',reuse)
    for name,path in [('parent',parentpath),('V20',v20)]:
        p=out/(name+'_retention.json')
        if not completed(p):
            save(p,read(args.plan_dir/'execution_v2'/(name+'_retention.json')))
    baselines=[read(out/'parent_retention.json'),read(out/'V20_retention.json')]
    # Do not spend on a pilot whose starting parent already violates the frozen gate.
    if retention_fail(baselines[0],[baselines[1]]):
        model=load_model(parentpath);confirm=retention(model,fixed,source_rows,deadline)
        if not completed(out/'parent_gate_confirmation.json'):save(out/'parent_gate_confirmation.json',confirm)
        del model;gc.collect();torch.cuda.empty_cache()
        if retention_fail(confirm,[baselines[1]]):
            save(out/'preflight_stop.json',{'reason':'PARENT_ALREADY_FAILS_V20_RETENTION','optimizer_updates':0})
            print('PREFLIGHT STOP: PARENT_ALREADY_FAILS_V20_RETENTION',flush=True);return
    if args.mode=='preflight':print('PREFLIGHT PASS',flush=True);return
    tokenizer=load_tokenizer(plan['tokenizer_dir'])
    if not completed(out/'parent_generation_reference.json'):
        save(out/'parent_generation_reference.json',{'path':str(args.plan_dir/'execution_v2/parent_generation'),
            'provenance':'inference_provenance.json','records':192})
    for arm,norm in [('A','row_mean'),('B','weighted_token_mean')]:
        result=train_arm(arm,norm,plan,out/arm,{**identity,'arm':arm,'normalization':norm},
                         deadline,payloads,source_rows,fixed,baselines)
        if result.get('paused'):return
        if not result['clean_boundary']:
            save(out/'safety_stop.json',result);return
        if result['final_checkpoint'] and time.time()<deadline-60:
            model=load_model(result['final_checkpoint'])
            generate_holdout(model,tokenizer,holdout,out/(arm+'_generation'),deadline)
            del model;gc.collect();torch.cuda.empty_cache()
    save(out/'completed.json',{'arms':[read(out/a/'final_status.json') for a in ['A','B']],
        'elapsed_seconds':time.time()-read(clock)['start_unix'],
        'automatic_promotion':False})


if __name__=='__main__':main()
