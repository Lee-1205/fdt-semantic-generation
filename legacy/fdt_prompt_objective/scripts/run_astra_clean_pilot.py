"""Fresh repaired-data A/B within the original authorization, no promotion."""
from pathlib import Path
import argparse,ast,gc,json,sys,time,uuid,shutil
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_astra_bounded_pilot as p
from astra_evidence_io import save,completed,ensure_run_identity
from astra_bounded_pilot import digest

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def start_segment(folder,budget):
    segments=folder/'segments';segments.mkdir(exist_ok=True)
    charged=0.
    for path in segments.glob('*.start.json'):
        end=path.with_name(path.name.replace('.start.json','.end.json'))
        if not end.exists():raise RuntimeError('unclosed execution segment; reconcile prior process before restart')
        charged+=read(end)['charged_seconds']
    remaining=budget['remaining_gpu_seconds']-charged
    if remaining<=60:raise p.PilotStop('TIME_BUDGET')
    start=time.time();key=uuid.uuid4().hex;deadline=start+remaining-60
    save(segments/(key+'.start.json'),{'start_unix':start,'deadline_unix':deadline,
        'prior_segment_seconds':charged,'budget_sha256':digest(folder.parent/'budget_accounting.json')})
    return segments/key,start,deadline

def verify_prior_inference(prior):
    reuse=read(prior/'inference_reuse_verification.json')
    identity=read(prior/'execution_v2/run_identity.json')
    if digest(prior/'execution_v2/run_identity.json')!=reuse['old_identity_sha256']:raise RuntimeError('prior identity changed')
    for path,want in identity['code_sha256'].items():
        if path.startswith('src/') and digest(ROOT/path)!=want:raise RuntimeError('model code changed; re-evaluate parent')
    funcs={n.name:p.hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
        for n in ast.parse(Path(p.__file__).read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for name,want in reuse['unchanged_functions'].items():
        if funcs[name]!=want:raise RuntimeError('inference function changed: '+name)
    for r in reuse['files']:
        if digest(r['path'])!=r['sha256']:raise RuntimeError('prior inference artifact changed')
    return reuse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-dir',type=Path,required=True)
    ap.add_argument('--mode',choices=['preflight','train'],required=True);args=ap.parse_args()
    folder=args.plan_dir.resolve();p.require_semantic_clearance(folder)
    plan=read(folder/'data_plan.json');budget=read(folder/'budget_accounting.json');prior=Path(budget['prior_dir'])
    if digest(prior/'DATA_SEMANTICS_STOP.json')!=budget['prior_stop_sha256']:raise RuntimeError('prior stop changed')
    if plan['supervised']!=budget['per_arm_supervised_limit'] or budget['prior_charged_tokens']+2*plan['supervised']>500000:
        raise RuntimeError('token authorization exceeded')
    if shutil.disk_usage(folder).free<35*1024**3:raise RuntimeError('need 35 GiB headroom')
    for path,want in [(plan['parent'],plan['parent_sha256']),(plan['fixed52']['path'],plan['fixed52']['sha256']),
                     (Path(plan['tokenizer_dir'])/'tokenizer.json',plan['tokenizer_sha256'])]+[(s['path'],s['sha256']) for s in plan['shards'].values()]:
        if digest(path)!=want:raise RuntimeError('input hash drift: '+str(path))
    reuse=verify_prior_inference(prior)
    if digest(folder/'holdout.json')!=digest(prior/'holdout.json'):raise RuntimeError('holdout changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    identity={'plan_sha256':digest(folder/'data_plan.json'),'budget_sha256':digest(folder/'budget_accounting.json'),
        'holdout_sha256':digest(folder/'holdout.json'),'torch':torch.__version__,
        'code_sha256':{path.relative_to(ROOT).as_posix():digest(path) for d in ['src','scripts'] for path in sorted((ROOT/d).rglob('*.py'))}}
    out=folder/'execution';ensure_run_identity(out,identity)
    if completed(out/'completed.json'):print('ALREADY COMPLETE');return
    segment,start,deadline=start_segment(out,budget);status='FAILED'
    try:
        if not completed(out/'prior_inference_provenance.json'):save(out/'prior_inference_provenance.json',reuse)
        payloads,source_rows,fixed=p.load_rows(plan)
        for name in ['parent','V20']:
            path=out/(name+'_retention.json')
            if completed(path):continue
            # Reuse only fixed52. Source rows changed and must be evaluated anew.
            old=read(prior/'execution_v2'/(name+'_retention.json'))
            if name=='parent':
                model=p.load_model(plan['parent'])
                sources={n:p.evaluate_nll(model,rows,deadline) for n,rows in source_rows.items()}
                del model;gc.collect();torch.cuda.empty_cache()
            else:sources={}
            save(path,{'fixed52':old['fixed52'],'sources':sources,'fixed52_origin':'verified prior inference',
                'source_origin':'fresh repaired-data retention rows' if name=='parent' else 'not needed for source gate'})
        baselines=[read(out/(name+'_retention.json')) for name in ['parent','V20']]
        if p.retention_fail(baselines[0],[baselines[1]]):raise p.PilotStop('PARENT_RETENTION_GATE')
        if args.mode=='preflight':status='PREFLIGHT_PASS';print(status,flush=True);return
        free,total=torch.cuda.mem_get_info()
        if free<12*1024**3:
            raise p.PilotStop('GPU_HEADROOM: need 12 GiB free for training; preserve other applications')
        holdout=read(folder/'holdout.json');tokenizer=p.load_tokenizer(plan['tokenizer_dir'])
        if not completed(out/'parent_generation_reference.json'):
            save(out/'parent_generation_reference.json',{'path':str(prior/'execution_v2/parent_generation'),
                'records':192,'origin':'verified reuse, not newly generated'})
        for arm,norm in [('A','row_mean'),('B','weighted_token_mean')]:
            result=p.train_arm(arm,norm,plan,out/arm,{**identity,'arm':arm,'normalization':norm},
                deadline,payloads,source_rows,fixed,baselines)
            if result.get('paused'):status='COOPERATIVE_PAUSE';return
            if result['stop']!='PLAN_COMPLETE' or result['supervised']!=plan['supervised'] or not result['clean_boundary']:
                if not completed(out/'safety_stop.json'):save(out/'safety_stop.json',result)
                status='SAFETY_STOP';return
            model=p.load_model(result['final_checkpoint'])
            p.generate_holdout(model,tokenizer,holdout,out/(arm+'_generation'),deadline)
            del model;gc.collect();torch.cuda.empty_cache()
        save(out/'completed.json',{'arms':[read(out/a/'final_status.json') for a in ['A','B']],
            'comparison':'repaired-data A/B only','automatic_promotion':False})
        status='COMPLETE'
    except BaseException as exc:
        save(out/f'failure_{uuid.uuid4().hex}.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        elapsed=max(0,time.time()-start)
        # Charge shutdown allowance too. A crash without this record fails closed.
        save(segment.with_suffix('.end.json'),{'elapsed_seconds':elapsed,'charged_seconds':elapsed+60,
            'status':status,'automatic_promotion':False})

if __name__=='__main__':main()
