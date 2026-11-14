"""One read-only component probe, not a new benchmark or training gate."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,sys,time,traceback
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.text_language_continuation_v7 import LanguageContinuation,ContinuationConfig
from research.text_diffusion_v1 import pack_prompts,PAD
from research.text_context_decoder_v2 import BOS
from text_language_engine_v6 import row_noise
from text_language_runtime_support_v6 import RecoveryBudget
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_decoder_boundary_readonly_v1'
LIMITS={'attempts':0,'batch':4,'micro':4,'wall':120,'reserve':30,'max_calls':160,
    'max_output_bytes':1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
SOURCE_SHA='edcfb82588fb5c7fb2281fd2b2702b340bd91d4073e28e52e93f94ac0eb063dc'
MODES=('normal','zeroVP','swapVP','zeroPrompt','swapPrompt','allZero')

def context_variant(model,enc,z,mode):
    prompt,mask=enc;shift=torch.arange(len(z),device=z.device).roll(1)
    if mode in ('zeroVP','allZero'):z=torch.zeros_like(z)
    elif mode=='swapVP':z=z[shift]
    if mode in ('zeroPrompt','allZero'):prompt=torch.zeros_like(prompt)
    elif mode=='swapPrompt':prompt,mask=prompt[shift],mask[shift]
    elif mode not in MODES:raise ValueError('unknown intervention')
    return model.decoder.context(prompt,mask,z)

def packed_targets(rows,length,device):
    ids=torch.full((len(rows),length),PAD,device=device,dtype=torch.long)
    for i,row in enumerate(rows):ids[i,:len(row)]=torch.tensor(row,device=device)
    return ids

@torch.no_grad()
def trace_batch(model,rows,teacher,own,budget,device):
    tokens=pack_prompts([r['prompt'] for r in rows],model.policy.prompt_bytes,device)
    amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    saved={}
    with amp:
        budget.charge('encode');enc=model.vp.encode(tokens)
        noise=row_noise(rows,model.policy.target_bytes_with_eos,model.core_config.z_dim,17,device)
        z=model.free_proposal(enc,noise,budget.charge)
        budget.charge('readout');direct=model.vp.readout(z)
        for mode in MODES:
            context=context_variant(model,enc,z,mode)
            for name,ids in (('teacher',teacher),('own',own)):
                prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1)
                budget.charge('decoder');saved[mode+'_'+name]=model.decoder(prefix,context).float().cpu()
    saved.update(VP=z.cpu(),prompt=enc[0].cpu(),prompt_mask=enc[1].cpu(),direct_logits=direct.cpu(),
        teacher_ids=teacher.cpu(),own_ids=own.cpu(),ids=[r['id'] for r in rows])
    return saved

def make_plan(output):
    parent=ROOT/'runs/text_language_continuation_20261028_v7';artifact=ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu'
    files=[Path(__file__),ROOT/'scripts/supervise_decoder_boundary_20261028.py',ROOT/'tests/test_decoder_boundary_20261028.py',
        ROOT/'research/text_language_continuation_v7.py',ROOT/'research/text_diffusion_v1.py',ROOT/'research/text_context_decoder_v2.py',
        ROOT/'scripts/text_language_engine_v6.py',ROOT/'scripts/text_language_runtime_support_v6.py',
        ROOT/'scripts/text_generation_support_v1.py',ROOT/'scripts/learning_io_v2.py',ROOT/'scripts/supervise_querycount_partial9_20261027_v3.py',
        parent/'model/checkpoints/state_0818.pt',parent/'plan.json',parent/'final_generation.json',parent/'final_NLL_logits.pt']
    files+=sorted((ROOT/'src/fdt_local').glob('*.py'));files+=sorted(artifact.glob('*.json'))
    return {'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'limits':LIMITS,'epochs':0,
        'cost_contract':{'dev4batches':120,'train_codec_reconstruction_only':20,'total_max':140},
        'source':str(parent/'model/checkpoints/state_0818.pt'),'source_sha':SOURCE_SHA,'artifact':str(artifact),
        'parent':str(parent),'files':{str(p.resolve()):support.sha(p) for p in files},
        'allocation_start':None,'deadline':None,'cutoff':None,'E2_final_accessed':False,'optimizer_steps':0,
        'purpose':'Latest human: strong repetition penalty recalculation, diffusion-vs-decoder boundary, DIANA decoder ancestry. Readonly16olddev and4train clean-codec reconstruction diagnostic ONLY; never score gold conditioning as generation capability.',
        'runtime_gold_semantics':False,'training_reconstruction_probe_is_oracle_diagnostic_only':True,'automatic_restart':False}

def validate(plan_path,output,worker=None):
    p=io.load_json(plan_path)
    if p.get('schema')!=SCHEMA or p.get('limits')!=LIMITS or type(p.get('authorized')) is not bool:raise ValueError('schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('paths')
    if p.get('optimizer_steps')!=0 or p.get('epochs')!=0 or p.get('E2_final_accessed') is not False:raise ValueError('readonly/holdout')
    if p.get('source_sha')!=SOURCE_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('source/core')
    if p['cost_contract']!={'dev4batches':120,'train_codec_reconstruction_only':20,'total_max':140}:raise ValueError('calls')
    for name,h in p['files'].items():
        if support.sha(name)!=h:raise ValueError('pinned file changed '+name)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('worker')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=120 or p['cutoff']!=p['deadline']-30):raise ValueError('clock')
    return p

def load_model(p,device):
    state=torch.load(p['source'],map_location='cpu',weights_only=True)
    m=LanguageContinuation(Config(**state['core']),ContinuationConfig(**state['policy']))
    m.load_state_dict(state['model'],strict=True);m.requires_grad_(False);m.eval();del state
    return m.to(device)

def run(plan_path,output):
    p=validate(plan_path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('one launch')
    configure_determinism();torch.set_num_threads(4);budget=RecoveryBudget(p);budget.calls['readout']=0;budget.allocation_calls['readout']=0
    phase='loading';guard=False;started=time.time()
    def progress():io.atomic_json(output/'progress.json',{'phase':phase,'attempts':0,'updates':0,'native_pid':os.getpid(),'parent_pid':os.getppid(),'actual_calls':budget.calls,'time_unix':time.time()})
    budget.tick=progress;progress()
    try:
        budget.check()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        model=load_model(p,'cuda');before=support.state_sha(model.state_dict())
        features=io.load_json(Path(p['artifact'])/'features.json');targets=io.load_json(Path(p['artifact'])/'targets.json')
        panel=io.load_json(Path(p['parent'])/'final_generation.json')['normal'];index={r['id']:i for i,r in enumerate(features['dev'])}
        traces=[];phase='component_trace';progress()
        for start in range(0,16,4):
            rows=panel[start:start+4]
            teacher=packed_targets([targets['dev'][index[r['id']]]['target_ids'] for r in rows],129,'cuda')
            own=packed_targets([r['generated']['token_ids'] for r in rows],129,'cuda')
            traces.append(trace_batch(model,rows,teacher,own,budget,'cuda'))
        # Train-only clean codec reconstruction isolates the learned interface.
        # Gold conditioning here is an explicitly labelled ORACLE diagnostic, not inference success.
        phase='train_codec_reconstruction_only';progress()
        selected=sorted(range(len(features['train'])),key=lambda i:features['train'][i]['id'])[:4]
        rows=[features['train'][i] for i in selected];ys=[targets['train'][i]['target_ids'] for i in selected]
        ids=packed_targets(ys,129,'cuda');tokens=pack_prompts([r['prompt'] for r in rows],256,'cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            budget.charge('encode');enc=model.vp.encode(tokens);noise=row_noise(rows,129,model.core_config.z_dim,17,'cuda')
            z=model.free_proposal(enc,noise,budget.charge);budget.charge('readout');direct=model.vp.readout(z)
            prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1);train={'ids':[r['id'] for r in rows],'target_ids':ids.cpu(),'free_direct_logits':direct.cpu()}
            for name,latent in (('freeVP',z),('cleanCodec_ORACLE_train_only',model.vp.target(ids))):
                budget.charge('decoder');train[name]=model.decoder(prefix,model.decoder.context(enc[0],enc[1],latent)).float().cpu()
        phase='saving';progress()
        with (output/'traces.pt').open('xb') as f:torch.save({'dev':traces,'train_reconstruction_only':train},f)
        after=support.state_sha(model.state_dict())
        if before!=after or support.sha(p['source'])!=SOURCE_SHA:raise RuntimeError('readonly state changed')
        if sum(budget.calls.values())!=140:raise RuntimeError('actual cost mismatch')
        io.atomic_json(output/'result.json',{'status':'DECODER_BOUNDARY_READONLY_COMPLETE','model_before_sha256':before,'model_after_sha256':after,
            'actual_calls':budget.calls,'actual_call_total':sum(budget.calls.values()),'optimizer_steps':0,'source_sha256':SOURCE_SHA,
            'traces_sha256':support.sha(output/'traces.pt'),'seconds':time.time()-started,'train_oracle_is_not_generation_performance':True,'E2_final_accessed':False})
        phase='complete';progress();return 0
    except BaseException as exc:
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'actual_calls':budget.calls});phase='error';progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--preflight',action='store_true');ap.add_argument('--create',action='store_true');a=ap.parse_args()
    if a.create:
        output=ROOT/'runs/decoder_boundary_readonly_20261028_v1'
        if output.exists():raise FileExistsError('new output')
        p=make_plan(output);output.mkdir();io.atomic_json(output/'plan.json',p);(output/'code_snapshot').mkdir()
        for name in p['files']:
            q=Path(name)
            if q.suffix=='.py' and q.parent.name in ('scripts','research','tests'):(output/'code_snapshot'/q.name).write_bytes(q.read_bytes())
        print('DRAFT_CREATED');return 0
    if a.preflight:
        p=validate(a.plan,a.output,__file__);load_model(p,'cpu')
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
