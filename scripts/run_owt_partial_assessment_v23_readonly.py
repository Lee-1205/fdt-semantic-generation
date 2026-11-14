"""ONE new read-only assessment of the saved time-guard partial; no SGD/resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism,numpy_state
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE
from run_owt_fuzzy_reconstruction_20261028_v12 import evaluate,generate
from run_owt_fuzzy_reconstruction_20261028_v23 import reconstruct_training_sentences,TRAIN_RECONSTRUCTION_INDICES
from text_language_runtime_support_v6 import RecoveryBudget
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_owt_partial_assessment_v23_readonly'
LIMITS={'attempts':0,'batch':8,'wall':120,'reserve':30,'max_calls':2400,
        'max_output_bytes':256*1024**2,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
COST={'dev_teacher_controls':128,'dev_state_reconstruction_max':1552,'fixed_train_state_reconstruction_max':520,'total_max':2200}
SOURCE_SHA='c45b5efb9b57d3cba1bda234bda025295b6e275fe99dab378181d971c9360a69'
SOURCE_ALGORITHM='0635fd824fab9908114c169eef2ac4ff15c78fc41ba09e3ba626c2c7212670c9'
SOURCE_RUN=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v23_final'
ARTIFACT=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final'
OUTPUT=ROOT/'runs/owt_partial_assessment_20261028_v23_readonly'

def prepare():
    if OUTPUT.exists():raise FileExistsError('new read-only output only')
    source=SOURCE_RUN/'emergency/checkpoints/state_7440.pt'
    review=io.load_json(SOURCE_RUN/'root_partial_state_review.json')
    if review['status']!='CLOSED_PARTIAL_TIME_RESERVE_ROOT_VERIFIED' or review['global_updates']!=47061:raise ValueError('actual source audit')
    names=['scripts/run_owt_partial_assessment_v23_readonly.py','scripts/supervise_owt_partial_assessment_v23_readonly.py',
        'tests/test_owt_partial_assessment_v23_readonly.py','research/owt_prefix_reconstruction_v19.py',
        'research/owt_fuzzy_reconstruction_v12.py','research/text_context_decoder_v2.py','research/text_diffusion_v1.py',
        'scripts/run_owt_fuzzy_reconstruction_20261028_v12.py','scripts/run_owt_fuzzy_reconstruction_20261028_v23.py',
        'scripts/owt_fuzzy_engine_v23.py','scripts/owt_fuzzy_engine_v17_base.py','scripts/owt_fuzzy_engine_v13.py',
        'scripts/owt_fuzzy_engine_v12.py','scripts/prepare_owt_expansion_v23_r2.py','scripts/learning_io_v2.py',
        'scripts/text_generation_support_v1.py','scripts/text_language_runtime_support_v6.py','scripts/supervise_querycount_partial9_20261027_v3.py']
    files=[ROOT/n for n in names]+list((ROOT/'src/fdt_local').glob('*.py'))
    files += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    files += [source,SOURCE_RUN/'plan.json',SOURCE_RUN/'incident.json',SOURCE_RUN/'root_partial_state_review.json']
    files += [ROOT/'runs/owt_fuzzy_reconstruction_20261028_v22'/n for n in ('final_metrics.json','final_generation.json','final_train_reconstruction.json')]
    if support.sha(source)!=SOURCE_SHA:raise ValueError('partial SHA')
    p={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(OUTPUT),'source':str(source),
       'source_sha':SOURCE_SHA,'source_algorithm':SOURCE_ALGORITHM,'artifact':str(ARTIFACT),'source_global_updates':47061,
       'limits':LIMITS,'cost_contract':COST,'optimizer_updates':0,'checkpoint_writes':0,'reasoning_diffusion_training':False,
       'E2_final_accessed':False,'allocation_start':None,'deadline':None,'cutoff':None,
       'generation_indices':[4*i for i in range(16)],'train_reconstruction_indices':TRAIN_RECONSTRUCTION_INDICES,
       'files':{str(x.resolve()):support.sha(x) for x in files},
       'purpose':'Read missing final same-input reconstruction once after partial training was closed by time reserve; no reuse of old learning time/calls/752attempts, no new SGD.'}
    OUTPUT.mkdir();(OUTPUT/'code_snapshot').mkdir();io.atomic_json(OUTPUT/'plan.json',p)
    for x in files:
        if x.suffix=='.py' and x.parent.name in ('scripts','research','tests'):(OUTPUT/'code_snapshot'/x.name).write_bytes(x.read_bytes())
    print(json.dumps({'draft_plan_sha256':support.sha(OUTPUT/'plan.json'),'cost':COST,'limits':LIMITS}))

def validate(path,output,worker=None):
    p=io.load_json(path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p['limits']!=LIMITS or p['cost_contract']!=COST:raise ValueError('actual read-only schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p['source_sha']!=SOURCE_SHA or p['source_algorithm']!=SOURCE_ALGORITHM or p.get('optimizer_updates')!=0 or p.get('checkpoint_writes')!=0 or p.get('E2_final_accessed') is not False:raise ValueError('read-only source contract')
    if p.get('deadline') is not None and (p['deadline']-p['allocation_start']!=120 or p['cutoff']!=p['deadline']-30):raise ValueError('new fixed read-only clock')
    for name,h in p['files'].items():
        if support.sha(name)!=h:raise ValueError('pinned file changed '+name)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('actual worker not pinned')
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    f=io.load_json(Path(p['artifact'])/'features.json');y=io.load_json(Path(p['artifact'])/'targets.json')
    if len(f['dev'])!=256 or p['generation_indices']!=[4*i for i in range(16)] or p['train_reconstruction_indices']!=TRAIN_RECONSTRUCTION_INDICES:raise ValueError('same observed panels')
    return p,f,y

def load_model(p,device):
    s=torch.load(p['source'],map_location='cpu',weights_only=True)
    if support.state_sha(s)!=SOURCE_ALGORITHM or s['schema']!='fdt_owt_fuzzy_AE_v23_fullstate' or (s['attempts'],s['updates'],s['cursor'])!=(7440,7437,59520) or s['gradients']:raise ValueError('actual complete partial fullstate')
    model=PrefixRobustFuzzyAE(Config(**s['core']));model.load_state_dict(s['model'],strict=True)
    digest=support.state_sha(s['model']);model.requires_grad_(False).eval().to(device)
    torch.set_rng_state(s['torch_rng'])
    if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(s['cuda_rng'])
    random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
    if support.state_sha(model.state_dict())!=digest:raise ValueError('loaded model changed')
    return model,digest

def assess(model,dev,targets,dev_probe,train_probe,budget,device):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('all parameters frozen in read-only assessment')
    metrics,tensors=evaluate(model,dev,targets,budget,device)
    raw=generate(model,dev_probe,budget,device)
    train=reconstruct_training_sentences(model,train_probe,budget,device)
    return metrics,tensors,raw,train

def run(path,output):
    p,f,y=validate(path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation')
    configure_determinism();torch.set_num_threads(4);budget=RecoveryBudget(p);guard=False;phase='source_load';start=time.time()
    def progress():io.atomic_json(output/'progress.json',{'phase':phase,'attempts':0,'updates':0,'actual_calls':dict(budget.calls),'native_pid':os.getpid(),'parent_pid':os.getppid(),'time_unix':time.time(),'elapsed':time.time()-start,'latest':None})
    budget.tick=progress;progress()
    try:
        budget.check()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        model,digest=load_model(p,'cuda')
        rng=support.state_sha({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':numpy_state()})
        phase='same_input_readonly_assessment';progress()
        metrics,tensors,raw,train=assess(model,f['dev'],y['dev'],[f['dev'][i] for i in p['generation_indices']],[f['train'][i] for i in p['train_reconstruction_indices']],budget,'cuda')
        torch.save(tensors,output/'final_logits.pt');del tensors
        io.atomic_json(output/'final_metrics.json',metrics);io.atomic_json(output/'final_generation.json',raw);io.atomic_json(output/'final_train_reconstruction.json',train)
        phase='source_preservation';progress()
        after=support.state_sha(model.state_dict())
        rng_after=support.state_sha({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':numpy_state()})
        if after!=digest or rng_after!=rng or support.sha(p['source'])!=SOURCE_SHA:raise RuntimeError('read-only source/model/RNG changed')
        result={'status':'OWT_PARTIAL_V23_READONLY_ASSESSMENT_COMPLETE','source_sha256':SOURCE_SHA,'source_model_sha256':digest,'model_after_sha256':after,'all_used_RNG_unchanged':True,'source_file_unchanged':True,
            'source_global_updates':47061,'actual_calls':dict(budget.calls),'optimizer_updates':0,'checkpoint_writes':0,'seconds':time.time()-start,'final':metrics,
            'generation':{m:{'rows':len(v),'exact':sum(r['reconstruction_exact'] for r in v),'UTF8':sum(r['generated']['utf8_valid'] for r in v),'EOS':sum(r['generated']['eos'] for r in v)} for m,v in raw.items()},
            'train_reconstruction':{'rows':len(train),'exact':sum(r['reconstruction_exact'] for r in train),'UTF8':sum(r['generated']['utf8_valid'] for r in train),'EOS':sum(r['generated']['eos'] for r in train)},'reasoning_or_diffusion_training':False,'E2_final_accessed':False}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'calls':budget.calls});phase='stopped_or_error';progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--preflight',action='store_true');ap.add_argument('--plan',type=Path);ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.prepare:prepare();return 0
    if a.preflight:
        p,f,y=validate(a.plan,a.output,__file__);torch.set_num_threads(2);load_model(p,'cpu')
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())

if __name__=='__main__':raise SystemExit(main())
