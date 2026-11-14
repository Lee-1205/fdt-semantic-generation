"""ONE bounded three-phase input/output-meaning pilot. ROOT owns the OS supervisor."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,random,sys,time,traceback
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.semantic_output_pipeline_v11 import SemanticPipeline,PHASES
from research.semantic_output_data_v11 import build
from research.text_span_data_v4 import finite_pair_orders
from research.text_diffusion_v1 import pack_prompts,decode_bytes
from semantic_output_engine_v11 import SemanticEngine
from text_language_runtime_support_v6 import RecoveryBudget
import learning_io_v2 as io
import text_generation_support_v1 as support

SCHEMA='fdt_semantic_output_pipeline_v11'
LIMITS={'attempts':3072,'batch':8,'wall':900,'reserve':90,'max_calls':14000,
    'max_output_bytes':10*1024**3,'start_free_bytes':42*1024**3,'min_gpu_free_mib':12288}
SEEDS={'input':141,'meaning':142,'language':143}
SOURCES={
 'language':('runs/text_decoder_addressing_20261028_v9/model/checkpoints/state_1636.pt','daaaaf0b780e93844efd045321a27a0a96769c21014d426858640c32810181b2'),
 'span':('runs/text_span_alignment_20261028_v4/realizer/checkpoints/state_1024.pt','5abf9351ddd12509e3622783dc95373f0ac1aeb1fef1a1feb5bb241c44a608df'),
 'typed':('runs/querycount_alignment_20261027_v1/model/payload_000010.pt','5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675')}

def validate(path,output,worker=None):
    p=io.load_json(path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('actual schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('phase_limits')!={x:1024 for x in PHASES} or p.get('epochs_each_phase')!=4 or p.get('output_meaning_target')!='query role16 + answer relation4 + constant1':raise ValueError('meaning/phase contract')
    if p.get('E2_final_accessed') is not False or p.get('evaluation_teacher_allowed') is not False:raise ValueError('holdout/predicted-only eval')
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('core changed')
    for name,(relative,h) in SOURCES.items():
        q=str((ROOT/relative).resolve())
        if p['files'].get(q)!=h:raise ValueError('source fullstate pin '+name)
    for file,h in p['files'].items():
        if support.sha(file)!=h:raise ValueError('pinned file changed '+file)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('actual invoked worker unpinned')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=900 or p['cutoff']!=p['deadline']-90):raise ValueError('fixed clock')
    a=Path(p['artifact']);f=io.load_json(a/'features.json');y=io.load_json(a/'targets.json');orders=io.load_json(a/'orders.json')
    if {k:len(v) for k,v in f.items()}!={'train':2048,'dev_pair':64,'dev_surface':64,'dev_position':64}:raise ValueError('fixed pools')
    if any(set(r)!={'id','prompt'} for rows in f.values() for r in rows):raise ValueError('gold in features')
    if build(f,y)!=y:raise ValueError('independent offline target reconstruction')
    pairs=io.load_json(ROOT/'research/artifacts/text_surface_alignment_20261028_v5_cpu/pairs.json')['train']
    flat={p:finite_pair_orders(pairs,2048,4,seed) for p,seed in SEEDS.items()}
    if orders!={p:[v[i:i+8] for i in range(0,len(v),8)] for p,v in flat.items()}:raise ValueError('finite order changed')
    if p.get('cost_contract')!={'input_training':2048,'meaning_training':3072,'frozen_prediction_cache':5040,'language_training':1024,'initial_meaning':432,'final_generation_max':960,'oracle_diagnostic_max':80,'total_max':12656}:raise ValueError('actual cost')
    return p,f,y,orders

def load_model():
    span=torch.load(ROOT/SOURCES['span'][0],map_location='cpu',weights_only=True)
    core=span['core'];old_span=span['model']['span_generator'];del span
    language=torch.load(ROOT/SOURCES['language'][0],map_location='cpu',weights_only=True)['model']
    typed=torch.load(ROOT/SOURCES['typed'][0],map_location='cpu',weights_only=True)['model']
    model=SemanticPipeline(Config(**core));mapping=model.warmstart(language,old_span,typed)
    return model,mapping

def predictions(model,features,budget,device):
    model.eval();out={};amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    with torch.no_grad(),amp:
        for panel,rows in features.items():
            chunks=[]
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];tokens=pack_prompts([r['prompt'] for r in batch],160,device)
                noise=torch.stack([torch.randn(1,model.c.z_dim,generator=torch.Generator().manual_seed(int(r['id'][:15],16)^151)) for r in batch]).to(device)
                read,settled=model.infer(tokens,noise,budget.charge)
                inp=torch.cat((read['entities'].argmax(-1),read['operator'].argmax(-1)[:,None],read['language'].argmax(-1)[:,None]),1)
                chunks.append({'input_labels':inp.cpu(),'labels':settled['labels'].cpu(),'latent':settled['latent'].float().cpu(),
                    'output_logits':settled['logits'].float().cpu(),'pointer_logits':read['pointer_logits'].float().cpu(),
                    'candidate_mask':read['candidate_mask'].cpu(),'operator_logits':read['operator_logits'].float().cpu(),
                    'language_logits':read['language_logits'].float().cpu()})
            out[panel]={k:torch.cat([v[k] for v in chunks]) for k in chunks[0]}
    return out

def meaning_scores(cache,targets):
    result={}
    for panel,x in cache.items():
        ys=targets[panel];a=torch.tensor([y['input_labels'] for y in ys]);b=torch.tensor([y['output_labels'] for y in ys])
        input_match=(x['input_labels']==a).all(-1);output_match=(x['labels']==b).all(-1)
        result[panel]={'rows':len(ys),'input_head_accuracy':(x['input_labels']==a).float().mean(0).tolist(),
            'output_head_accuracy':(x['labels']==b).float().mean(0).tolist(),'input_all_correct':int(input_match.sum()),
            'output_all_correct':int(output_match.sum()),'output_correct_with_input_correct':int((input_match&output_match).sum()),
            'input_correct_condition_rows':int(input_match.sum())}
    return result

def canonical_for_score(labels):
    s,o,relation,lang=map(int,labels);a=chr(65+s);b=chr(65+o)
    if relation==0:return 'It is unknown.' if lang==0 else '알 수 없습니다.'
    if relation==3:return 'The statements conflict.' if lang==0 else '관계가 상충합니다.'
    return (f'{a} is {"larger" if relation==1 else "smaller"} than {b}.' if lang==0 else f'{a}는 {b}보다 {"큽니다" if relation==1 else "작습니다"}.')

def evaluate_generation(model,features,targets,cache,budget,device,oracle=False):
    model.eval();result={};amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    with torch.no_grad(),amp:
        for panel,rows in features.items():
            raw=[]
            for start in range(0,len(rows),8):
                ys=targets[panel][start:start+8];committed=(torch.tensor([y['output_labels'] for y in ys]) if oracle else cache[panel]['labels'][start:start+8]).to(device)
                generated=model.realizer.generate(committed,budget.charge).cpu()
                # Gold, canonical text, and score rules are first used after generation.
                for j,y in enumerate(ys):
                    decoded=decode_bytes(generated[j].tolist());labels=committed[j].cpu().tolist()
                    raw.append({'id':rows[start+j]['id'],'prompt':rows[start+j]['prompt'],'gold_answer':y['answer'],
                        'output_gold':y['output_labels'],'output_committed':labels,'input_predicted':cache[panel]['input_labels'][start+j].tolist(),
                        'generated':decoded,'exact':decoded['well_formed'] and decoded['text']==y['answer'],
                        'canonical_fidelity_to_committed':decoded['well_formed'] and decoded['text']==canonical_for_score(labels)})
            result[panel]={'rows':len(raw),'exact':sum(r['exact'] for r in raw),'UTF8':sum(r['generated']['utf8_valid'] for r in raw),
                'EOS':sum(r['generated']['eos'] for r in raw),'canonical_fidelity_to_committed':sum(r['canonical_fidelity_to_committed'] for r in raw),
                'oracle_diagnostic_only':oracle,'raw':raw}
    return result

def run(plan_path,output):
    p,f,y,orders=validate(plan_path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4);torch.manual_seed(151);torch.cuda.manual_seed_all(151);random.seed(151);np.random.seed(151)
    budget=RecoveryBudget(p);budget.calls['reader']=0;budget.allocation_calls['reader']=0
    engine=None;latest=None;phase='loading';guard=False;start=time.time()
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,'updates':engine.updates if engine else 0,
            'attempts_by_phase':engine.attempts_by_phase if engine else None,'actual_calls':dict(budget.calls),'allocation_calls':dict(budget.allocation_calls),
            'native_pid':os.getpid(),'parent_pid':os.getppid(),'time_unix':time.time(),'elapsed':time.time()-start,'latest':latest})
    def commit(kind):
        nonlocal latest
        if kind=='emergency':
            if time.time()>=p['deadline']:raise TimeoutError('owner end')
        else:budget.check();engine.verify_frozen()
        directory=output/kind;directory.mkdir(exist_ok=True)
        latest={**support.save_full(directory,engine.export()),'relative_directory':kind};io.atomic_json(output/'latest.json',latest)
    budget.tick=progress;progress()
    try:
        budget.check()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        model,mapping=load_model();engine=SemanticEngine(model,f,y,orders,budget,support.sha(plan_path),'cuda')
        io.atomic_json(output/'specification.json',{'mapping':mapping,'parameters':sum(p.numel() for p in model.parameters()),
            'phase_parameters':{s:sum(p.numel() for n,p in model.named_parameters() if n in set(model.potential_trainable) and
                (n.startswith(('encoder.','input.')) if s=='input' else n.startswith('meaning.') if s=='meaning' else n.startswith('realizer.'))) for s in PHASES},
            'runtime_raw_prompt_to_realizer':False,'diffusion_target':'OUTPUT query-role/answer-relation semantic basis, not bytes','fuzzy_closure_inference':False,'fresh_optimizers':True})
        commit('initial');phase='initial_meaning';progress()
        initial=predictions(model,{k:v for k,v in f.items() if k!='train'},budget,'cuda')
        torch.save(initial,output/'initial_predictions.pt');io.atomic_json(output/'initial_meaning.json',meaning_scores(initial,y));del initial
        for stage in PHASES:
            if stage!='input':engine.set_phase(stage)
            phase=stage;progress()
            if stage=='language':
                phase='frozen_meaning_cache';progress();cache=predictions(model,f,budget,'cuda')
                torch.save(cache,output/'meaning_cache.pt');engine.cache=cache;engine.cache_reference={'path':'meaning_cache.pt','sha256':support.sha(output/'meaning_cache.pt')}
                io.atomic_json(output/'final_meaning.json',meaning_scores(cache,y));commit('language_boundary');phase=stage;progress()
            while engine.attempts_by_phase[stage]<1024:
                if time.time()+90>=p['cutoff']:raise TimeoutError('final evaluation reserve')
                engine.step()
                if engine.attempts_by_phase[stage] in (512,1024):commit(stage)
                progress()
        phase='final_generation';progress()
        devf={k:v for k,v in f.items() if k!='train'}
        raw=evaluate_generation(model,devf,y,engine.cache,budget,'cuda');io.atomic_json(output/'final_generation.json',raw)
        small={'dev_pair':f['dev_pair'][:16]};oracle=evaluate_generation(model,small,{'dev_pair':y['dev_pair'][:16]},engine.cache,budget,'cuda',True)
        io.atomic_json(output/'oracle_realizer_diagnostic.json',oracle)
        engine.evaluations={'final_generation':support.sha(output/'final_generation.json'),'final_meaning':support.sha(output/'final_meaning.json')}
        commit('final')
        for relative,h in SOURCES.values():
            if support.sha(ROOT/relative)!=h:raise RuntimeError('source changed')
        if support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('core changed')
        result={'status':'SEMANTIC_OUTPUT_PIPELINE_V11_COMPLETE','attempts':engine.attempts,'updates':engine.updates,
            'attempts_by_phase':engine.attempts_by_phase,'updates_by_phase':engine.updates_by_phase,'skips':engine.skips,
            'exposure':engine.exposure,'actual_calls':budget.calls,'latest':latest,'seconds':time.time()-start,
            'final_meaning':meaning_scores({k:v for k,v in engine.cache.items() if k!='train'},y),
            'generation_summary':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in raw.items()},
            'E2_final_accessed':False,'automatic_promotion':False,'interpretation':'First narrow raw input-meaning -> conditional output-meaning diffusion -> committed-state neural realization composition. Limited A-H single comparison domain, correlated observed controls, not broad semantic/LM completion.'}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if engine:
            try:commit('emergency')
            except BaseException as save:io.atomic_json(output/'save_error.json',{'error':repr(save)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'actual_calls':budget.calls});progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    if a.preflight:
        p,f,y,orders=validate(a.plan,a.output,__file__);torch.set_num_threads(2);m,mapping=load_model()
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
