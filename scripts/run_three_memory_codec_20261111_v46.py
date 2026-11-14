"""V46 same-input codec training on new TRAIN spans and replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, ctypes, json, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.position_sensitive_codec_v46 import ADAPTER_SCALE, PositionSensitiveCodec
from research.text_diffusion_v1 import pack_prompts, pack_answers
from run_three_memory_codec_20261029_v33 import (
    CORPUS_MANIFEST, FOCUSED_MANIFEST, teacher_assessment, reconstruct_inputs,
)
from three_memory_codec_engine_v46 import CodecBudget, PositionCodecEngine
from codec_resume_v46_contract import (
    TRAIN_DATA, FRESH, PRIOR, REPLAY, GLOBAL_ATTEMPTS, GLOBAL_UPDATES, LIMITS,
    SOURCE_ALGORITHM, SOURCE_PLAN, SOURCE_SHA, STATUS, validate as validate_pins,
)
import codec_initial_reference_v42_gzip as checkpoint_storage
import codec_single_snapshot_v42 as snapshot_storage
import text_generation_support_v1 as support
import learning_io_v2 as io


def validate(path, output, worker=None):
    plan = validate_pins(path, output, worker or __file__)
    if support.core_sha(ROOT) != support.CORE_SHA:
        raise ValueError('original reasoning core changed')
    corpus, focus = Path(plan['artifact']), Path(plan['focused_artifact'])
    if support.sha(corpus/'manifest.json') != CORPUS_MANIFEST or support.sha(focus/'manifest.json') != FOCUSED_MANIFEST:
        raise ValueError('original evaluation panel changed')
    features = {'train': io.load_json(TRAIN_DATA/'features.json'),
                'dev': io.load_json(corpus/'features.json')['dev'],
                'focused': io.load_json(focus/'features.json')}
    targets = {'train': io.load_json(TRAIN_DATA/'targets.json'),
               'dev': io.load_json(corpus/'targets.json')['dev'],
               'focused': io.load_json(focus/'targets.json')}
    orders = io.load_json(TRAIN_DATA/'orders.json')
    if sorted(i for batch in orders for i in batch) != list(range(FRESH+REPLAY)):
        raise ValueError('finite row coverage')
    if len(orders) != LIMITS['attempts'] or any(len(batch) != 8 or sum(i < FRESH for i in batch) != 2 or sum(FRESH <= i < FRESH+PRIOR for i in batch) != 2 for batch in orders):
        raise ValueError('exact balanced minibatches')
    for name, count in [('train', FRESH+REPLAY), ('dev', 256), ('focused', 64)]:
        if len(features[name]) != count or len(targets[name]) != count:
            raise ValueError('actual row count for '+name)
        for x, y in zip(features[name], targets[name]):
            if (set(x) != {'id','prompt'} or x['id'] != y['id'] or x['prompt'] != y['target']
                    or y['target_ids'] != list(x['prompt'].encode())+[256]
                    or not 1 <= len(x['prompt'].encode()) <= 128):
                raise ValueError('SAME-input feature/target contract')
    source = io.load_json(Path(plan['source_plan_path']))
    if plan['external_closed_assessment_calls'] != source['external_closed_assessment_calls']:
        raise ValueError('closed call ledger')
    return plan, features, targets, orders


def verify_position_path(engine, feature):
    model = engine.model
    model.eval()
    x = pack_prompts([feature['prompt']], 128, 'cpu')
    y = pack_answers([feature['prompt']], 129, 'cpu')
    args = dict(source_ids=(feature['id'],), source_versions=(engine.data_revision,),
                model_epoch=engine.memory_epoch())
    with torch.no_grad():
        original = model.teacher_logits(model.encode_input(x, **args).commit, y)
    adapter = engine.enable_position_adapter()
    active = model.teacher_logits(model.encode_input(x, **args).commit, y)
    if not torch.equal(original, active):
        raise ValueError('zero-adapter teacher logits must match exactly')
    grad, = torch.autograd.grad(active[0,0,y[0,0]], adapter)
    gains, tags = grad[:,0], grad[:,1:]
    if not torch.isfinite(grad).all() or gains.abs().max() <= 0 or tags.abs().max() <= 0:
        raise ValueError('BOS prediction must reach segment gains and position tags')
    return {'zero_adapter_teacher_parity':True, 'first_byte_segment_grad':float(gains.abs().max()),
            'first_byte_position_grad':float(tags.abs().max()), 'new_Adam_initial_state':0}


def load_import(p,features,targets,orders,budget,plan_sha,device):
    state=torch.load(p['source'],weights_only=True,map_location='cpu')
    if support.state_sha(state)!=SOURCE_ALGORITHM or (state['attempts'],state['updates'],state['cursor'])!=(2048,2047,16384):raise ValueError('actual complete source')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(846);model=PositionSensitiveCodec(Config(**state['core']))
    engine=PositionCodecEngine(model,features['train'],targets['train'],orders,budget,plan_sha,device,
        source_parameter_groups=state['optimizer_parameter_names'],data_revision=support.sha(TRAIN_DATA/'manifest.json'))
    report=engine.import_completed_v45(state, SOURCE_PLAN, SOURCE_SHA)
    if sum(x.numel() for x in model.parameters())!=111732306 or len(engine.optimizer.state)!=91:raise ValueError('unchanged model/ALL91 Adam')
    if str(device)=='cpu':
        restored,sizes=checkpoint_storage.safe_memory_roundtrip(engine.export(),state,p['source'],SOURCE_SHA,SOURCE_ALGORITHM)
        if restored['attempts']!=0 or restored['plan_sha256']!=plan_sha:raise ValueError('initial reference allocation identity')
        report.update(actual_fullsize_initial_reference_bytes=sizes['compressed_bytes'],actual_fullsize_initial_reference_serialized_bytes=sizes['serialized_bytes'],actual_initial_reference_CPU_roundtrip=True,initial_reference_checkpoint_format=checkpoint_storage.SCHEMA)
        report.update(verify_position_path(engine, features['train'][0]))
    return engine,report

def run(path,output):
    p,f,y,orders=validate(path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('new plan not sealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4)
    budget=CodecBudget(p);engine=None;latest=None;phase='importing';began=time.time();guard=False
    slot=snapshot_storage.SingleSnapshot()
    def progress():
        if engine:engine.worker_stage=phase
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,'updates':engine.updates if engine else 0,
            'actual_calls':dict(budget.calls),'allocation_calls':dict(budget.allocation_calls),'native_pid':os.getpid(),
            'parent_pid':os.getppid(),'time_unix':time.time(),'elapsed':time.time()-began,'minimum_C_free_bytes':budget.minimum_free_bytes,'latest':latest,'resource_scans':budget.resource_scans})
    def commit(kind):
        nonlocal latest
        if kind=='emergency':
            if time.time()>=p['deadline']:raise TimeoutError('owner deadline')
        budget.check_write((32 if kind=='import' else 800)*1024**2, emergency=kind=='emergency')
        if kind!='emergency':engine.verify_frozen()
        engine.worker_stage=phase;directory=output/kind;directory.mkdir(exist_ok=True)
        payload=engine.export()
        if kind=='import':
            saved=checkpoint_storage.save_initial_reference(directory,payload,p['source'],SOURCE_SHA,SOURCE_ALGORITHM)
        else:
            saved={**slot.save(directory,payload),'standalone':True,'checkpoint_format':'self_contained'}
        latest={**saved,'relative_directory':kind};io.atomic_json(output/'latest.json',latest)
    def assess(label,features,targets,raw=False,disabled=(),modes=('normal','swap','zero')):
        nonlocal phase
        phase=label+'_teacher';progress()
        scores,tensors=teacher_assessment(engine,features,targets,budget,'cuda',disabled,modes,p['generation_indices'] if label=='held_final' else ())
        budget.check(force=True,upcoming_bytes=128*1024**2);torch.save(tensors,output/(label+'_logits.pt'));del tensors
        io.atomic_json(output/(label+'_metrics.json'),scores)
        for suffix in ('logits.pt','metrics.json'):engine.evaluations[label+'_'+suffix]=support.sha(output/(label+'_'+suffix))
        if raw:
            phase=label+'_same_input_BOS';progress()
            io.atomic_json(output/(label+'_reconstruction.json'),reconstruct_inputs(engine,features,budget,'cuda'))
            engine.evaluations[label+'_reconstruction']=support.sha(output/(label+'_reconstruction.json'))
        return scores
    budget.tick=progress;progress()
    try:
        budget.check(force=True)
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        engine,report=load_import(p,f,y,orders,budget,support.sha(path),'cuda');io.atomic_json(output/'import_review.json',report)
        phase='initial_commit';progress();commit('import')
        engine.enable_position_adapter()
        # The V45 results give the baseline on the same assessment panels.
        phase='position_sensitive_codec_training';progress()
        while engine.attempts<LIMITS['attempts']:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final assessment reserve')
            engine.step()
            progress()
        focused=assess('focused_final',f['focused'],y['focused'],raw=True)
        held=assess('held_final',f['dev'],y['dev'])
        sf=[f['dev'][i] for i in p['generation_indices']];sy=[y['dev'][i] for i in p['generation_indices']]
        phase='held_final_same_input_BOS';progress()
        full_rows = reconstruct_inputs(engine, [f['dev'][i] for i in p['full_BOS_indices']], budget, 'cuda')
        io.atomic_json(output/'held_final_reconstruction.json', full_rows[:len(sf)])
        io.atomic_json(output/'held64_reconstruction.json', full_rows)
        engine.evaluations['held64_reconstruction.json']=support.sha(output/'held64_reconstruction.json')
        engine.evaluations['held_final_reconstruction']=support.sha(output/'held_final_reconstruction.json')
        ablations={m:assess('held16_without_'+m,sf,sy,disabled=(m,),modes=('normal',)) for m in ('persistent','contextual','exact')}
        phase='final_commit';progress();commit('model')
        if support.sha(p['source'])!=SOURCE_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('source/core changed')
        io.atomic_json(output/'result.json',{'status':STATUS,'new_attempts':engine.attempts,'new_updates':engine.updates,
            'AMP_skips':engine.skips,'global_attempts':GLOBAL_ATTEMPTS+engine.attempts,'global_updates':GLOBAL_UPDATES+engine.updates,
            'actual_calls':dict(budget.allocation_calls),'cumulative_calls':dict(budget.calls),'external_closed_assessment_calls':p['external_closed_assessment_calls'],
            'exposure':engine.exposure,'focused_final':focused,'held_final':held,'held16_ablations':ablations,'latest':latest,
            'seconds':time.time()-began,'minimum_C_free_bytes':budget.minimum_free_bytes,'resource_scans':budget.resource_scans,'reasoning_or_diffusion_training':False,'automatic_source_copy':False,
            'E2_final_accessed':False,'position_adapter_gains':(ADAPTER_SCALE*engine.model.memory_codec.position_adapter[:,0]).detach().tanh().cpu().tolist(),
            'interpretation':'One finite TRAIN pass on the same spans with position-sensitive contextual reads and slot tags.'})
        phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if engine:
            engine.optimizer.zero_grad(set_to_none=True)
            try:
                if slot.full_write_started:
                    io.atomic_json(output/'recovery.json',slot.verify_current(engine.export()))
                else:commit('emergency')
            except BaseException as save:io.atomic_json(output/'save_error.json',{'error':repr(save)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'new_calls':budget.allocation_calls,'cumulative_calls':budget.calls})
        progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight:
        p,f,y,orders=validate(a.plan,a.output,__file__);torch.set_num_threads(2)
        engine,report=load_import(p,f,y,orders,CodecBudget(p),support.sha(a.plan),'cpu')
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
