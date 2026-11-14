"""ONE bounded actual-Adam direction repair; original V80 TRAIN anchor kept."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, ctypes, gc, json, random, shutil, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'src')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64, FuzzyRoundTripV64
from research.fp32_ordered_readout_v86 import FP32OrderedFuzzyReadout
from research.readout_precision_policy_v81 import POLICY
from research.immutable_reference_anchor_v89 import verify
from research import linked_lastlinear_runtime_v79 as linked
from feasible_critical_engine_v90 import FeasibleCriticalEngine
from run_critical_repair_20261113_v88 import assess, size, save, final_reference
import learning_io_v2 as io
import text_generation_support_v1 as support

SCHEMA='fdt_feasible_actual_Adam_original_anchor_repair_v91'
SOURCE_SHA='ebe939a6c14b205bd92a6cad772b33cf2ee462473bb77daadcc6ba48926c84d7'
ALGORITHM='bd3edf2d661231a571d6c0af8c45a2f725545793128ad114b263a9e2c06765dd'
STORAGE_SHA='734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7'
ANCHOR_SHA='73d2eee286442ecfb00c480a28f54e4f1522f5a44d71b874a2e36796562dab54'
OUTPUT=144*1024**2
START=108*1024**3+OUTPUT
CAPS={k:v*1024**2 for k,v in dict(initial=13,model=32,emergency=32,temporary=32,assessment=22,TRAIN_reference=1,other=12).items()}
COST=dict(head_encode=480, head_readout=480, reference_linear=194, final_cached_readout=194)


def validate(path, out):
    p=io.load_json(path)
    fixed=dict(schema=SCHEMA, attempts=96, group_size=5, reused_static_teacher_rows=1024, fresh_teacher_calls=0,
        source_SHA=SOURCE_SHA, source_algorithm_SHA=ALGORITHM, storage_parent_SHA=STORAGE_SHA,
        anchor_SHA=ANCHOR_SHA, max_calls=1400, expected_max_calls=1348, max_output_bytes=OUTPUT,
        start_free_bytes=START, wall_seconds=600, no_accepted_window=32, precision_policy=POLICY,
        reference_patch_work_cap=8192*194, reference_calls_per_group_cap=2)
    if (any(p.get(k)!=v for k,v in fixed.items()) or p['component_caps']!=CAPS or p['cost']!=COST
            or sum(CAPS.values())!=OUTPUT or sum(COST.values())!=1348 or Path(p['output']).resolve()!=out):
        raise ValueError('fixed NEW purpose/cost/whole-output/finite96 schema')
    if p['authorized'] and (p['deadline']-p['start']!=600 or p['cutoff']!=p['deadline']-120):
        raise ValueError('new fixed clock; no old deadline or remaining-budget reuse')
    for f, expected in p['files'].items():
        if support.sha(f)!=expected: raise ValueError('input pin '+f)
    if support.core_sha(ROOT)!=support.CORE_SHA: raise ValueError('original core changed')
    anchor=io.load_json(p['anchor'])
    if anchor['anchor_SHA']!=ANCHOR_SHA: raise ValueError('ORIGINAL reference anchor')
    verify(anchor, anchor['representation'], anchor['bank_reference'])
    rows=io.load_json(Path(p['data'])/'train_features.json')
    orders=io.load_json(Path(p['data'])/'orders.json')[:96]
    items=io.load_json(p['eval_inputs'])
    if len(rows)!=1024 or len(orders)!=96 or any(len(g)!=5 for g in orders) or len(items)!=194:
        raise ValueError('fixed TRAIN and previously observed assessment inputs')
    source_result=io.load_json(p['source_result'])
    baseline=io.load_json(p['baseline_records'])
    if (source_result['model']['SHA']!=SOURCE_SHA or len(baseline)!=194
            or source_result['model']['reconstructed_fullstate_SHA']!=ALGORITHM):
        raise ValueError('saved baseline is coupled to exact warm source, not a fresh NN baseline')
    for record,item in zip(baseline,items):
        if record['id']!=item['row']['id'] or record['panel']!=item['panel'] or record['index']!=item['index']:
            raise ValueError('matched saved baseline row ordering')
        if p['files'].get(record['trace'])!=record['SHA']: raise ValueError('source final trace pin')
    return p, rows, orders


class Budget:
    def __init__(self,p,preflight):
        self.p=p; self.preflight=preflight; self.output=Path(p['output']); self.began=time.monotonic()
        self.calls=dict(encode=0,readout=0,reference_linear=0,denoise=0); self.block_work=0
        self.data_sha=support.sha(Path(p['data'])/'manifest.json'); self.parent_SHA=SOURCE_SHA
        self.parent_reference=dict(source=p['source'],source_SHA=SOURCE_SHA,storage_parent=p['storage_parent'],storage_parent_SHA=STORAGE_SHA,
            old2Adam_scaler_ALLRNG='V88 populated16steps/ALLRNG preserved external; fresh2Adam under new actual-delta correction')
        self.anchor_indices=io.load_json(p['anchor_selection'])['indices']
        self.critical_targets={r['id']:r['critical_spans'] for r in io.load_json(Path(p['data'])/'train_targets.json')}
        self.frozen_symbol_profile=io.load_json(p['symbol_profile'])
    def check(self,write_bytes=0,emergency=False):
        if not emergency:
            if self.preflight:
                if time.monotonic()-self.began>75: raise TimeoutError('CPU75sec; owner90+30reserve')
            elif time.time()>=self.p['cutoff'] or (self.output/'STOP').exists():
                raise InterruptedError('new fixed cutoff/cooperative STOP')
        if shutil.disk_usage(ROOT).free<100*1024**3+write_bytes: raise OSError('100GiB plus pending write protected')
        if size(self.output)+write_bytes>(64*1024**2 if self.preflight else OUTPUT): raise OSError('whole output incl STOP/temp')
        if not self.preflight:
            for name in ('initial','model','emergency','assessment','TRAIN_reference'):
                if size(self.output/name)>CAPS[name]: raise OSError('component cap '+name)
    def charge(self,kind):
        self.check()
        if kind not in self.calls or kind=='denoise' or sum(self.calls.values())>=(8 if self.preflight else 1400):
            raise ValueError('fixed new call cap/type')
        self.calls[kind]+=1


def initialize(p,rows,orders,budget,plan_sha,device):
    source=linked.load_verified(p['source'],SOURCE_SHA,ROOT,support.state_sha)
    if (support.state_sha(source)!=ALGORITHM or source['updates']!=16 or source['attempts']!=108
            or len(source['optimizer']['state'])!=2 or any(int(v['step'])!=16 for v in source['optimizer']['state'].values())):
        raise ValueError('exact V88 fullstate including populated old Adam')
    storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True)
    if support.sha(p['storage_parent'])!=STORAGE_SHA: raise ValueError('standalone base SHA')
    for f, expected in io.load_json(p['representation_operators']).items():
        if support.sha(f)!=expected: raise ValueError('original representation operator '+f)
    anchor=io.load_json(p['anchor'])
    if anchor['bank_reference']!=source['bank_ref']: raise ValueError('source still uses ORIGINAL V80 bank')
    static=dict(anchor=anchor,bank_ref=source['bank_ref'],teacher_refs=source['cache_refs'],teacher_sha=source['teacher_sha'])
    torch.manual_seed(91113); random.seed(91113); np.random.seed(91113)
    if device=='cuda': torch.cuda.manual_seed_all(91113)
    encoder=FuzzyCodecV64(Config(**source['core']),input_capacity=source['input_capacity'])
    encoder.enable_position_adapter(); encoder.enable_long_capacity(); encoder.enable_local_banks()
    model=FuzzyRoundTripV64(encoder,FP32OrderedFuzzyReadout()); model.load_state_dict(source['model'],strict=True)
    if support.state_sha(model.state_dict())!=support.state_sha(source['model']): raise ValueError('all warm tensors exact')
    engine=FeasibleCriticalEngine(model,rows,orders,budget,plan_sha,device,
        static_reference=static,representation=anchor['representation'])
    if (engine.embedding_sha!=source['embedding_sha'] or engine.frozen_except_last2_SHA!=source['frozen_except_last2_SHA']
            or engine.optimizer.state or len(engine.names)!=2
            or sum(dict(model.named_parameters())[n].numel() for n in engine.names)!=2117664
            or not torch.equal(engine.symbol_weights.cpu(),source['symbol_weights'])):
        raise ValueError('fresh2Adam scope/full frozen identity/profile')
    engine.storage_parent_model=storage['model']
    engine.storage_parent_reference=dict(path=p['storage_parent'],SHA=STORAGE_SHA,model_key='model',model_SHA=support.state_sha(storage['model']))
    return engine,dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,
        all_source_Adam_scaler_ALLRNG_preserved_external=True,source_model_SHA=support.state_sha(source['model']),
        fresh2Adam=True,only2_optimizer_parameters=2117664,seven_stored_overrides=True,
        all_codec_and_fiveprefix_frozen=True,original_anchor_SHA=ANCHOR_SHA,original_weak_floors=152,
        reused_static_teacher_rows=1024,fresh_teacher_encoder_forwards=0,
        inference_operator_pins_verified=True,warm_teacher_identities_distinct=engine.warm_readout_SHA!=engine.teacher_sha)


def run(path,out,preflight=False):
    p,rows,orders=validate(path,out)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    budget=Budget(p,preflight)
    if preflight:
        budget.output=path.parent/'cpu_preflight_owned/runtime'; budget.output.mkdir(exist_ok=False)
    engine,restore_proof=initialize(p,rows,orders,budget,support.sha(path),'cpu' if preflight else 'cuda')
    if preflight:
        initial=save(engine,'initial')
        engine.restore(linked.load_verified(initial['path'],initial['SHA'],ROOT,support.state_sha))
        engine.validate_warm_reference()
        commit=engine.extract_live(rows[0]); budget.charge('readout'); z=engine.model.verbalize(commit); t=engine.packet(0)
        from research.competing_margin_objective_v78 import decoder_losses
        loss,_=decoder_losses(z,t['labels'],engine.symbol_weights,t['critical_mask'],t['teacher_gold_probability'],t['teacher_correct'],t['teacher_margin'])
        loss.backward(); named=dict(engine.model.named_parameters())
        if any(named[n].grad is None for n in engine.names) or any(v.grad is not None for n,v in named.items() if n not in engine.names):
            raise ValueError('actual fullsize only2gradient')
        engine.optimizer.zero_grad(set_to_none=True); engine.verify_frozen()
        verified=save(engine,'verified')
        engine.restore(linked.load_verified(verified['path'],verified['SHA'],ROOT,support.state_sha))
        if support.state_sha(engine.export())!=verified['reconstructed_fullstate_SHA'] or torch.cuda.is_initialized():
            raise ValueError('actual fullsize original-anchor exact restore')
        report=dict(restore_proof,fullsize_engine_restore=True,CUDA_initialized=False,actual_encoder_forwards=1,
            actual_readout_forwards=1,actual_reference_projections=1,actual_only2_gradient=True,optimizer_steps=0,
            actual_fullsize_initial_and_verified_restore=True,enabled_CPU_scaler=engine.scaler.is_enabled(),
            fullFP32_readout_dtype=str(z.dtype),all_original_TRAIN_reference_pass=True,original_floor_not_recalibrated=True,
            initial_bytes=initial['bytes'],verified_bytes=verified['bytes'])
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report)); print('CPU_PREFLIGHT_PASS'); return 0
    if not p['authorized'] or (out/'worker_started.json').exists(): raise ValueError('ONE new execution')
    started=time.time(); io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=started,plan_SHA=support.sha(path)))
    latest=None; sleepguard=False; terminal='FEASIBLE_REPAIR_V91_COMPLETE'
    def progress(phase):
        engine.phase=phase
        io.atomic_json(out/'progress.json',dict(phase=phase,attempt=engine.attempts,updates=engine.updates,rejections=engine.rejections,
            static_teacher_rows=1024,new_teacher_rows=0,calls=budget.calls,native_pid=os.getpid(),parent_pid=os.getppid(),
            time=time.time(),elapsed=time.time()-started,latest_durable=latest))
    try:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); sleepguard=True
        latest=save(engine,'initial'); progress('verify_warm_against_ORIGINAL_reference')
        engine.validate_warm_reference()
        io.atomic_json(out/'baseline_reference.json',dict(source_SHA=SOURCE_SHA,source_algorithm_SHA=ALGORITHM,
            saved_records=p['baseline_records'],saved_records_SHA=p['files'][p['baseline_records']],
            rows=194,new_baseline_model_calls=0,matched_FP32=True,already_observed=True))
        progress('bounded_actual_Adam_direction_repair')
        while engine.attempts<len(orders):
            if time.time()>=p['cutoff']-90: terminal='FEASIBLE_REPAIR_V91_EARLY_STOP_MODEL_RESERVE'; break
            engine.step(); progress('bounded_actual_Adam_direction_repair')
            if len(engine.history)>=32 and not any(h['update_receipt']['accepted'] for h in engine.history[-32:]):
                terminal='FEASIBLE_REPAIR_V91_EARLY_STOP_32_NO_ACCEPTED_UPDATES'; break
        engine.verify_frozen(); progress('terminal_ORIGINAL_TRAIN_reference')
        (out/'TRAIN_reference').mkdir(exist_ok=False)
        final_reference(engine)
        progress('terminal_same_cached_Commit_assessment')
        assess(engine,io.load_json(p['eval_inputs']),progress)
        engine.verify_frozen(); latest=save(engine,'model'); progress('terminal_complete')
        io.atomic_json(out/'result.json',dict(status=terminal,attempts=engine.attempts,updates=engine.updates,
            rejections=engine.rejections,AMP_skips=engine.skips,seconds=time.time()-started,calls=budget.calls,
            reference_patch_work=engine.reference_patch_work,bank_patch_count=engine.bank.patch_count,
            weak_critical_positions=engine.bank.weak_count,bank_reference=engine.bank_ref,original_anchor_SHA=ANCHOR_SHA,
            final_reference=engine.reference_final,exposure=engine.exposure,evaluations=engine.evaluations,model=latest,
            restore_proof=restore_proof,static_teacher_rows_reused=1024,new_teacher_rows=0,new_baseline_model_calls=0,
            Exact_entries=0,copy_bytes=0,denoise=0,whole_language_cycle_complete=False,automatic_promotion=False,
            output_bytes=size(out),fresh_source_encoding_only_TRAIN=True,long_evaluation_cached_Commit_only=True,
            evidence_schema='V88 decision trace; NOT full258 raw; blocker IDs from same training projections',old_budgets_closed=True))
        return 0
    except BaseException as error:
        incident=dict(error=repr(error),traceback=traceback.format_exc(),attempts=engine.attempts,calls=budget.calls)
        try: incident['emergency']=save(engine,'emergency',True)
        except BaseException as second: incident['emergency_error']=repr(second)
        io.atomic_json(out/'incident.json',incident); return 2
    finally:
        if sleepguard: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--plan',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--preflight',action='store_true'); args=a.parse_args()
    raise SystemExit(run(args.plan.resolve(),args.output.resolve(),args.preflight))
