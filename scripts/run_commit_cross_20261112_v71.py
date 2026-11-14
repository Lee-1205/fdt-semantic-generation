"""Four read-only CPU final-readout calls, no encoder/denoiser/SGD."""
import argparse,gc,io as memoryio,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout,PatchCommit
from owned_output_size_v52 import owned_output_bytes

def validate(path,output):
    p=io.load_json(path)
    required=dict(schema='fdt_commit_cross_v71',output=str(output.resolve()),wall_seconds=120,max_calls=4,cases=8,device='cpu',attempts=0,epochs=0,max_output_bytes=16*1024**2)
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('new CPU cross contract')
    if p['authorized'] and (p['deadline']-p['start']!=120 or p['cutoff']!=p['deadline']-30):raise ValueError('new clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    return p

def load_heads(p):
    heads={};digests={};rng=torch.get_rng_state().clone()
    for side in ('old','new'):
        parent=torch.load(p[side+'_source'],map_location='cpu',weights_only=True);before=support.state_sha(parent)
        if before!=p[side+'_algorithm_sha'] or len(parent['optimizer']['state'])!=30:raise ValueError('full immutable parent')
        expected={k[8:]:v for k,v in parent['model'].items() if k.startswith('readout.')}
        with torch.random.fork_rng(devices=[]):head=OrderedFuzzyReadout().eval().requires_grad_(False)
        head.load_state_dict(expected,strict=True)
        if support.state_sha(head.state_dict())!=support.state_sha(expected) or support.state_sha(parent)!=before:raise ValueError('exact loaded head / unchanged parent Adam+RNG')
        digests[side]=support.state_sha(head.state_dict());heads[side]=head;del parent,expected;gc.collect()
    assert torch.equal(torch.get_rng_state(),rng) and not torch.cuda.is_initialized()
    return heads,digests

@torch.no_grad()
def run(path,output,preflight=False):
    p=validate(path,output);torch.set_num_threads(4);heads,digests=load_heads(p)
    if preflight:
        for side,head in heads.items():
            b=memoryio.BytesIO();torch.save(head.state_dict(),b);b.seek(0);head.load_state_dict(torch.load(b,map_location='cpu',weights_only=True),strict=True)
            assert support.state_sha(head.state_dict())==digests[side]
        report=dict(fullsize_verbalizer_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,
            parent_Adam_scaler_ALLRNG_unchanged=True,heads=2,head_parameters_each=sum(v.numel() for v in heads['old'].parameters()),
            CUDA_initialized=torch.cuda.is_initialized(),actual_parent_encoder_forward=0,actual_verbalizer_forward=0,optimizer_steps=0,
            no_preflight_access_to_eval_targets=True,runtime_schema='commit_cross_v71')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE CPU diagnosis')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));calls=0;began=time.time()
    def check():
        if (output/'STOP').exists() or time.time()>=p['cutoff']:raise InterruptedError('new CPU clock/STOP')
        if shutil.disk_usage(ROOT).free<101*1024**3 or owned_output_bytes(output)>16*1024**2:raise OSError('resource bound')
        if torch.cuda.is_initialized():raise ValueError('CUDA prohibited')
    try:
        # All feature arguments are old/new neural memory values. The label
        # file is read only AFTER all four final-readout calls have completed.
        features=torch.load(Path(p['data'])/'features.pt',map_location='cpu',weights_only=True)
        if set(features)!={'old_local','old_global','new_local','new_global','terminal'}:raise ValueError('no labels or source features')
        logits={};rng=torch.get_rng_state().clone()
        for commit_side in ('old','new'):
            commit=PatchCommit(features[commit_side+'_local'],features[commit_side+'_global'],features['terminal'])
            for head_side in ('old','new'):
                check()
                if calls>=4:raise ValueError('four CPU call cap')
                calls+=1;z=heads[head_side].decode_patch(commit).float()
                if z.shape!=(8,16,258) or not torch.isfinite(z).all():raise ValueError('finite readout batch')
                logits[commit_side+'Commit_'+head_side+'Readout']=z
                io.atomic_json(output/'progress.json',dict(phase='four_CPU_commit_readout_interchanges',calls=calls,total=4,native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
        assert torch.equal(rng,torch.get_rng_state()) and calls==4
        for side,head in heads.items():assert support.state_sha(head.state_dict())==digests[side]
        reference=torch.load(Path(p['data'])/'reference_logits.pt',map_location='cpu',weights_only=True);cases=io.load_json(Path(p['data'])/'cases.json');records=[]
        for i,c in enumerate(cases):
            role=c['focus_role'];gold=c['gold_byte'];values={}
            for name,z in logits.items():
                scores=z[i,role];without=scores.clone();without[gold]=-torch.inf
                values[name]=dict(argmax=int(scores.argmax()),gold_margin=float(scores[gold]-without.max()),gold_NLL=float(-scores.log_softmax(-1)[gold]))
            diag={}
            for side in ('old','new'):
                z=logits[side+'Commit_'+side+'Readout'][i];ref=reference[side][i]
                diag[side]=dict(focus_matches_native=int(z[role].argmax())==int(ref[role].argmax()),
                    patch_argmax_agreement=float((z.argmax(-1)==ref.argmax(-1)).float().mean()),
                    logits_relative_L2=float((z.double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-12)))
            records.append(dict(name=c['name'],gold_byte=gold,topic=c['topic'],source_context=c['source_context'],values=values,diagonal_parity=diag,
                focus_interpretation_eligible=diag['old']['focus_matches_native'] and diag['new']['focus_matches_native']))
        check()
        with (output/'raw_cross_logits.pt').open('xb') as f:torch.save(logits,f)
        io.atomic_json(output/'cases_result.json',records)
        result=dict(status='COMMIT_CROSS_V71_COMPLETE',calls=calls,case_count=8,CPU_encoder_calls=0,CUDA_calls=0,optimizer_updates=0,checkpoints=0,denoise_calls=0,Exact_entries=0,
            seconds=time.time()-began,all_head_weights_unchanged=True,all_parent_Adam_scaler_RNG_preserved=True,global_Torch_RNG_unchanged=True,
            eligible_cases=sum(r['focus_interpretation_eligible'] for r in records),head_digests=digests,
            limitations='purposively selected observed cases; old/new bound Commit contains upstream role_projection. Off-diagonal states may be out-of-distribution; CPU FP32 versus native BF16/FP32 checked on diagonal; no general causal or semantic-completeness proof.')
        io.atomic_json(output/'result.json',result);return 0
    except BaseException as error:
        io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc(),calls=calls));return 2

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');args=a.parse_args()
    raise SystemExit(run(args.plan.resolve(),args.output.resolve(),args.preflight))
