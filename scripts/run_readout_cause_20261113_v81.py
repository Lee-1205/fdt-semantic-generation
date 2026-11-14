"""ONE read-only, cached-Commit diagnostic. No encoder, optimizer or denoising."""
import argparse,gc,io as memoryio,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.precision_readout_v81 import from_commit,MODES
from research.lossless_raw_tensor_v80 import unpack
from saved_commit_reference_v76_v2 import resolve_saved_commit
from owned_output_size_v52 import owned_output_bytes

SCHEMA='fdt_readonly_readout_cause_v81'
EMBED='3ea6672a445e984751fc0489b1e2067eaf6858c704291bc8848f0f81f5c4805c'
LIMITS=dict(wall_seconds=120,max_calls=160,max_output_bytes=32*1024**2,attempts=0,epochs=0)

def validate(path,output):
    p=io.load_json(path)
    if p['schema']!=SCHEMA or Path(p['output']).resolve()!=output.resolve() or any(p.get(k)!=v for k,v in LIMITS.items()):raise ValueError('readonly fixed schema/caps')
    if p['device']!='cuda' or p['precision_modes']!=list(MODES) or p['expected_calls']!=133:raise ValueError('fixed precision diagnostic')
    if p['authorized'] and (p['deadline']-p['start']!=120 or p['cutoff']!=p['deadline']-30):raise ValueError('new diagnostic clock')
    for file,digest in p['files'].items():
        if support.sha(file)!=digest:raise ValueError('pin '+file)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    return p

def load_heads(p):
    base=torch.load(p['storage_parent'],map_location='cpu',weights_only=True);heads={};proof={}
    for version in ('V75','V80'):
        env=torch.load(p['heads'][version]['path'],map_location='cpu',weights_only=True)
        if env['parent_reference']['path']!=p['storage_parent'] or env['parent_reference']['SHA']!=p['files'][p['storage_parent']]:raise ValueError('standalone parent identity')
        model=dict(base['model']);model.update(env['active_weights']);full=dict(env['runtime'],model=model)
        if support.state_sha(full)!=p['heads'][version]['algorithm_SHA'] or support.state_sha(full)!=env['reconstructed_fullstate_SHA']:raise ValueError('fullstate including old Adam/scaler/ALL RNG')
        expected={k.removeprefix('readout.readout.'):v for k,v in model.items() if k.startswith('readout.readout.')}
        with torch.random.fork_rng(devices=[]):head=CommitPatchReadout().eval().requires_grad_(False)
        head.load_state_dict(expected,strict=True);digest=support.state_sha(head.state_dict())
        if digest!=support.state_sha(expected):raise ValueError('exact fullsize readout restore')
        heads[version]=head;proof[version]=dict(head_SHA=digest,parent_runtime_SHA=support.state_sha(env['runtime']),parameters=sum(v.numel() for v in head.parameters()))
        del model,full,env,expected;gc.collect()
    if any(not torch.equal(v,heads['V80'].state_dict()[k]) for k,v in heads['V75'].state_dict().items() if not k.startswith('net.2.')):raise ValueError('matched frozen prefix')
    return heads,proof

def commit_of(p,info):
    packet=unpack(torch.load(info['raw'],map_location='cpu',weights_only=True))
    z,ref=resolve_saved_commit(packet,info['row'],EMBED,p['files'],ROOT,512)
    return z,packet['logits'],packet['targets']

@torch.no_grad()
def run(path,output,preflight=False):
    p=validate(path,output);torch.set_num_threads(4);heads,proof=load_heads(p)
    cases=io.load_json(p['cases']);inputs=io.load_json(p['inputs'])
    if len(cases)!=12 or len(inputs)!=11:raise ValueError('fixed purposive cases')
    if preflight:
        calls=0
        for version,h in heads.items():
            b=memoryio.BytesIO();torch.save(h.state_dict(),b);b.seek(0);h.load_state_dict(torch.load(b,map_location='cpu',weights_only=True),strict=True)
            assert support.state_sha(h.state_dict())==proof[version]['head_SHA']
            z,_,_=commit_of(p,next(iter(inputs.values())))
            c=DifferentiableCodecCommit.from_semantic(z,(('diagnostic','v1','metadata-only'),),'unchanged-cached-Commit')
            out,trace=from_commit(h,c,'full_fp32');calls+=1
            assert out.shape[1:]==(16,258) and torch.isfinite(out).all()
        report=dict(fullsize_engine_restore=True,restored_component='actual 2908192-parameter Commit readout ONLY; full original model+Adam+RNG digest reconstructed, encoder not instantiated/forwarded',
            all_parent_tensors_exact=True,all_old_Adam_exact=True,parent_Adam_scaler_ALLRNG_preserved=True,
            CUDA_initialized=torch.cuda.is_initialized(),actual_CPU_readout_calls=calls,encoder_calls=0,optimizer_steps=0,denoise_calls=0,heads=proof)
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE new diagnostic')
    began=time.time();calls=0;raw={};records=[]
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=began))
    def check():
        if (output/'STOP').exists() or time.time()>=p['cutoff']:raise InterruptedError('fixed diagnostic cutoff/STOP')
        if calls>=p['max_calls']:raise ValueError('call cap')
        if shutil.disk_usage(ROOT).free<101*1024**3 or owned_output_bytes(output)>p['max_output_bytes']:raise OSError('protected disk/output cap')
    def invoke(h,z,mode,key,selected):
        nonlocal calls
        check();calls+=1
        c=DifferentiableCodecCommit.from_semantic(z,(('diagnostic','v1','metadata-only'),),'unchanged-cached-Commit')
        logits,trace=from_commit(h,c,mode)
        if not torch.isfinite(logits).all() or torch.cuda.max_memory_allocated()>12*1024**3:raise ValueError('numeric/GPU cap')
        for case in selected:raw[key+'/'+case['name']]=logits[case['offset']//16].float().cpu().clone()
        io.atomic_json(output/'progress.json',dict(phase='readonly_precision_and_context_intervention',calls=calls,total=133,native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
        del logits,c;return trace
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        for h in heads.values():h.to('cuda')
        for key,info in inputs.items():
            selected=[c for c in cases if c['input']==key];z,reference,targets=commit_of(p,info);z=z.to('cuda')
            for version,h in heads.items():
                for mode in MODES:invoke(h,z,mode,version+'/'+mode,selected)
            # Stored native V80 diagonal. Long was FP32, short was BF16.
            normal='full_fp32' if key.startswith('long') else 'bf16'
            for case in selected:
                patch=case['offset']//16;role=case['offset']%16;native=reference[patch].float();now=raw['V80/'+normal+'/'+case['name']]
                if int(targets.flatten()[case['offset']])!=case['gold']:raise ValueError('source-bound label integrity')
                raw['native/'+case['name']]=native.clone()
                records.append(dict(case=case,normal_mode=normal,diagonal_focus_matches_native=int(now[role].argmax())==int(native[role].argmax()),
                    diagonal_patch_argmax_agreement=float((now.argmax(-1)==native.argmax(-1)).float().mean()),diagonal_max_abs=float((now-native).abs().max())))
            del z,reference,targets;gc.collect()
        for case in cases:
            if 'partner' not in case:continue
            info=inputs[case['input']];z,_,_=commit_of(p,info);other,_,_=commit_of(p,case['partner']);slot=2+case['offset']//16
            if z.shape!=other.shape:raise ValueError('matched intervention shape')
            for kind in ('global_swap','local_swap','both_swap','zero_global','partner_native'):
                value=z.clone()
                if kind in ('global_swap','both_swap'):value[:,:2]=other[:,:2]
                if kind in ('local_swap','both_swap'):value[:,slot]=other[:,slot]
                if kind=='zero_global':value[:,:2]=0
                if kind=='partner_native':value=other
                invoke(heads['V80'],value.to('cuda'),'full_fp32','V80/'+kind,[case])
            del z,other
        if calls!=133:raise ValueError('exact expected diagnostic calls')
        for version,h in heads.items():
            if support.state_sha(h.state_dict())!=proof[version]['head_SHA']:raise ValueError('immutable weights changed')
        with (output/'focused_patch_logits.pt').open('xb') as f:torch.save(raw,f)
        io.atomic_json(output/'case_diagonals.json',records)
        io.atomic_json(output/'result.json',dict(status='READONLY_READOUT_CAUSE_V81_COMPLETE',calls=calls,seconds=time.time()-began,selected_cases=12,unique_normal_inputs=11,
            optimizer_steps=0,encoder_calls=0,denoise_calls=0,checkpoints=0,Exact_entries=0,all_weights_unchanged=True,parent_Adam_scaler_ALLRNG_preserved=True,
            heads=proof,peak_GPU_bytes=torch.cuda.max_memory_allocated(),raw_SHA=support.sha(output/'focused_patch_logits.pt'),
            scope='Purposive observed cases. Labels choose intervention/score only; no ordinary decoder feature. Off-diagonal memory states are diagnostic, not deployable repair. No whole-model semantic-completeness or unseen-generalization claim.'))
        return 0
    except BaseException as error:
        io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc(),calls=calls));return 2

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');args=a.parse_args()
    raise SystemExit(run(args.plan.resolve(),args.output.resolve(),args.preflight))
