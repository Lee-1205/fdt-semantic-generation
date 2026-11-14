"""One actual saved-state serialization roundtrip; actual model forward/SGD zero."""
import datetime,gc,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research import linked_runtime_v75 as linked

def main():
    started=time.monotonic();torch.set_num_threads(4)
    run=ROOT/'runs/context_verbalizer_20261112_v74';p=io.load_json(run/'plan.json');r=io.load_json(run/'root_result.json')
    assert r['all_owned_and_proxy_gone_before_analysis'] and shutil.disk_usage(ROOT).free>100*1024**3+128*1024**2
    out=ROOT/'research/artifacts/linked_runtime_20261112_v75_cpu';out.mkdir(exist_ok=False)
    source=run/'model/checkpoints/state_0640.pt';parent=Path(p['cycle_source'])
    assert support.sha(source)==r['final_state']['sha256'] and support.sha(parent)==p['cycle_sha']
    full=torch.load(source,map_location='cpu',weights_only=True);original=support.state_sha(full)
    assert original==r['final_state']['algorithm_sha256']
    base=torch.load(parent,map_location='cpu',weights_only=True);names=sum(full['optimizer_names'],[])
    assert len(names)==len(full['optimizer']['state'])==7 and all(n.startswith('readout.readout.') for n in names)
    reference=dict(path=str(parent.resolve()),SHA=p['cycle_sha'],model_key='model',model_SHA=support.state_sha(base['model']))
    envelope=linked.pack(full,base['model'],reference,names,support.state_sha)
    del base;gc.collect()
    meta=linked.save_verified(out/'actual_V74_final_linked_runtime.pt',envelope,ROOT,support.state_sha)
    assert meta['bytes']<64*1024**2
    restored=linked.load_verified(meta['path'],meta['SHA'],ROOT,support.state_sha)
    assert support.state_sha(restored)==original and support.state_sha(full)==original
    assert restored['attempts']==restored['updates']==restored['cursor']==640 and restored['inflight'] is None
    checked={}
    for k in ('model','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng','orders','cursor','history','exposure','calls','cache_refs','new_dev_refs','embedding_sha','teacher_sha','parent_reference'):
        checked[k]=support.state_sha(restored[k]);assert checked[k]==support.state_sha(full[k])
    assert all(int(s['step'])==640 for s in restored['optimizer']['state'].values())
    assert support.sha(source)==r['final_state']['sha256'] and support.sha(parent)==p['cycle_sha']
    assert not torch.cuda.is_initialized() and time.monotonic()-started<120
    proof=dict(status='CPU_LINKED_RUNTIME_ACTUAL_V74_FULLSTATE_EXACT_ROUNDTRIP_PASS',time=datetime.datetime.now().astimezone().isoformat(),
        seconds=time.monotonic()-started,source=dict(path=str(source),SHA=r['final_state']['sha256'],bytes=source.stat().st_size,algorithm_SHA=original),
        linked=meta,active_tensors=7,active_parameters=sum(full['model'][n].numel() for n in names),checked_component_digests=checked,
        five_focused_CPU_contracts_passed=True,actual_model_forward_calls=0,actual_optimizer_steps=0,CUDA_initialized=False,
        actual_V75_worker_or_GPU_trial_ready=False,not_actual_CUDA_restart_proof=True,
        limitations='This is a parent-required representation of existing V74 state. Old standalone source and parent unchanged. V75 engine/plan/owner/preflight absent. Inference/Exact policies unchanged.',
        code_SHA={str(f):support.sha(f) for f in [ROOT/'research/linked_runtime_v75.py',ROOT/'scripts/test_linked_runtime_v75_cpu.py',Path(__file__)]},
        C_free_bytes=shutil.disk_usage(ROOT).free)
    io.atomic_json(out/'root_CPU_review.json',proof);io.atomic_json(out/'linked_manifest.json',meta)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('checked_component_digests','code_SHA')}))
if __name__=='__main__':main()
