"""ONE cached E(X)->fixed D screen over ALL480 current source rows.
Only byte-exact/EOS/UTF8 TRAIN rows qualify for later thought-only identity work.
No encoder call, thought call, optimizer, target-answer fitting, or GPU.
"""
import argparse,hashlib,io as memio,json,shutil,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_thought_cache_v95 import load_codec
from research.thought_cache_v95 import assemble_input
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.readout_decision_trace_v88 import pack as trace_pack,decode

CAP=96*1024**2

def main(path):
    began=time.monotonic();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    p=io.load_json(path)
    assert p['schema']=='cached_roundtrip_all480_identity_screen_v108' and not p['authorized'] and p['CPU_fixture_only']
    assert p['max_calls']==480 and p['max_output_bytes']==CAP and p['work_seconds']==75
    assert shutil.disk_usage(P).free>=108*1024**3+CAP
    for f,sha in p['files'].items():assert support.sha(f)==sha,f
    out=path.parent/'cpu_preflight_owned/runtime';out.mkdir(exist_ok=False)
    codec,epoch,before,selection=load_codec(p,'cpu');cache=Path(p['cache']);data=Path(p['data'])
    cached=io.load_json(cache/'result.json');static=torch.load(cached['static_memories']['path'],map_location='cpu',weights_only=True)
    indices=io.load_json(cache/'input_index.json');rows=[];ledger=0;calls=0
    assert sum(map(len,indices.values()))==480
    for split in ('train','dev_pair','dev_surface','dev_position'):
        features=io.load_json(data/(split+'_features.json'));refs={x['id']:x for x in indices[split]}
        assert len(features)==(384 if split=='train' else 32) and set(refs)=={x['id'] for x in features}
        for row in features:
            if (path.parent/'STOP').exists() or (path.parent/'cpu_preflight_owned/STOP').exists():raise InterruptedError('cooperative STOP')
            if time.monotonic()-began>=75 or calls>=480:raise TimeoutError('NEW75seconds/480D cap')
            assert set(row)=={'id','prompt'}
            ref=refs[row['id']];assert support.sha(ref['path'])==ref['SHA']
            dynamic=torch.load(ref['path'],map_location='cpu',weights_only=True)
            source_SHA=hashlib.sha256(row['prompt'].encode()).hexdigest();assert dynamic['source_SHA']==source_SHA and dynamic['source_id']==row['id']
            m=assemble_input(static,dynamic,'cpu');assert not m.exact_valid.any()
            with torch.no_grad():
                commit=DifferentiableCodecCommit.from_semantic(m.source,m.memory_revisions,m.model_epoch)
                calls+=1;logits=codec.verbalize(commit).float()
                labels=torch.full(logits.shape[:2],257,dtype=torch.long);bs=row['prompt'].encode();labels.flatten()[:len(bs)]=torch.tensor(list(bs));labels.flatten()[len(bs)]=256
                trace=trace_pack(logits,labels,list(range(len(logits))));d=decode(trace)
            passed=bool(d['EOS'] and d['UTF8'] and d['text']==row['prompt'])
            norm=' '.join(d['text'].split())==' '.join(row['prompt'].split())
            raw=dict(schema='cached_source_E_then_frozen_D_FULL258_v108',source_id=row['id'],source_SHA=source_SHA,
                source_condition_ref=ref,trace=trace,source_encoder_calls0=True,thought_calls0=True,all_full258_positions_saved=True)
            stream=memio.BytesIO();torch.save(raw,stream);payload=stream.getvalue()
            assert ledger+len(payload)+2*1024**2<=CAP and shutil.disk_usage(P).free>=100*1024**3+len(payload)
            rawpath=out/('raw_'+row['id']+'.pt')
            with rawpath.open('xb') as f:f.write(payload)
            ledger+=len(payload)
            rr=dict(id=row['id'],split=split,prompt=row['prompt'],output=d['text'],EOS=d['EOS'],UTF8=d['UTF8'],
                source_SHA=source_SHA,strict_exact=passed,normalized_only_diagnostic=norm,
                expected_prefix=d['text'].startswith(row['prompt']),source_condition_ref=ref,
                raw=dict(path=str(rawpath),SHA=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
            rows.append(rr)
            if len(rows)%32==0:io.atomic_json(out/'progress.json',dict(completed=len(rows),readout_calls=calls,strict_TRAIN=sum(x['strict_exact'] and x['split']=='train' for x in rows),seconds=time.monotonic()-began))
    assert calls==len(rows)==480 and not torch.cuda.is_initialized()
    assert support.state_sha(codec.state_dict())==before and all(not x.requires_grad and x.grad is None for x in codec.parameters())
    bysplit={s:dict(rows=sum(x['split']==s for x in rows),strict_exact=sum(x['split']==s and x['strict_exact'] for x in rows),
        normalized=sum(x['split']==s and x['normalized_only_diagnostic'] for x in rows),prefix=sum(x['split']==s and x['expected_prefix'] for x in rows)) for s in indices}
    train=[dict(id=x['id'],source_SHA=x['source_SHA'],source_condition_ref=x['source_condition_ref'],verified_roundtrip_raw=x['raw']) for x in rows if x['split']=='train' and x['strict_exact']]
    io.atomic_json(out/'all480.json',rows);io.atomic_json(out/'selected_TRAIN_only.json',train)
    report=dict(status='ALL480_CACHED_LANGUAGE_MEANING_LANGUAGE_SCREEN_COMPLETE',splits=bysplit,selected_TRAIN=len(train),
        selected_DEV_for_training=0,criterion='literal_UTF8byte_equal_AND_neural_EOS_AND_valid_UTF8; no prefix/trim/fuzzy acceptance',
        calls=dict(readout=480,encode=0,denoise=0),optimizer_steps=0,codec_frozen_SHA_before_after=before,
        runtime_seconds=time.monotonic()-began,all480_source_cache_reused=True,raw_bytes=ledger,
        next='ONLY strict TRAIN selected; identity meaning denoising first, not relation-answer success',scope_closed=True,CUDA_initialized=False)
    io.atomic_json(out/'result.json',report)
    print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(diagnostic_only=True,source_encode0=True,readout480=True,optimizer0=True,allcodec_frozen=True,CUDA_initialized=False)))
    print('CPU_PREFLIGHT_PASS')

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--preflight',action='store_true');a=q.parse_args();assert a.preflight;main(a.plan.resolve())
