"""Final largest two observed V69 inputs on NEW trained weights, no new baseline."""
import gc,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from research.streaming_fuzzy_codec_v69 import StreamingFuzzyCodecV69
from research.fuzzy_roundtrip_v64 import FuzzyRoundTripV64,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.fixed_utf8_readout_v69 import decode_long_commit
from research.text_diffusion_v1 import pack_prompts
from run_semantic_binding_20261111_v61 import save_tensor
import learning_io_v2 as io
import text_generation_support_v1 as support

def clone_for_long(model,device):
    enc=StreamingFuzzyCodecV69(Config(**model.encoder.c.__dict__),input_capacity=16384,stream_capacity=262144,compute_chunk=2048)
    enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks();enc.load_state_dict(model.encoder.state_dict(),strict=True)
    head=OrderedFuzzyReadout();head.load_state_dict(model.readout.state_dict(),strict=True)
    copied=FuzzyRoundTripV64(enc,head).eval().requires_grad_(False)
    if support.state_sha(copied.state_dict())!=support.state_sha(model.state_dict()):raise ValueError('trained long wrapper exact parity')
    return copied.to(device)

@torch.no_grad()
def assess_long(engine,progress):
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=clone_for_long(engine.model,engine.device);budget=engine.budget;records=[];last=0
    rows=io.load_json(Path(budget.p['long_data'])/'features.json')[-2:]
    def guard(kind,start,end,layer):
        nonlocal last
        budget.check()
        if kind=='layer':
            budget.block_work+=1
            if budget.block_work>2304:raise ValueError('fixed final long block cap')
        elif kind=='chunk':
            budget.charge('stream_chunk')
            if budget.calls['stream_chunk']>256:raise ValueError('fixed final long chunk cap')
            import time
            if time.monotonic()-last>2:progress('long_retention');last=time.monotonic()
    model.encoder.stream_guard=guard
    for i,row in enumerate(rows):
        b=row['prompt'].encode();x=pack_prompts([row['prompt']],len(b),engine.device);budget.charge('encode');budget.charge('readout')
        z,c,ep=model(x,source_ids=(row['id'],),source_versions=(budget.p['cycle_sha'],),model_epoch=engine.plan_sha+':final_long')
        y=torch.full(z.shape[:2],257,device=engine.device,dtype=torch.long);flat=y.flatten();flat[:len(b)]=torch.tensor(list(b),device=engine.device);flat[len(b)]=256
        if ep.memories.exact.valid.any() or not torch.isfinite(z).all():raise ValueError('numeric/Exact final long')
        decoded=decode_long_commit(z,budget.check);text=decoded.pop('text');ce=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'))
        raw_sha=save_tensor(budget.output/f'raw_long_{i}.pt',dict(logits=z.cpu(),targets=y.cpu(),fuzzy_commit=c.reader_tensor().cpu(),Exact_entries=0,Exact_policy=EXACT_POLICY),budget)
        records.append(dict(id=row['id'],source_bytes=len(b),output=text,raw_sha256=raw_sha,NLL_sum=ce,correct=int(((z.argmax(-1)==y)&(y!=257)).sum()),
            byte_correct=int(((z.argmax(-1)==y)&(y<256)).sum()),stream_stats=model.encoder.stream_stats,**decoded))
        del x,z,c,ep,y;gc.collect()
    io.atomic_json(budget.output/'long_reconstructions.json',records)
    del model;gc.collect();torch.cuda.empty_cache()
    return dict(rows=2,trained_input_max_bytes=16384,test_input_max_bytes=262042,stream_chunks=budget.calls['stream_chunk'],block_work=budget.block_work,
        UTF8=sum(r['UTF8'] for r in records),EOS=sum(r['EOS'] for r in records),NLL=sum(r['NLL_sum'] for r in records)/sum(r['source_bytes']+1 for r in records),
        baseline_reused='V69 saved raw on SAME two observed inputs; this is a new trained-weight comparison, not fresh blind data')
