"""Bounded hardware capacity measurement. No optimizer steps or checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from research.long_memory_codec_v49 import LongMemoryCodec
from research.text_diffusion_v1 import pack_prompts,pack_answers
import text_generation_support_v1 as support
import learning_io_v2 as io


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=io.load_json(args.plan)
    if p['schema']!='fdt_long_codec_capacity_profile_v49' or p['output']!=str(args.output.resolve()):
        raise ValueError('profile schema/output')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:
            raise ValueError('profile pin changed '+name)
    if time.time()>=p['cutoff'] or (args.output/'profile_started.json').exists():
        raise ValueError('one unexpired profile only')
    if support.core_sha(ROOT)!=support.CORE_SHA:
        raise ValueError('original core changed')
    began=time.time()
    io.atomic_json(args.output/'profile_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=began))
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    state=torch.load(p['source'],map_location='cpu',weights_only=True)
    config=Config(**state['core'])
    config.gradient_checkpointing=True
    model=LongMemoryCodec(config,input_capacity=p['lengths'][-1])
    model.enable_position_adapter()
    model.load_state_dict(state['model'],strict=True)
    model.enable_long_capacity()
    model.to('cuda')
    groups=copy.deepcopy(state['optimizer_parameter_names'])
    named=dict(model.named_parameters())
    optimizer=torch.optim.AdamW([{'params':[named[n] for n in group]} for group in groups])
    old_opt=state['optimizer']
    id_to_name={i:name for group,names in zip(old_opt['param_groups'],groups) for i,name in zip(group['params'],names)}
    for i,item in old_opt['state'].items():
        name=id_to_name[i]
        for key,value in list(item.items()):
            if isinstance(value,torch.Tensor) and value.ndim and value.shape!=named[name].shape:
                if name!='decoder.positions.weight' or value.shape[1:]!=named[name].shape[1:]:
                    raise ValueError('unexpected optimizer shape migration')
                extended=value.new_zeros(named[name].shape)
                extended[:len(value)]=value
                item[key]=extended
    optimizer.load_state_dict(old_opt)
    # Reserve the moments that the new local tensors acquire on their first update.
    local_reserve=torch.empty(2*(model.decoder.local_read_gain.numel()+model.decoder.local_slot_tag.numel()),
                              device='cuda',dtype=torch.float32)
    scaler=torch.amp.GradScaler('cuda')
    scaler.load_state_dict(state['scaler'])
    del state,old_opt
    documents=io.load_json(Path(p['corpus'])/'documents.json')['train']
    doc=max(documents,key=lambda d:len(d['text'].encode('utf-8')))
    raw=doc['text'].encode('utf-8')
    rows=[]
    calls=0
    selected=0
    def check():
        if time.time()>=p['cutoff'] or (args.output/'STOP').exists():
            raise InterruptedError('profile fixed clock/STOP')
        import shutil
        if shutil.disk_usage(args.output).free<101*1024**3:
            raise OSError('100GiB protection')
    def charge(kind):
        nonlocal calls
        check()
        calls+=1
        if calls>p['max_calls']:
            raise RuntimeError('profile call cap')
    model.train()
    status='PROFILE_COMPLETE'
    for requested in p['lengths']:
        check()
        if len(raw)<requested:
            status='SOURCE_LENGTH_BOUND'
            break
        text=raw[:requested].decode('utf-8',errors='ignore')
        byte_length=len(text.encode())
        io.atomic_json(args.output/'progress.json',dict(phase='capacity_probe',requested=requested,
            completed=rows,native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        try:
            x=pack_prompts([text],byte_length,'cuda')
            y=pack_answers([text],byte_length+1,'cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss,episode=model.reconstruction_loss(x,y,source_ids=(doc['id'],),
                    source_versions=(p['data_sha'],),model_epoch='capacity_only',charge=charge)
            scaler.scale(loss).backward()
            finite_gradients=all(bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters() if parameter.grad is not None)
            torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()
            reserved=torch.cuda.max_memory_reserved()
            free,total=torch.cuda.mem_get_info()
            finite=bool(torch.isfinite(loss))
            row=dict(requested_bytes=requested,actual_bytes=byte_length,
                     seconds=time.perf_counter()-start,peak_allocated_bytes=peak,
                     peak_reserved_bytes=reserved,GPU_free_bytes=free,finite_loss=finite,
                     loss=float(loss.detach()),finite_scaled_gradients=finite_gradients,
                     commit_slots=episode.commit.reader_tensor().shape[1])
            row['qualified']=finite and finite_gradients and peak<=p['max_allocated_bytes'] and free>=p['min_remaining_GPU_bytes']
            rows.append(row)
            if row['qualified']:
                selected=byte_length
            del loss,episode,x,y
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if not row['qualified']:
                status='MEASURED_RESOURCE_BOUND'
                break
        except torch.cuda.OutOfMemoryError as error:
            rows.append(dict(requested_bytes=requested,OOM=True,error=str(error)[:250]))
            status='MEASURED_OOM_BOUND'
            break
        io.atomic_json(args.output/'progress.json',dict(phase='probe_complete',completed=rows,
            native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    result=dict(status=status,largest_qualified_input_bytes=selected,probes=rows,
                elapsed=time.time()-began,actual_model_calls=calls,optimizer_steps=0,
                checkpoint_writes=0,source_SHA_unchanged=support.sha(p['source'])==p['source_sha'],
                limitation='Largest measured qualified prefix, not a global maximum or reconstruction-quality result.')
    io.atomic_json(args.output/'result.json',result)
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
