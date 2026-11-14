"""All-row fresh source assessments, including eight new length challenges."""
import gc,time
from pathlib import Path
import torch
from research.text_diffusion_v1 import pack_prompts
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from frozen_verbalizer_assess_v72 import score_store,summarize
from factual_cycle_long_eval_v70 import clone_for_long
import learning_io_v2 as io

@torch.no_grad()
def assess_rows(engine,rows,label,progress):
    engine.model.eval();records=[]
    for i,row in enumerate(rows):
        engine.budget.charge('encode');engine.budget.charge('readout');x=pack_prompts([row['prompt']],len(row['prompt'].encode()),engine.device)
        with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):
            z,c,ep=engine.model(x,source_ids=(row['id'],),source_versions=(engine.budget.data_sha,),model_epoch=engine.plan_sha+':'+label)
        if ep.memories.exact.valid.any():raise ValueError('assessment Exact NULL')
        records.append(score_store(engine,row,z,c,label,i,'fresh_source_memories_Commit_language'));del x,z,c,ep
        if i%8==0:progress(label)
    io.atomic_json(engine.budget.output/(label+'_reconstructions.json'),records);engine.evaluations[label]=summarize(records)
    return records

@torch.no_grad()
def baseline(engine,progress):
    return assess_rows(engine,io.load_json(Path(engine.budget.p['eval_data'])/'short_features.json'),'new_baseline',progress)

@torch.no_grad()
def final_assessment(engine,progress):
    p=engine.budget.p;data=Path(p['parent_data']);evaldata=Path(p['eval_data'])
    for label,rows in [('OWT',io.load_json(p['retention_OWT'])),('retained_English',io.load_json(data/'retained_English.json')),
        ('prior_English',io.load_json(data/'dev_features.json')),('old_expression',io.load_json(Path(p['old_expression_data'])/'dev_features.json')),
        ('new_English',io.load_json(evaldata/'short_features.json'))]:assess_rows(engine,rows,label,progress)
    model=clone_for_long(engine.model,engine.device);budget=engine.budget;last=0.
    def guard(kind,start,end,layer):
        nonlocal last
        budget.check()
        if kind=='layer':
            budget.block_work+=1
            if budget.block_work>6642:raise ValueError('fixed long blockwork cap')
        if kind=='chunk':
            budget.charge('stream_chunk')
            if budget.calls['stream_chunk']>738:raise ValueError('fixed long chunks')
            if time.monotonic()-last>2:progress('fresh_long_source');last=time.monotonic()
    model.encoder.stream_guard=guard
    # Same final weights, same 4096-byte input: execution parity before the
    # beyond-training-length assessment, not an extra performance baseline.
    row=io.load_json(evaldata/'long_features.json')[2];prompt=row['prompt'][:4096];x=pack_prompts([prompt],len(prompt.encode()),engine.device);values=[]
    for label,instance in [('native',engine.model),('stream',model)]:
        budget.charge('encode');budget.charge('readout');instance.eval()
        z,c,ep=instance(x,source_ids=('execution_parity73',),source_versions=(budget.data_sha,),model_epoch=engine.plan_sha+':parity')
        if ep.memories.exact.valid.any():raise ValueError('parity Exact NULL')
        values.append(dict(logits=z.cpu(),commit=c.reader_tensor().cpu()))
    a,b=values;relative=float((a['commit'].double()-b['commit'].double()).norm()/a['commit'].double().norm().clamp_min(1e-12));same=torch.equal(a['logits'].argmax(-1),b['logits'].argmax(-1))
    if relative>1e-4 or not same:raise ValueError('actual final native/stream execution parity')
    from run_semantic_binding_20261111_v61 import save_tensor
    rawsha=save_tensor(budget.output/'raw_execution_parity.pt',dict(native=a,stream=b),budget)
    io.atomic_json(budget.output/'execution_parity.json',dict(commit_relative_L2=relative,all_argmax_identical=same,raw_SHA=rawsha,NNcalls=4,stream_chunks=2,source_bytes=len(prompt.encode())))
    del x,z,c,ep,values,a,b;gc.collect();records=[]
    for i,row in enumerate(io.load_json(evaldata/'long_features.json')):
        budget.charge('encode');budget.charge('readout');x=pack_prompts([row['prompt']],len(row['prompt'].encode()),engine.device)
        torch.cuda.reset_peak_memory_stats()
        z,c,ep=model(x,source_ids=(row['id'],),source_versions=(budget.data_sha,),model_epoch=engine.plan_sha+':final_long')
        if ep.memories.exact.valid.any():raise ValueError('long Exact NULL')
        record=score_store(engine,row,z,c,'long',i,'fresh_final_weights_streaming_source',long=True);record['stream_stats']=dict(model.encoder.stream_stats);record['GPU_peak_bytes']=torch.cuda.max_memory_allocated()
        if record['GPU_peak_bytes']>12*1024**3:raise MemoryError('longGPU12GiB')
        records.append(record);del x,z,c,ep;gc.collect();progress('long_row_complete')
    io.atomic_json(budget.output/'long_reconstructions.json',records);engine.evaluations['long']=summarize(records)
    engine.evaluations['long'].update(source_execution='all10 fresh final upstream; eight new inputs/two observed not TRAIN',chunks=budget.calls['stream_chunk'],block_work=budget.block_work)
    del model;gc.collect();torch.cuda.empty_cache()
