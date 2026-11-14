"""Fixed V73 upstream: cached-boundary comparison plus two fresh max-length paths."""
import gc,time
from pathlib import Path
import torch
from research.text_diffusion_v1 import pack_prompts
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from frozen_verbalizer_assess_v72 import score_store,summarize,commit_from_cache
from factual_cycle_long_eval_v70 import clone_for_long
from run_semantic_binding_20261111_v61 import save_tensor
import learning_io_v2 as io
import text_generation_support_v1 as support
@torch.no_grad()
def assess(engine,newrows,progress):
    p=engine.budget.p;run=Path(p['baseline_run']);data=Path(p['parent_data']);engine.model.eval()
    specs=[('OWT',io.load_json(p['retention_OWT']),'OWT'),('retained_English',io.load_json(data/'retained_English.json'),'retained_English'),
        ('old_untrained128',io.load_json(Path(p['old_eval_data'])/'short_features.json'),'new_English'),('old_expression',io.load_json(Path(p['old_expression_data'])/'dev_features.json'),'old_expression')]
    for label,rows,oldlabel in specs:
        records=[]
        for i,row in enumerate(rows):
            packet=torch.load(run/f'raw_{oldlabel}_{i:02d}.pt',map_location='cpu',weights_only=True);c=commit_from_cache(engine,row,packet);engine.budget.charge('readout')
            with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):z=engine.model.verbalize(c)
            records.append(score_store(engine,row,z,c,label,i,'saved_V73_identical_upstream_readout'));del packet,z,c
            if i%8==0:progress('cached_'+label)
        io.atomic_json(engine.budget.output/(label+'_reconstructions.json'),records);engine.evaluations[label]=summarize(records)
    records=[]
    for i,(row,ref) in enumerate(zip(newrows,engine.new_dev_refs)):
        if support.sha(ref['features'])!=ref['features_SHA']:raise ValueError('new DEV cache SHA')
        f=torch.load(ref['features'],map_location='cpu',weights_only=True)
        if f['embedding_sha']!=engine.embedding_sha or f['cache_key']!=engine.source_key(row):raise ValueError('new DEV embedding/source epoch')
        c=DifferentiableCodecCommit.from_semantic(f['semantic'].to(engine.device),f['memory_revisions'],engine.embedding_sha);engine.budget.charge('readout')
        with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):z=engine.model.verbalize(c)
        records.append(score_store(engine,row,z,c,'new_English',i,'new_DEV_same_fixed_source_Commit'));del z,c
        if i%8==0:progress('cached_new_English')
    io.atomic_json(engine.budget.output/'new_English_reconstructions.json',records);engine.evaluations['new_English']=summarize(records)
    longrows=io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2];cached=[]
    for i,row in enumerate(longrows):
        packet=torch.load(run/f'raw_long_{i:02d}.pt',map_location='cpu',weights_only=True);c=commit_from_cache(engine,row,packet);engine.budget.charge('readout');z=engine.model.verbalize(c)
        cached.append(score_store(engine,row,z,c,'cached_long',i,'saved_V73_identical_upstream_readout',long=True));del packet,z,c
    io.atomic_json(engine.budget.output/'cached_long_reconstructions.json',cached);engine.evaluations['cached_long']=summarize(cached)
    model=clone_for_long(engine.model,engine.device);budget=engine.budget;last=0.
    def guard(kind,start,end,layer):
        nonlocal last
        budget.check()
        if kind=='layer':
            budget.block_work+=1
            if budget.block_work>2322:raise ValueError('V74 fixed block work')
        elif kind=='chunk':
            budget.charge('stream_chunk')
            if budget.calls['stream_chunk']>258:raise ValueError('V74 fixed chunks')
            if time.monotonic()-last>2:progress('fresh_long_source');last=time.monotonic()
    model.encoder.stream_guard=guard
    x=pack_prompts([longrows[0]['prompt'][:4096]],4096,engine.device);parity=[]
    for instance in (engine.model,model):
        budget.charge('encode');budget.charge('readout');z,c,ep=instance(x,source_ids=('parity74',),source_versions=(budget.data_sha,),model_epoch=engine.plan_sha)
        parity.append(dict(logits=z.cpu(),commit=c.reader_tensor().cpu()));assert not ep.memories.exact.valid.any()
    relative=float((parity[0]['commit'].double()-parity[1]['commit'].double()).norm()/parity[0]['commit'].double().norm().clamp_min(1e-12));same=torch.equal(parity[0]['logits'].argmax(-1),parity[1]['logits'].argmax(-1))
    if relative>1e-4 or not same:raise ValueError('native/stream execution parity')
    digest=save_tensor(budget.output/'raw_execution_parity.pt',dict(native=parity[0],stream=parity[1]),budget);io.atomic_json(budget.output/'execution_parity.json',dict(commit_relative_L2=relative,all_argmax_identical=same,raw_SHA=digest))
    del x,z,c,ep,parity;gc.collect();fresh=[];agreement=[]
    for i,row in enumerate(longrows):
        budget.charge('encode');budget.charge('readout');x=pack_prompts([row['prompt']],len(row['prompt'].encode()),engine.device);torch.cuda.reset_peak_memory_stats()
        z,c,ep=model(x,source_ids=(row['id'],),source_versions=(budget.data_sha,),model_epoch=engine.plan_sha+':fresh_long');assert not ep.memories.exact.valid.any()
        record=score_store(engine,row,z,c,'long',i,'fresh_full_source_frozen_upstream_new_readout',long=True);record['stream_stats']=dict(model.encoder.stream_stats);record['GPU_peak_bytes']=torch.cuda.max_memory_allocated()
        if record['GPU_peak_bytes']>12*1024**3:raise MemoryError('long GPU12GiB')
        fresh.append(record);old=torch.load(budget.output/f'raw_cached_long_{i:02d}.pt',map_location='cpu',weights_only=True);a=old['fuzzy_commit'].double();b=c.reader_tensor().cpu().double();relative=float((a-b).norm()/a.norm().clamp_min(1e-12));same=torch.equal(old['logits'].argmax(-1),z.cpu().argmax(-1))
        if relative>1e-4 or not same or record['output']!=cached[i]['output']:raise ValueError('fresh long path differs from fixed-upstream cache')
        agreement.append(dict(id=row['id'],commit_relative_L2=relative,all_argmax_identical=same,same_decoded_output=True));del x,z,c,ep,old,a,b;gc.collect();progress('fresh_long_complete')
    io.atomic_json(budget.output/'long_reconstructions.json',fresh);io.atomic_json(budget.output/'fresh_long_cache_parity.json',agreement);engine.evaluations['long']=summarize(fresh)
    del model;gc.collect();torch.cuda.empty_cache()
