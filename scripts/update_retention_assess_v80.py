"""V80 lossless evidence; grammar-revised64 fresh-source reconstruction; observed340/long2 retention.

V75 saved Commit references are resolved exactly one pinned level. No source,
gold labels, or teacher probabilities are inputs to the Commit-only readout.
"""
import gc,hashlib,time
from pathlib import Path
import torch
from research.text_diffusion_v1 import pack_prompts
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from research.codec_commit_stop_v56 import reconstruct_commit_logits
from research.fixed_utf8_readout_v69 import decode_long_commit
from frozen_verbalizer_assess_v72 import target_like,summarize
from saved_commit_reference_v76_v2 import resolve_saved_commit
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from factual_cycle_long_eval_v70 import clone_for_long
from run_semantic_binding_20261111_v61 import save_tensor as ordinary_save_tensor
from research.lossless_raw_tensor_v80 import pack

def save_tensor(path,payload,budget):return ordinary_save_tensor(path,pack(payload),budget)
import learning_io_v2 as io
import text_generation_support_v1 as support

def commit_from_cache(engine,row,packet):
    tensor,_=resolve_saved_commit(packet,row,engine.embedding_sha,engine.budget.p['files'],engine.budget.p['root'],engine.model.encoder.c.z_dim)
    return DifferentiableCodecCommit.from_semantic(tensor.to(engine.device),((row['id'],'verified_fixed_upstream_retention',hashlib.sha256(row['prompt'].encode()).hexdigest()),),engine.embedding_sha)

def store_record(engine,row,z,c,label,index,mode,long=False,reference=None):
    z=z.float();y=target_like(z,row);budget=engine.budget
    if not torch.isfinite(z).all():raise FloatingPointError('V77 evaluation logits')
    decoded=decode_long_commit(z,budget.check) if long else reconstruct_commit_logits(z);text=decoded.pop('text')
    packet=dict(logits=z.cpu(),targets=y.cpu(),Exact_entries=0,Exact_policy=EXACT_POLICY,evaluation_mode=mode,
        embedding_sha=engine.embedding_sha,source_id=row['id'],source_SHA=hashlib.sha256(row['prompt'].encode()).hexdigest())
    if reference is None:packet['fuzzy_commit']=c.reader_tensor().cpu()
    else:packet['fuzzy_commit_reference']=reference
    sha=save_tensor(budget.output/f'raw_{label}_{index:02d}.pt',packet,budget)
    return dict(id=row['id'],source=row['prompt'],output=text,symbols=int((y!=257).sum()),source_bytes=len(row['prompt'].encode()),
        NLL_sum=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum')),
        correct=int(((z.argmax(-1)==y)&(y!=257)).sum()),raw_sha256=sha,evaluation_mode=mode,exact=text==row['prompt'],**decoded)

@torch.no_grad()
def source_panel(engine,rows,label,progress):
    records=[];engine.model.eval()
    for i,row in enumerate(rows):
        c=engine.extract_live(row);engine.budget.charge('readout')
        with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):z=engine.model.verbalize(c)
        records.append(store_record(engine,row,z,c,label,i,'fresh_full_source_fixed_upstream'))
        if i%8==0:progress('fresh_'+label)
    io.atomic_json(engine.budget.output/(label+'_reconstructions.json'),records);engine.evaluations[label]=summarize(records)

@torch.no_grad()
def assess(engine,newrows,progress):
    p=engine.budget.p;base=Path(p['baseline_run']);budget=engine.budget;engine.model.eval()
    assert io.load_json(base/'result.json')['frozen_upstream_SHA']==engine.embedding_sha
    specs=[('OWT',io.load_json(p['retention_OWT']),'OWT'),('retained_English',io.load_json(Path(p['parent_data'])/'retained_English.json'),'retained_English'),
        ('old_untrained128',io.load_json(Path(p['old_eval_data'])/'short_features.json'),'old_untrained128'),('old_expression',io.load_json(Path(p['old_expression_data'])/'dev_features.json'),'old_expression'),
        ('observed_V74_DEV',io.load_json(Path(p['observed_V74_data'])/'dev_features.json'),'observed_V74_DEV'),
        ('observed_V75_DEV',io.load_json(Path(p['observed_V75_data'])/'dev_features.json'),'new_English')]
    for label,rows,oldlabel in specs:
        records=[]
        for i,row in enumerate(rows):
            file=base/f'raw_{oldlabel}_{i:02d}.pt';packet=torch.load(file,map_location='cpu',weights_only=True);c=commit_from_cache(engine,row,packet);budget.charge('readout')
            with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):z=engine.model.verbalize(c)
            if not torch.equal(target_like(z,row).cpu(),packet['targets']):raise ValueError('old cache source identity')
            # Canonicalize to a DIRECT saved Commit rather than building a
            # V77->V75->V74 chain of checkpoint packet references.
            ref=(dict(packet['fuzzy_commit_reference']) if 'fuzzy_commit_reference' in packet else
                dict(path=str(file),SHA=p['files'][str(file)],key='fuzzy_commit',embedding_sha=engine.embedding_sha))
            records.append(store_record(engine,row,z,c,label,i,'V75_identical_frozen_upstream_saved_Commit_onelevel_verified',reference=ref));del packet,c,z
            if i%8==0:progress('cached_'+label)
        io.atomic_json(budget.output/(label+'_reconstructions.json'),records);engine.evaluations[label]=summarize(records)
    source_panel(engine,newrows,'new_English',progress)
    longrows=io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2];model=clone_for_long(engine.model,engine.device);last=0.
    def guard(kind,start,end,layer):
        nonlocal last
        budget.check()
        if kind=='layer':
            budget.block_work+=1
            if budget.block_work>2322:raise ValueError('V77 long block cap')
        elif kind=='chunk':
            budget.charge('stream_chunk')
            if budget.calls['stream_chunk']>258:raise ValueError('V77 stream chunk cap')
            if time.monotonic()-last>2:progress('fresh_long_source');last=time.monotonic()
    model.encoder.stream_guard=guard
    x=pack_prompts([longrows[0]['prompt'][:4096]],4096,engine.device);parity=[]
    for instance in (engine.model,model):
        budget.charge('encode');budget.charge('readout');z,c,ep=instance(x,source_ids=('parity77',),source_versions=(budget.data_sha,),model_epoch=engine.plan_sha)
        parity.append(dict(logits=z.cpu(),commit=c.reader_tensor().cpu()));assert not ep.memories.exact.valid.any()
    relative=float((parity[0]['commit'].double()-parity[1]['commit'].double()).norm()/parity[0]['commit'].double().norm().clamp_min(1e-12));same=torch.equal(parity[0]['logits'].argmax(-1),parity[1]['logits'].argmax(-1))
    if relative>1e-4 or not same:raise ValueError('native/stream parity')
    sha=save_tensor(budget.output/'raw_execution_parity.pt',dict(native=parity[0],stream=parity[1]),budget);io.atomic_json(budget.output/'execution_parity.json',dict(commit_relative_L2=relative,all_argmax_identical=same,raw_SHA=sha))
    del x,z,c,ep,parity;gc.collect();records=[];checks=[]
    for i,row in enumerate(longrows):
        budget.charge('encode');budget.charge('readout');x=pack_prompts([row['prompt']],len(row['prompt'].encode()),engine.device);torch.cuda.reset_peak_memory_stats()
        z,c,ep=model(x,source_ids=(row['id'],),source_versions=(budget.data_sha,),model_epoch=engine.embedding_sha)
        assert not ep.memories.exact.valid.any();record=store_record(engine,row,z,c,'long',i,'fresh_full_source_frozen_upstream_new_head',long=True)
        record.update(stream_stats=dict(model.encoder.stream_stats),GPU_peak_bytes=torch.cuda.max_memory_allocated())
        if record['GPU_peak_bytes']>12*1024**3:raise MemoryError('long GPU12GiB')
        old=torch.load(base/f'raw_long_{i:02d}.pt',map_location='cpu',weights_only=True);oldc=commit_from_cache(engine,row,old);a=old['fuzzy_commit'].double();b=c.reader_tensor().cpu().double()
        relative=float((a-b).norm()/a.norm().clamp_min(1e-12));budget.charge('readout');other=engine.model.verbalize(oldc);same=torch.equal(other.argmax(-1).cpu(),z.argmax(-1).cpu())
        if relative>1e-4 or not same:raise ValueError('fresh long/source epoch parity with current head')
        checks.append(dict(id=row['id'],commit_relative_L2=relative,current_head_all_argmax_identical=same,old_Commit_file=str(base/f'raw_long_{i:02d}.pt'),duplicated_full_cached_long_raw_not_written=True))
        records.append(record);del old,oldc,a,b,other,x,z,c,ep;gc.collect();progress('fresh_long_complete')
    io.atomic_json(budget.output/'long_reconstructions.json',records);io.atomic_json(budget.output/'fresh_long_cache_parity.json',checks);engine.evaluations['long']=summarize(records)
    del model;gc.collect();torch.cuda.empty_cache()
