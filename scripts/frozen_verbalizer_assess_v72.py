"""Cached fixed-upstream readout comparisons and four fresh full-path checks."""
import hashlib,json
from pathlib import Path
import torch
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from research.text_diffusion_v1 import pack_prompts
from research.codec_commit_stop_v56 import reconstruct_commit_logits
from research.fixed_utf8_readout_v69 import decode_long_commit
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support
import learning_io_v2 as io

def target_like(logits,row):
    b=row['prompt'].encode();y=torch.full(logits.shape[:2],257,dtype=torch.long,device=logits.device)
    y.flatten()[:len(b)]=torch.tensor(list(b),device=logits.device);y.flatten()[len(b)]=256
    return y

def score_store(engine,row,logits,commit,label,index,mode,long=False):
    budget=engine.budget;z=logits.float();y=target_like(z,row)
    if not torch.isfinite(z).all():raise FloatingPointError('evaluation logits')
    decoded=decode_long_commit(z,budget.check) if long else reconstruct_commit_logits(z)
    text=decoded.pop('text');n=int((y!=257).sum());nll=float(torch.nn.functional.cross_entropy(z.transpose(1,2),y,ignore_index=257,reduction='sum'))
    raw_path=budget.output/f'raw_{label}_{index:02d}.pt'
    sha=save_tensor(raw_path,dict(logits=z.cpu(),targets=y.cpu(),fuzzy_commit=commit.reader_tensor().cpu(),Exact_entries=0,Exact_policy=EXACT_POLICY,evaluation_mode=mode),budget)
    return dict(id=row['id'],source=row['prompt'],output=text,symbols=n,source_bytes=len(row['prompt'].encode()),NLL_sum=nll,
        correct=int(((z.argmax(-1)==y)&(y!=257)).sum()),raw_sha256=sha,evaluation_mode=mode,exact=text==row['prompt'],**decoded)

def summarize(records):
    n=sum(r['symbols'] for r in records)
    return dict(rows=len(records),NLL=sum(r['NLL_sum'] for r in records)/n,byte_EOS_accuracy=sum(r['correct'] for r in records)/n,
        exact_auxiliary=sum(r['exact'] for r in records),UTF8=sum(r['UTF8'] for r in records),EOS=sum(r['EOS'] for r in records),symbols=n)

@torch.no_grad()
def new_dev_baseline(engine,rows,labels,progress):
    directory=engine.budget.output/'new_DEV_cache';directory.mkdir();refs=[];records=[]
    for i,(row,label) in enumerate(zip(rows,labels)):
        f,t,z=engine.encode_packet(row,label['critical_spans']);fp=directory/f'{i:02d}.features.pt'
        sha=save_tensor(fp,f,engine.budget);c=DifferentiableCodecCommit.from_semantic(f['semantic'],f['memory_revisions'],engine.embedding_sha)
        records.append(score_store(engine,row,z,c,'new_baseline',i,'fresh_source_before_head_training'))
        refs.append(dict(features=str(fp),features_SHA=sha,cache_key=f['cache_key']))
        if i%8==0:progress('new_DEV_initial_source_path')
    io.atomic_json(engine.budget.output/'new_baseline_reconstructions.json',records)
    engine.evaluations['new_baseline']=summarize(records);return refs

def commit_from_cache(engine,row,packet):
    # The embedding digest is checked against the unchanged full V70 upstream
    # model. Saved targets are data-integrity checks, never decoder features.
    z=packet['fuzzy_commit'].to(engine.device)
    revisions=packet.get('memory_revisions',((row['id'],'verified_V70_saved_commit',hashlib.sha256(row['prompt'].encode()).hexdigest()),))
    return DifferentiableCodecCommit.from_semantic(z,revisions,engine.embedding_sha)

@torch.no_grad()
def final_assessment(engine,dev_rows,new_dev_refs,progress):
    p=engine.budget.p;parent=Path(p['baseline_run']);d70=Path(p['parent_data']);records_by_label={}
    specs=[('OWT',io.load_json(p['retention_OWT']),[parent/f'raw_final_{i:02d}.pt' for i in range(16)],False),
        ('retained_English',io.load_json(d70/'retained_English.json'),[parent/f'raw_retained_English_{i:02d}.pt' for i in range(36)],False),
        ('prior_English',io.load_json(d70/'dev_features.json'),[parent/f'raw_new_English_{i:02d}.pt' for i in range(32)],False),
        ('long',io.load_json(Path(p['long_data'])/'features.json')[-2:],[parent/f'raw_long_{i}.pt' for i in range(2)],True)]
    engine.model.eval()
    for label,rows,files,long in specs:
        records=[]
        for i,(row,file) in enumerate(zip(rows,files)):
            engine.budget.check();packet=torch.load(file,map_location='cpu',weights_only=True);c=commit_from_cache(engine,row,packet)
            engine.budget.charge('readout')
            with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda' and not long):z=engine.model.verbalize(c)
            if not torch.equal(target_like(z,row).cpu(),packet['targets']):raise ValueError('parent cache row identity')
            records.append(score_store(engine,row,z,c,label,i,'saved_V70_unchanged_upstream_readout',long));del packet,z,c
            if i%8==0:progress('cached_final_'+label)
        io.atomic_json(engine.budget.output/(label+'_reconstructions.json'),records);engine.evaluations[label]=summarize(records);records_by_label[label]=records
    records=[]
    for i,(row,ref) in enumerate(zip(dev_rows,new_dev_refs)):
        if support.sha(ref['features'])!=ref['features_SHA']:raise ValueError('new DEV cache SHA')
        f=torch.load(ref['features'],map_location='cpu',weights_only=True)
        if f['embedding_sha']!=engine.embedding_sha or f['cache_key']!=engine.source_key(row):raise ValueError('new DEV source epoch')
        c=DifferentiableCodecCommit.from_semantic(f['semantic'].to(engine.device),f['memory_revisions'],engine.embedding_sha);engine.budget.charge('readout')
        with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):z=engine.model.verbalize(c)
        records.append(score_store(engine,row,z,c,'new_expression',i,'new_DEV_fixed_upstream_readout'));del z,c
    io.atomic_json(engine.budget.output/'new_expression_reconstructions.json',records);engine.evaluations['new_expression']=summarize(records);records_by_label['new_expression']=records
    return records_by_label

@torch.no_grad()
def full_path_spotchecks(engine,dev_rows,all_records,progress):
    p=engine.budget.p;d70=Path(p['parent_data']);held=io.load_json(d70/'retained_English.json')
    cases=[(held[4],all_records['retained_English'][4]),(held[32],all_records['retained_English'][32]),(dev_rows[0],all_records['new_expression'][0]),(dev_rows[1],all_records['new_expression'][1])]
    engine.model.eval();records=[];parity=[]
    for i,(row,cached) in enumerate(cases):
        b=row['prompt'].encode();x=pack_prompts([row['prompt']],len(b),engine.device);engine.budget.charge('encode');engine.budget.charge('readout')
        with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):
            z,c,ep=engine.model(x,source_ids=(row['id'],),source_versions=(engine.budget.data_sha,),model_epoch=engine.embedding_sha)
        if ep.memories.exact.valid.any():raise ValueError('fresh source Exact must be NULL')
        record=score_store(engine,row,z,c,'fresh_spot',i,'fresh_full_language_memory_language');records.append(record)
        raw=torch.load(engine.budget.output/f"raw_{'retained_English' if i<2 else 'new_expression'}_{(4 if i==0 else 32) if i<2 else i-2:02d}.pt",map_location='cpu',weights_only=True)
        old=raw['fuzzy_commit'].double();new=c.reader_tensor().cpu().double();relative=float((old-new).norm()/old.norm().clamp_min(1e-12))
        agreement=float((raw['logits'].argmax(-1)==z.cpu().argmax(-1)).float().mean())
        parity.append(dict(id=row['id'],commit_relative_L2=relative,all_argmax_agreement=agreement,same_decoded_text=record['output']==cached['output']))
        if relative>1e-4 or agreement!=1. or record['output']!=cached['output']:raise ValueError('fresh full path differs from fixed-upstream cache')
        progress('fresh_full_path_spotchecks')
    io.atomic_json(engine.budget.output/'fresh_spot_reconstructions.json',records);io.atomic_json(engine.budget.output/'fresh_cache_parity.json',parity)
    engine.evaluations['fresh_spot']=summarize(records)
