"""A bounded semantic-extraction component trial, not language/denoiser training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,io as memoryio,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.semantic_binding_v60 import SemanticBinderV60,BoundMeaningState
from research.text_diffusion_v1 import pack_prompts
from semantic_binding_engine_v60 import SemanticEngine,select_packet,SCHEMA
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget,load_parent,SOURCE_SHA

BINDER_SHA='66ec25782dd2f3a9ea1a84f31971ca5a8140d2d3f58fa665b29b3202bc089f3d'
BINDER_ALGORITHM='f0381ce21cadfe984d9798927604a89f23c16ca3b6587385c1828afc3063e163'

def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_semantic_binding_v60' or p['output']!=str(output.resolve()) or p['source_sha']!=SOURCE_SHA
        or p['wall_seconds']!=900 or p['max_calls']!=950 or p['max_output_bytes']!=3*1024**3
        or p['attempts']!=576 or p['epochs']!=4 or p['cost']!=816 or p['hidden']!=128
        or p['binder_sha']!=BINDER_SHA):raise ValueError('actual V60 scope')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new fixed semantic clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    data=Path(p['data']);train=io.load_json(data/'train_features.json');dev=io.load_json(data/'dev_features.json');orders=io.load_json(data/'orders.json')
    if len(train)!=2880 or len(dev)!=256 or len(orders)!=576 or any(set(r)!={'id','prompt'} for r in train+dev):raise ValueError('finite feature contract')
    return p,train,dev,orders

class Budget(BaseBudget):
    def __init__(self,p):
        super().__init__(p);self.calls={'encode':0,'bind':0,'denoise':0}
        self.parent_reference=dict(path=p['source'],sha256=p['source_sha'],codec_path=p['head_source'],codec_sha256=p['head_sha'],
            binder_path=p['binder_source'],binder_sha256=p['binder_sha'],old_binder_Adam='preserved closed; active optimizer fresh after graph change',
            old_source96Adam='preserved unchanged in source parent',old_codec8Adam='preserved unchanged; NOT used by new semantic encoder')
    def charge(self,kind):
        self.check()
        if kind not in ('encode','bind') or sum(self.calls.values())>=950:raise ValueError('semantic fixed model call cap')
        self.calls[kind]+=1

def tensor_bytes(obj):
    if isinstance(obj,torch.Tensor):return obj.numel()*obj.element_size()
    if isinstance(obj,dict):return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj,(list,tuple)):return sum(map(tensor_bytes,obj))
    return 0

def save_tensor(path,payload,budget):
    budget.check(write_bytes=tensor_bytes(payload)+8*1024**2)
    with path.open('xb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    return support.sha(path)

@torch.no_grad()
def source_features(source,rows,engine,label,progress):
    width=max(len(r['prompt'].encode()) for r in rows);parts=[];out=engine.budget.output;directory=out/'cache_shards';directory.mkdir(exist_ok=True)
    engine.phase='extract_'+label
    for start in range(0,len(rows),16):
        if start<engine.extract_cursor[label]:
            saved=engine.extract_shards[label][start//16];parts.append(torch.load(saved['path'],weights_only=True,map_location='cpu'));continue
        selected=rows[start:start+16];engine.budget.charge('encode');x=pack_prompts([r['prompt'] for r in selected],width,'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ep=source.encode_input(x,source_ids=tuple(r['id'] for r in selected),source_versions=(SOURCE_SHA,)*len(selected),model_epoch=SOURCE_SHA)
        exact=ep.memories.exact;keys=torch.zeros(len(x),width,128);values=torch.zeros(len(x),width,256)
        starts=torch.zeros(len(x),width,dtype=torch.bool);ends=starts.clone()
        for b,spans in enumerate(exact.spans):
            k=exact.keys[b].float().cpu();v=exact.values[b].float().cpu()
            for j,(a,z) in enumerate(spans):keys[b,a:z]=k[j];values[b,a:z]=v[j];starts[b,a]=True;ends[b,z-1]=True
        piece=dict(mu=ep.diagnostics['token_memberships'].float().cpu(),contextual_values=ep.diagnostics['token_values'].float().cpu(),
            exact_keys=keys,exact_values=values,valid=(x!=256).cpu(),starts=starts,ends=ends,
            source_revisions=[(s.source_id,s.version,s.sha256) for s in exact.sources],source_epoch=SOURCE_SHA)
        path=directory/f'{label}_{start:04d}.pt';digest=save_tensor(path,piece,engine.budget)
        engine.cache_refs[str(path)]=digest;engine.extract_shards[label].append(dict(path=str(path),sha256=digest))
        engine.extract_cursor[label]=start+len(selected);parts.append(piece)
        io.atomic_json(out/'extraction_journal.json',dict(phase=engine.phase,cursor=engine.extract_cursor,shards=engine.extract_shards,calls=engine.budget.calls))
        progress('extract_'+label,completed=start+len(selected),total=len(rows))
    packed={key:torch.cat([r[key] for r in parts]) for key in ('mu','contextual_values','exact_keys','exact_values','valid','starts','ends')}
    packed.update(source_revisions=sum([r['source_revisions'] for r in parts],[]),source_epoch=SOURCE_SHA)
    return packed

def target_tensors(rows):
    ids={key:i for i,key in enumerate(sorted({r['meaning_id'] for r in rows}))}
    return dict(spans=torch.tensor([r['spans'] for r in rows]),relation=torch.tensor([r['meaning']['relation'] for r in rows]),
        polarity=torch.tensor([r['meaning']['polarity'] for r in rows]),quantity=torch.tensor([r['meaning']['quantity'] for r in rows]),
        meaning_ids=torch.tensor([ids[r['meaning_id']] for r in rows]))

def score_predictions(payload,rows,targets):
    correct=role_ok=present_ok=null_ok=present_n=null_n=0;examples=[]
    spans=payload['spans'].tolist();relations=payload['relation_probabilities'].argmax(-1).tolist()
    polarities=payload['polarity_probabilities'].argmax(-1).tolist();quantities=payload['quantity_probabilities'].argmax(-1).tolist()
    factor_counts=[0,0,0]
    for i,(row,target) in enumerate(zip(rows,targets)):
        raw=row['prompt'].encode();literal=[]
        for a,b in spans[i]:
            try:literal.append(None if a<0 else raw[a:b].decode())
            except UnicodeDecodeError:literal.append('<invalid UTF8>')
        truth=target['meaning'];pred=dict(relation=relations[i],polarity=polarities[i],quantity=quantities[i],roles=dict(zip(('primary','recipient','theme'),literal)))
        correct+=int(pred==truth)
        for r,key in enumerate(('primary','recipient','theme')):
            ok=literal[r]==truth['roles'][key];role_ok+=int(ok)
            if truth['roles'][key] is None:null_n+=1;null_ok+=int(ok)
            else:present_n+=1;present_ok+=int(ok)
        for j,key in enumerate(('relation','polarity','quantity')):factor_counts[j]+=int(pred[key]==truth[key])
        examples.append(dict(id=row['id'],source=row['prompt'],predicted=pred,gold=truth,exact=pred==truth,predicted_spans=spans[i]))
    return dict(semantic_fact_exact=correct/len(rows),correct_facts=correct,rows=len(rows),role_literal_accuracy=role_ok/(3*len(rows)),
        present_role_literal_accuracy=present_ok/max(1,present_n),NULL_role_accuracy=null_ok/max(1,null_n),
        relation_accuracy=factor_counts[0]/len(rows),polarity_accuracy=factor_counts[1]/len(rows),quantity_accuracy=factor_counts[2]/len(rows)),examples

@torch.no_grad()
def evaluate(engine,features,rows,targets,condition,indices=None,disabled=()):
    engine.model.eval();parts=[];indices=list(range(len(rows))) if indices is None else indices
    for start in range(0,len(indices),16):
        ids=torch.tensor(indices[start:start+16]);packet=select_packet(features,ids,'cuda');engine.budget.charge('bind')
        with torch.autocast('cuda',dtype=torch.bfloat16):commit,raw=engine.model(packet,engine.plan_sha+':'+condition,disabled)
        values=dict(role_N=commit.role_N,role_D=commit.role_D,role_values=commit.role_values,spans=commit.predicted_spans,
            relation_probabilities=commit.relation_probabilities,polarity_probabilities=commit.polarity_probabilities,
            quantity_probabilities=commit.quantity_probabilities,semantic_vector=commit.semantic_vector(),**raw)
        parts.append({k:v.detach().cpu() for k,v in values.items()})
    payload={k:torch.cat([r[k] for r in parts]) for k in parts[0]}
    selected_rows=[rows[i] for i in indices];selected_targets=[targets[i] for i in indices]
    metrics,examples=score_predictions(payload,selected_rows,selected_targets)
    file=engine.budget.output/('raw_'+condition+'.pt');engine.cache_refs[str(file)]=save_tensor(file,payload,engine.budget)
    io.atomic_json(engine.budget.output/(condition+'_predictions.json'),examples);io.atomic_json(engine.budget.output/(condition+'_metrics.json'),metrics)
    return metrics

@torch.no_grad()
def interchange_events(engine,features,targets):
    raw=torch.load(engine.budget.output/'raw_final.pt',weights_only=True,map_location='cpu')
    groups=io.load_json(Path(engine.budget.p['data'])/'dev_groups.json');examples=[]
    for kind in (1,2):
        selected=[g for g in groups if targets[g[0]]['meaning']['relation']==kind][:8]
        for group in selected:
            a,b=group[0],group[3];values=raw['role_values'][a:a+1].clone();values[:,3]=raw['role_values'][b,3]
            numerator=raw['role_N'][a:a+1].clone();denominator=raw['role_D'][a:a+1].clone()
            numerator[:,3]=raw['role_N'][b,3];denominator[:,3]=raw['role_D'][b,3]
            state=BoundMeaningState(numerator.cuda(),denominator.cuda(),values.cuda(),raw['spans'][a:a+1].cuda(),
                (features['source_revisions'][a],),SOURCE_SHA,engine.plan_sha+':interchange',
                donor_revisions=((3,features['source_revisions'][b]),))
            engine.budget.charge('bind')
            with torch.autocast('cuda',dtype=torch.bfloat16):commit,factors=engine.model.read_state(state,diagnostic=True)
            examples.append(dict(source_id=targets[a]['id'],donor_id=targets[b]['id'],axis='polarity' if kind==1 else 'quantity',
                source_fact=targets[a]['meaning'],donor_fact=targets[b]['meaning'],diagnostic_only=True,
                predicted_relation=int(commit.relation_probabilities[0].argmax()),predicted_polarity=int(commit.polarity_probabilities[0].argmax()),
                predicted_quantity=int(commit.quantity_probabilities[0].argmax()),
                factor_logits={k:v.float().cpu().tolist() for k,v in factors.items()}))
    io.atomic_json(engine.budget.output/'event_interchange.json',examples)
    return dict(cases=len(examples),diagnostic_only=True)

def run(path,output,preflight=False):
    p,train,dev,orders=validate(path,output);torch.set_num_threads(4);budget=Budget(p);device='cpu' if preflight else 'cuda'
    source,report=load_parent(p,device)
    torch.manual_seed(60111);random.seed(60111);np.random.seed(60111)
    if device=='cuda':torch.cuda.manual_seed_all(60111)
    model=SemanticBinderV60(source.core.prior.detach().cpu(),SOURCE_SHA,hidden=128)
    warm=torch.load(p['binder_source'],weights_only=True,map_location='cpu')
    if support.state_sha(warm)!=BINDER_ALGORITHM or warm['updates']!=576:raise ValueError('complete V59 warm parent')
    copied=model.load_state_dict(warm['binder'],strict=False)
    if set(copied.missing_keys)!={'role_start_gain','role_end_gain'} or copied.unexpected_keys:raise ValueError('only6 new context gains')
    if any(not torch.equal(model.state_dict()[k].cpu(),v) for k,v in warm['binder'].items()):raise ValueError('old binder tensor parity')
    if model.role_start_gain.count_nonzero() or model.role_end_gain.count_nonzero():raise ValueError('new gains must start0')
    engine=SemanticEngine(model,None,None,orders,budget,support.sha(path),device)
    report.update(new_semantic_parameters=sum(v.numel() for v in model.parameters()),new_optimizer_fresh=True,old_surface_codec_forward=0,
        all_V59_binder_tensors_exact=True,new_parameters6_zero=True,graph_changed_no_output_parity_claim=True,
        engine_schema=SCHEMA,graph_contract=engine.export()['graph_contract'])
    if preflight:
        state=engine.export();digest=support.state_sha(state);stream=memoryio.BytesIO();torch.save(state,stream);stream.seek(0)
        engine.restore(torch.load(stream,weights_only=True,map_location='cpu'))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('fullsize semantic fresh-state restore')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,CUDA_initialized=False,actual_parent_forward=0)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('one newly sealed semantic trial')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));latest=None;began=time.time();guard=False
    def progress(phase,**extra):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,attempts=engine.attempts,updates=engine.updates,
            cursor=engine.cursor,calls=budget.calls,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest,**extra))
    def save(name,emergency=False):
        nonlocal latest
        budget.check(write_bytes=40*1024**2,emergency=emergency);directory=output/name;directory.mkdir(exist_ok=True)
        latest=dict(support.save_full(directory,engine.export()),directory=name,standalone=False,parent_required=SOURCE_SHA)
        io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('before_source_extraction');save('seed');caches={};labels={}
        for label,rows in (('train',train),('dev',dev)):
            if label=='dev':
                f=torch.load(p['reused_dev_features'],weights_only=True,map_location='cpu')
                t=torch.load(p['reused_dev_targets'],weights_only=True,map_location='cpu')
                if f['source_epoch']!=SOURCE_SHA or [r[0] for r in f['source_revisions']]!=[r['id'] for r in rows]:
                    raise ValueError('unchanged held source-memory cache')
                for name in ('reused_dev_features','reused_dev_targets'):engine.cache_refs[p[name]]=p['files'][p[name]]
                engine.extract_cursor['dev']=len(rows);engine.extract_shards['dev']=[dict(path=p['reused_dev_features'],reused=True)]
                caches[label]=f;labels[label]=t;continue
            f=source_features(source,rows,engine,label,progress);t=target_tensors(io.load_json(Path(p['data'])/(label+'_targets.json')))
            for suffix,value in (('features',f),('targets',t)):
                file=output/(label+'_'+suffix+'.pt');engine.cache_refs[str(file)]=save_tensor(file,value,budget)
            caches[label]=f;labels[label]=t
        if support.state_sha(source.state_dict())!=report['parent_model_digest']:raise ValueError('frozen source tensors changed')
        io.atomic_json(output/'frozen_source_review.json',dict(source_unchanged=True,source_optimizer_updates=0,source_model_digest=report['parent_model_digest']))
        del source;gc.collect();torch.cuda.empty_cache()
        engine.features,engine.targets=caches['train'],labels['train'];progress('initial_semantic_commit');save('initial')
        dev_targets=io.load_json(Path(p['data'])/'dev_targets.json');progress('baseline_semantic_extraction')
        baseline=evaluate(engine,caches['dev'],dev,dev_targets,'baseline')
        progress('semantic_binding_training')
        while engine.attempts<len(orders):
            if time.time()+60>=p['cutoff']:raise TimeoutError('final semantic assessment reserve')
            engine.step()
            if engine.attempts in (1,288):save('model')
            if engine.attempts%8==0:progress('semantic_binding_training')
        progress('semantic_binding_assessment');final=evaluate(engine,caches['dev'],dev,dev_targets,'final')
        controls={name:evaluate(engine,caches['dev'],dev,dev_targets,'read_off_'+name,list(range(64)),(name,)) for name in ('persistent','contextual','exact')}
        interchanges=interchange_events(engine,caches['dev'],dev_targets)
        if support.state_sha({'prior':engine.model.persistent_prior})!=engine.prior_sha:raise ValueError('frozen K_P prior changed')
        progress('final_semantic_commit');save('model')
        io.atomic_json(output/'result.json',dict(status='SEMANTIC_BINDING_V60_COMPLETE',attempts=engine.attempts,updates=engine.updates,
            AMP_skips=engine.skips,exposure=engine.exposure,calls=budget.calls,baseline=baseline,final=final,read_channel_controls=controls,event_interchanges=interchanges,
            latest=latest,seconds=time.time()-began,source_optimizer_updates=0,language_decoder_updates=0,denoise_calls=0,source_copy_bytes=0,
            limitation='Memory-only fact readout in restricted give/open/contain curriculum; adaptive expanded TRAIN forms, old heldDEV unchanged. Not integrated with language/denoiser or full OWT semantics.'))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True);io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        try:save('emergency',True)
        except BaseException as failure:io.atomic_json(output/'save_error.json',dict(error=repr(failure)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args();raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
