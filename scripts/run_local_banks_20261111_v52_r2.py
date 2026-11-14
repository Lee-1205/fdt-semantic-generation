"""One bounded variable-length, three-memory same-input codec training run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,ctypes,gc,io as memoryio,json,random,shutil,sys,time,traceback
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.local_three_bank_codec_v52 import LocalThreeBankCodec as TrainableLongCodec
from research.text_diffusion_v1 import pack_prompts,pack_answers
from local_bank_engine_v52 import BankEngine as LongEngine
import text_generation_support_v1 as support
import learning_io_v2 as io
from owned_output_size_v52 import owned_output_bytes

SOURCE_SHA='695a5213d21212c3993d3411cdb51f4cb14d9b64eca8013ad87f8ebe116e8bed'
def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_local_three_bank_training_v52' or p['output']!=str(output.resolve())
        or p['capacity']!=16384 or p['attempts']!=1728 or p['epochs']!=4 or p['amp_dtype']!='bfloat16'
        or p['wall_seconds']!=2100 or p['max_calls']!=140000 or p['max_output_bytes']!=3*1024**3
        or p['source_sha']!=SOURCE_SHA or p['raw_indices']!=[0,4,8,12] or p['raw_max_symbols']!=16385):
        raise ValueError('actual long worker plan contract')
    if p['authorized'] and (p['deadline']-p['start']!=2100 or p['cutoff']!=p['deadline']-180):
        raise ValueError('fixed training clock')
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('changed pin '+name)
    data=Path(p['data']);train=io.load_json(data/'train.json');dev=io.load_json(data/'dev.json');orders=io.load_json(data/'orders.json')
    if len(train)!=432 or len(dev)!=16 or len(orders)!=1728:raise ValueError('actual finite pool')
    for epoch in range(4):
        if sorted(i for b in orders[epoch*432:(epoch+1)*432] for i in b)!=list(range(432)):
            raise ValueError('one exposure per epoch')
    return p,train,dev,orders

class Budget:
    def __init__(self,p):self.p=p;self.output=Path(p['output']);self.data_sha=support.sha(Path(p['data'])/'manifest.json');self.calls=dict(encode=0,decoder=0,denoise=0)
    def check(self,write_bytes=0,emergency=False):
        if not emergency and self.p['authorized'] and ((self.output/'STOP').exists() or time.time()>=self.p['cutoff']):raise InterruptedError('fixed cutoff/STOP')
        if shutil.disk_usage(ROOT).free<101*1024**3+write_bytes:raise OSError('C100GiB protection and write reserve')
        if self.output.exists() and owned_output_bytes(self.output)+write_bytes>self.p['max_output_bytes']:
            raise OSError('fixed output cap')
    def charge(self,kind):
        self.check()
        if kind=='denoise' or sum(self.calls.values())>=self.p['max_calls']:raise ValueError('model call contract')
        self.calls[kind]+=1

def initialize(p,rows,orders,budget,plan_sha,device):
    source=torch.load(p['source'],map_location='cpu',weights_only=True)
    if source['schema']!='fdt_long_three_memory_codec_v50_fullstate' or source['attempts']!=1728 or source['updates']!=1728:
        raise ValueError('complete V47 source')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(50050)
        model=TrainableLongCodec(replace(Config(**source['core']),gradient_checkpointing=True),input_capacity=p['capacity'])
        model.enable_position_adapter();model.enable_long_capacity()
        model.load_state_dict(source['model'],strict=True);model.enable_local_banks()
    engine=LongEngine(model,source['optimizer_parameter_names'],rows,orders,budget,plan_sha,device)
    engine.import_parent(source)
    for name,value in source['model'].items():
        if not torch.equal(value,model.state_dict()[name].cpu()):raise ValueError('parent tensor changed '+name)
    if engine.frozen_sha!=source['frozen_sha']:raise ValueError('frozen source identity')
    report=dict(old_populated_Adam_states=len(source['optimizer']['state']),new_parameters=sum(p.numel() for n,p in model.named_parameters() if n in engine.names[-1]),
        all_parent_tensors_exact=True,all_old_Adam_exact=True,source_scaler_exact=engine.scaler.state_dict()==source['scaler'],
        total_parameters=sum(p.numel() for p in model.parameters()),capacity=p['capacity'],CUDA_initialized=torch.cuda.is_initialized())
    return engine,report

@torch.no_grad()
def assess(engine,dev,label,output,raw_indices):
    engine.model.eval();sums=correct=count=0;records=[]
    for i,row in enumerate(dev):
        text=row['prompt'];length=len(text.encode())
        x=pack_prompts([text],length,engine.device);y=pack_answers([text],length+1,engine.device)
        engine.budget.charge('encode')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            episode=engine.model.encode_input(x,source_ids=(row['id'],),source_versions=(engine.budget.data_sha,),model_epoch='eval:'+label)
            engine.budget.charge('decoder');logits=engine.model.teacher_logits(episode.commit,y)
        loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),y,reduction='sum')
        pred=logits.argmax(-1);sums+=float(loss);correct+=int((pred==y).sum());count+=y.numel()
        record=dict(id=row['id'],bytes=length,NLL=float(loss)/y.numel(),teacher_accuracy=float((pred==y).float().mean()),
                    exact=bool(torch.equal(pred,y)),teacher_predicted_ids=pred.cpu().tolist()[0],target_ids=y.cpu().tolist()[0])
        if i in raw_indices:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output_ids=engine.model.reconstruct(episode.commit,charge=engine.budget.charge,max_symbols=16385)[0].cpu().tolist()
            raw=bytes(k for k in output_ids[:output_ids.index(256) if 256 in output_ids else len(output_ids)] if 0<=k<256)
            try:decoded=raw.decode('utf-8');utf8=True
            except UnicodeDecodeError:decoded=raw.decode('utf-8',errors='replace');utf8=False
            record.update(source=text,output=decoded,UTF8=utf8,EOS=256 in output_ids,raw_exact=decoded==text,output_ids=output_ids)
        records.append(record)
        io.atomic_json(output/(label+'_assessment.json'),dict(rows=records,teacher_NLL=sums/count,teacher_accuracy=correct/count,symbols=count))
        del episode,x,y,logits,pred
    return dict(teacher_NLL=sums/count,teacher_accuracy=correct/count,teacher_exact=sum(r['exact'] for r in records),
        raw_exact=sum(r.get('raw_exact',False) for r in records),raw_count=len(raw_indices),symbols=count)

def run(path,output,preflight=False):
    p,rows,dev,orders=validate(path,output);torch.set_num_threads(4)
    budget=Budget(p);engine,report=initialize(p,rows,orders,budget,support.sha(path),'cpu' if preflight else 'cuda')
    if preflight:
        state=engine.export();digest=support.state_sha(state)
        stream=memoryio.BytesIO();torch.save(state,stream);stream.seek(0)
        restored=torch.load(stream,map_location='cpu',weights_only=True)
        if support.state_sha(restored)!=digest:raise ValueError('actual fullsize safeload roundtrip')
        engine.restore(restored)
        if support.state_sha(engine.export())!=digest:raise ValueError('actual engine full restore')
        if torch.cuda.is_initialized():raise ValueError('CPU initialized CUDA')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,actual_parent_forward=0)))
        print('CPU_PREFLIGHT_PASS')
        return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('one sealed invocation')
    began=time.time();io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),start=began))
    guard=False;latest=None
    def progress(phase):
        io.atomic_json(output/'progress.json',dict(phase=phase,attempts=engine.attempts,updates=engine.updates,cursor=engine.cursor,
            native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time(),calls=budget.calls,latest=latest))
    def save(kind,emergency=False):
        nonlocal latest
        engine.verify_frozen();budget.check(write_bytes=1100*1024**2,emergency=emergency)
        directory=output/kind;directory.mkdir()
        latest=dict(support.save_full(directory,engine.export()),directory=kind)
        io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        io.atomic_json(output/'import_review.json',report);progress('initial_commit');save('initial')
        progress('baseline_saved_reference');baseline=io.load_json(Path(p['baseline_reference']))['final']
        progress('long_codec_training')
        while engine.attempts<p['attempts']:
            if time.time()+600>=p['cutoff']:raise TimeoutError('reserve final same-input assessment')
            engine.step();progress('long_codec_training')
        progress('final_same_input');final=assess(engine,dev,'final',output,p['raw_indices'])
        progress('three_memory_ablations')
        ablations=[]
        normal_rows=io.load_json(output/'final_assessment.json')['rows']
        engine.model.eval()
        with torch.no_grad():
            for i in range(0,16,2):
                row=dev[i];text=row['prompt'];length=len(text.encode())
                x=pack_prompts([text],length,'cuda');y=pack_answers([text],length+1,'cuda');scores={}
                for disabled in ('persistent','contextual','exact'):
                    budget.charge('encode')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        episode=engine.model.encode_input(x,source_ids=(row['id'],),source_versions=(budget.data_sha,),model_epoch='bank_ablation',disabled_memories=(disabled,))
                        budget.charge('decoder');logits=engine.model.teacher_logits(episode.commit,y)
                    scores[disabled]=float(torch.nn.functional.cross_entropy(logits.transpose(1,2),y))
                ablations.append(dict(index=i,bytes=length,normal_NLL=normal_rows[i]['NLL'],disabled_NLL=scores))
        io.atomic_json(output/'memory_ablations.json',ablations)
        progress('final_commit');save('model')
        if support.sha(p['source'])!=SOURCE_SHA:raise ValueError('parent modified')
        io.atomic_json(output/'result.json',dict(status='LOCAL_THREE_BANK_V52_COMPLETE',attempts=engine.attempts,updates=engine.updates,
            AMP_skips=engine.skips,capacity=16384,baseline=baseline,final=final,calls=budget.calls,
            exposure=engine.exposure,seconds=time.time()-began,latest=latest,denoise_calls=0,copy_bytes=0,
            interpretation='Length extension and codec training; not a losslessness or semantic-completion proof.'))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True)
        io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        try:save('emergency',True)
        except BaseException as save_error:io.atomic_json(output/'save_error.json',dict(error=repr(save_error)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
