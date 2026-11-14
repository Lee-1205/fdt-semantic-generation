"""Readonly semantic-organization question; no model/probe training or denoising."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,ctypes,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout,fuzzy_role_reads
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.codec_commit_stop_v56 import reconstruct_commit_logits
from research.text_diffusion_v1 import pack_prompts
from broad_ordered_engine_v57 import BroadOrderedEngine
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget,load_parent,SOURCE_SHA,save_tensor

HEAD_SHA='a7bbdd7a358bf6beb908a3df0f86c927d89126a1c0d15bc31c54241346b5e2b1'
HEAD_ALGORITHM='4608416d299da2643d2a99766ee4e90982d0119a6deff4056d82c3e2fd9f06c8'

def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_semantic_controls_v58' or p['output']!=str(output.resolve()) or p['source_sha']!=SOURCE_SHA
        or p['head_sha']!=HEAD_SHA or p['wall_seconds']!=180 or p['max_calls']!=240
        or p['max_output_bytes']!=64*1024**2 or p['attempts']!=0 or p['epochs']!=0 or p['cost']!=228):raise ValueError('readonly actual schema')
    if p['authorized'] and (p['deadline']-p['start']!=180 or p['cutoff']!=p['deadline']-45):raise ValueError('new readonly clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    data=Path(p['data']);features=io.load_json(data/'features.json');groups=io.load_json(data/'groups.json')
    if len(features)!=192 or len(groups)!=48 or any(set(r)!={'id','prompt'} for r in features):raise ValueError('no target model feature')
    return p,features,groups

class Budget(BaseBudget):
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout') or sum(self.calls.values())>=240:raise ValueError('fixed readonly call cap')
        self.calls[kind]+=1

def distance(a,b):
    x,y=a.double().flatten(),b.double().flatten()
    return float(1-torch.nn.functional.cosine_similarity(x,y,dim=0,eps=1e-12))

def score_signatures(packets,groups):
    scores={}
    for kind in ('roles','negation','quantity'):
        selected=[g for g in groups if g['kind']==kind];table={}
        for space in packets[selected[0]['ids'][0]]['spaces']:
            rows=[]
            for g in selected:
                a,b,c,d=(packets[r]['spaces'][space] for r in g['ids'])
                para,change,layout=distance(a,b),distance(a,c),distance(a,d)
                def rank(x,y):return .5 if abs(x-y)<1e-8 else float(x<y)
                rows.append(dict(group=g['id'],paraphrase=para,changed=change,layout=layout,
                    paraphrase_closer=rank(para,change),layout_closer=rank(layout,change)))
            table[space]=dict(paraphrase_rank=sum(r['paraphrase_closer'] for r in rows)/len(rows),
                layout_rank=sum(r['layout_closer'] for r in rows)/len(rows),rows=rows)
        scores[kind]=table
    return scores

def load_models(p,device,preflight):
    model,report=load_parent(p,device)
    state=torch.load(p['head_source'],map_location='cpu',weights_only=True)
    if support.state_sha(state)!=HEAD_ALGORITHM or state['updates']!=8192:raise ValueError('complete V57 head')
    head=OrderedFuzzyReadout();head.load_state_dict(state['head'],strict=True)
    if preflight:
        # Restore historical counters/Adam only in CPU RAM to verify provenance.
        # No old clock is reopened, charged or executed.
        old=io.load_json(p['head_plan'])
        reference=SimpleNamespace(p=old,parent_reference=state['parent_reference'],calls={})
        engine=BroadOrderedEngine(head,None,None,state['orders'],reference,state['plan_sha256'],'cpu')
        engine.restore(state)
        if support.state_sha(engine.export())!=HEAD_ALGORITHM or torch.cuda.is_initialized():raise ValueError('CPU historical fullstate restore')
        report.update(fullsize_engine_restore=True,closed_historical_state_executed=False,all8Adam_and_RNG_preserved=True,
            readonly_optimizer_updates=0,CUDA_initialized=False,actual_parent_forward=0)
    head=head.to(device).requires_grad_(False).eval()
    report['head_digest']=support.state_sha(head.state_dict())
    return model,head,report

@torch.no_grad()
def run(path,output,preflight=False):
    p,features,groups=validate(path,output);torch.set_num_threads(4)
    model,head,report=load_models(p,'cpu' if preflight else 'cuda',preflight)
    if preflight:
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('one new readonly launch')
    budget=Budget(p);began=time.time();guard=False;packets={}
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=began))
    def progress(phase,completed):
        io.atomic_json(output/'progress.json',dict(phase=phase,completed=completed,total=192,attempts=0,updates=0,calls=budget.calls,
            time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=None))
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        for i,row in enumerate(features):
            budget.charge('encode');raw=row['prompt'].encode();x=pack_prompts([row['prompt']],len(raw),'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ep=model.encode_input(x,source_ids=(row['id'],),source_versions=(SOURCE_SHA,),model_epoch=SOURCE_SHA)
                role=fuzzy_role_reads(ep.diagnostics['token_memberships'],ep.diagnostics['token_values'],x!=256)
                z=ep.commit.reader_tensor().float();bound=head.bind_commit(ep.commit,role).reader_tensor().float()
            long_values,_=ep.memories.contextual.means()
            spaces=dict(global_three_memory_Commit=z[0,:2].cpu(),contextual_anchor_means=long_values[0].float().cpu(),
                anchor_activation=ep.memories.contextual.long_D[0].float().cpu(),
                mean_ordered_local_Commit=bound[0,2:].mean(0).cpu(),byte_bag_control=torch.bincount(x.flatten(),minlength=256).float().cpu())
            packets[row['id']]=dict(spaces=spaces,committed=bound.cpu())
            if i%8==0:progress('readonly_semantic_encoding',i+1)
        progress('readonly_Commit_branch_interventions',192);interventions=[]
        for group in [g for g in groups if g['kind']=='roles'][:12]:
            a,b=packets[group['ids'][0]]['committed'],packets[group['ids'][2]]['committed']
            if a.shape!=b.shape:raise ValueError('role reversal control shape')
            cases={'normal':a,'swap_global':torch.cat((b[:,:2],a[:,2:]),1),'swap_local':torch.cat((a[:,:2],b[:,2:]),1)}
            for condition,z in cases.items():
                budget.charge('readout')
                commit=DifferentiableCodecCommit.from_semantic(z.cuda(),(('diagnostic','v58','same-byte-multiset'),),'readonlyV58')
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=head.from_commit(commit).float().cpu()
                decoded=reconstruct_commit_logits(logits)
                interventions.append(dict(group=group['id'],condition=condition,output=decoded.pop('text'),**decoded))
        size=sum(t.numel()*t.element_size() for packet in packets.values() for t in list(packet['spaces'].values())+[packet['committed']])
        budget.check(write_bytes=size+8*1024**2)
        save_tensor(output/'raw_semantic_packets.pt',packets,budget)
        io.atomic_json(output/'interventions.json',interventions)
        scores=score_signatures(packets,groups);io.atomic_json(output/'semantic_distances.json',scores)
        if support.state_sha(model.state_dict())!=report['parent_model_digest'] or support.state_sha(head.state_dict())!=report['head_digest']:
            raise ValueError('readonly tensors changed')
        summary={kind:{space:{key:values[key] for key in ('paraphrase_rank','layout_rank')} for space,values in table.items()} for kind,table in scores.items()}
        io.atomic_json(output/'result.json',dict(status='READONLY_SEMANTIC_CONTROLS_V58_COMPLETE',calls=budget.calls,
            optimizer_updates=0,checkpoints=0,denoise_calls=0,source_copy_bytes=0,seconds=time.time()-began,
            frozen_source_head_unchanged=True,summary=summary,
            limitation='48 correlated authored groups,47 duplicate rows; cosine diagnostics and branch swaps do not alone prove or disprove semantic formation.'))
        progress('complete',192);return 0
    except BaseException as error:
        io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()));progress('stopped',len(packets));return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
