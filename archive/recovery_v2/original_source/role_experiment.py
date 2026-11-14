"""Explicit, guarded v0.3.1 -> v0.3.2 weights migration and matched role experiment."""
import argparse,gc,json,pathlib
from dataclasses import asdict
import torch
from model import Config
from data import make_split,manifest
from train import Trainer,Plan,digest_file,evaluate,write_json,source_digest

PARENT_SOURCE='11e77fa0dd7b2cc9339f19e53378a77f775a6d9efcb8a608aa22ee9beb1e9a8a'

def new_test_sets():
    seen=set()
    for args in [(128,301,False,(4,6)),(64,20001,True,(4,6)),(64,40001,False,(7,8))]:
        for x in make_split('test',args[0],seed=args[1],heldout=args[2],n_range=args[3]):seen.add(x['graph_key'])
    out={}
    for name,count,seed,heldout,size in [('new_graph_test',128,60001,False,(4,6)),('new_heldout_phrase_test',64,90001,True,(4,6)),('new_larger_graph_test',64,120001,False,(7,8))]:
        candidates=make_split('test',count+len(seen),seed=seed,heldout=heldout,n_range=size)
        selected=[]
        for x in candidates:
            if x['graph_key'] in seen:continue
            selected.append(x);seen.add(x['graph_key'])
            if len(selected)==count:break
        assert len(selected)==count
        out[name]=selected
    return out

def migrate(out,parent,role_condition,steps=640):
    parent=pathlib.Path(parent);meta=json.loads(parent.with_suffix('.json').read_text())
    if parent.name!=meta['file'] or digest_file(parent)!=meta['sha256']:raise ValueError('parent runtime checksum mismatch')
    saved=torch.load(parent,map_location='cpu',weights_only=True)
    if saved['contract']['source']!=PARENT_SOURCE or not saved['training_complete']:raise ValueError('unapproved or incomplete parent')
    if saved['updates']!=1440 or not all(g['passed'] for g in saved['gates']):raise ValueError('parent training contract mismatch')
    cfg=Config(diffusion_role_condition=role_condition)
    if saved['config']!={k:v for k,v in asdict(cfg).items() if k!='diffusion_role_condition'}:raise ValueError('unexpected architecture migration')
    plan=Plan(phase_steps=(0,0,steps))
    trainer=Trainer(out,plan,cfg)
    if saved['contract']['data']!=trainer.contract['data']:raise ValueError('parent dataset mismatch')
    trainer.model.load_state_dict(saved['model'],strict=True)
    trainer.phase=2;trainer._optimizer()
    trainer.parent={'file':parent.name,'sha256':meta['sha256'],'source':PARENT_SOURCE,'updates':saved['updates'],
                    'migration':'same tensors; direct per-slot query role conditioning' if role_condition else 'matched no-role control',
                    'optimizer_reset':True,'exact_resume':False}
    write_json(trainer.out/'warm_start.json',trainer.parent)
    del saved;gc.collect()
    checked=evaluate(trainer.model,trainer.dev,'cuda',4,full=False)
    trainer.gates=[{'phase':'grounding','measured':checked['ground_accuracy'],'required':.9,'passed':checked['ground_accuracy']>=.9},
                   {'phase':'semantic','measured':checked['initializer']['accuracy'],'required':.8,'passed':checked['initializer']['accuracy']>=.8}]
    if not all(g['passed'] for g in trainer.gates):raise ValueError('parent semantics failed recheck')
    write_json(trainer.out/'parent_semantics_rechecked.json',checked)
    trainer.save()
    return trainer

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--parent',required=True)
    p.add_argument('--role-condition',type=int,choices=(0,1),required=True);p.add_argument('--steps',type=int,default=640);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.resume:trainer=Trainer(a.out,Plan(phase_steps=(0,0,a.steps)),Config(diffusion_role_condition=bool(a.role_condition)),resume=True)
    else:trainer=migrate(a.out,a.parent,bool(a.role_condition),a.steps)
    print('ROLE_EXPERIMENT='+json.dumps({'role_condition':bool(a.role_condition),'source':source_digest(),'plan':asdict(trainer.plan),'parent':trainer.parent}),flush=True)
    status=trainer.run(wall_seconds=3600)
    if status['phase']=='complete':
        tests=new_test_sets();write_json(trainer.out/'new_test_manifest.json',manifest(tests))
        result={name:evaluate(trainer.model,items,'cuda',4,(17,29,43)) for name,items in tests.items()}
        write_json(trainer.out/'new_final_evaluation.json',result)
        print('NEW_FINAL_EVALUATION='+json.dumps(result,ensure_ascii=False),flush=True)
    print('ROLE_FINAL_STATUS='+json.dumps({k:v for k,v in status.items() if k not in ('history','gradient_audit')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
