"""Preserved adapter objectives with pre-verifier neural evaluation."""
from dataclasses import asdict,dataclass
import argparse,gc,hashlib,json,math,os,pathlib,time
import torch
from torch.nn import functional as F
from .model import Config,FDT100
from .data import VERSION,LABELS,collate,make_split,manifest

PHASES=('grounding','semantic','diffusion')


@dataclass(frozen=True)
class Plan:
    seed:int=19
    phase_steps:tuple=(480,320,640)
    micro_batch:int=4
    accumulation:int=2
    save_every:int=100
    log_every:int=25
    train_graphs:int=4096
    dev_graphs:int=96
    test_graphs:int=128
    eval_seed:int=17
    encoder_lr:float=3e-5
    ground_lr:float=1e-3
    semantic_lr:float=3e-4
    diffusion_lr:float=3e-4
    ground_gate:float=.9
    semantic_gate:float=.8


def digest_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_digest():
    root=pathlib.Path(__file__).parent
    h=hashlib.sha256()
    for name in ('model.py','data.py','train.py','runtime.py'):
        h.update(name.encode());h.update((root/name).read_bytes())
    return h.hexdigest()


def write_json(path,value):
    p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,p)


def masked_ce(logits,labels,mask,weights=None):
    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.flatten(),weight=weights,reduction='none').reshape_as(labels)
    denominator=mask.sum() if weights is None else (mask*weights[labels]).sum()
    return (loss*mask).sum()/denominator.clamp_min(1)


def masked_mse(a,b,mask):
    return ((a.float()-b.float()).square()*mask[...,None]).sum()/(mask.sum()*a.shape[-1]).clamp_min(1)


def diffusion_loss(model,enc,target,labels,mask,generator,weights=None):
    b=len(target)
    t=torch.rand(b,device=target.device,generator=generator)
    t[:max(1,b//2)]=1.
    noise=torch.randn(target.shape,device=target.device,generator=generator)*mask[...,None]
    a=torch.cos(t*math.pi/2)[:,None,None];s=torch.sin(t*math.pi/2)[:,None,None]
    # Exact t=1 boundary, including FP32, avoids tiny signed cosine residue.
    a=torch.where(t[:,None,None]==1,torch.zeros_like(a),a)
    zt=a*target+s*noise;v_target=a*noise-s*target
    v=model.denoise(zt,t,enc).float();clean=a*zt-s*v
    lv=masked_mse(v,v_target,mask);lz=masked_mse(clean,target,mask)
    ce=masked_ce(model.readout(clean),labels,mask,weights)
    return lv+.25*lz+.5*ce,{'v_mse':lv.detach(),'clean_mse':lz.detach(),'diff_ce':ce.detach()}


def compute_metrics(pred,labels,mask,hops):
    valid=mask.bool();p=pred[valid].cpu();y=labels[valid].cpu();h=hops[valid].cpu()
    conf=torch.zeros(4,4,dtype=torch.int64)
    for a,b in zip(y.tolist(),p.tolist()):conf[a,b]+=1
    f1=[]
    for i in range(4):
        tp=conf[i,i].item();den=conf[i].sum().item()+conf[:,i].sum().item()
        f1.append(2*tp/den if den else 0.)
    result={'queries':len(y),'accuracy':float((p==y).float().mean()),'macro_f1':sum(f1)/4,
            'confusion':conf.tolist(),'class_f1':dict(zip(LABELS,f1))}
    for label,select in [('direct',(h==1)),('transitive',(h>=2)),('long_chain',(h>=3)),('unknown',(y==0)),('conflict',(y==3))]:
        result[label]={'count':int(select.sum()),'accuracy':float((p[select]==y[select]).float().mean()) if select.any() else None}
    return result


@torch.no_grad()
def evaluate(model,items,device,batch_size=4,seeds=(17,),full=True):
    model.eval();c=model.c
    predictions={k:[] for k in ('initializer','diffusion','fuzzy_removed','corrupted_initializer','repaired_latent')}
    labels=[];masks=[];hops=[];mse_before=[];mse_after=[];ground_correct=ground_total=0
    for start in range(0,len(items),batch_size):
        feat,gold,_=collate(items[start:start+batch_size],c,device)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            enc=model.encode(feat);oracle=model.target(feat,gold['labels'])
            ground_pred=enc['ground'].argmax(-1)
            ground_correct+=int(((ground_pred==gold['ground'])*feat['fact_mask']).sum())
            ground_total+=int(feat['fact_mask'].sum())
            ablated=model.encode(feat,True) if full else None
            for seed in seeds:
                g=torch.Generator(device=device).manual_seed(seed+start)
                noise=torch.randn(enc['init'].shape,device=device,generator=g)
                predictions['initializer'].append(model.readout(enc['init']).argmax(-1).cpu())
                if full:
                    refined=model.rollout(enc,noise)
                    removed=model.rollout(ablated,noise,ablate=True)
                    # Separate paired controlled latent corruption; same input conditions.
                    t=.5;corrupt=math.cos(math.pi*t/2)*enc['init'].float()+math.sin(math.pi*t/2)*noise
                    repaired=model.rollout(enc,corrupt,(.5,.375,.25,.125,0.))
                    for key,z in [('diffusion',refined),('fuzzy_removed',removed),('corrupted_initializer',corrupt),('repaired_latent',repaired)]:
                        predictions[key].append(model.readout(z).argmax(-1).cpu())
                    mse_before.append(float(masked_mse(corrupt,oracle,feat['query_mask'])))
                    mse_after.append(float(masked_mse(repaired,oracle,feat['query_mask'])))
                labels.append(gold['labels'].cpu());masks.append(feat['query_mask'].cpu());hops.append(gold['hops'].cpu())
    y,m,h=map(torch.cat,(labels,masks,hops))
    result={'graphs':len(items),'noise_seeds':list(seeds),'ground_accuracy':ground_correct/max(1,ground_total),
            'neural_before_verifier':True,'queries_are_correlated_within_graph_and_seeds':True}
    for key,values in predictions.items():
        if values:result[key]=compute_metrics(torch.cat(values),y,m,h)
    if mse_before:
        result['controlled_latent_repair_mse']={'before':sum(mse_before)/len(mse_before),'after':sum(mse_after)/len(mse_after)}
    return result


