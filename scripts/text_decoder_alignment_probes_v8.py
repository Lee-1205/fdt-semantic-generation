"""Explicit oracle INTERFACE probes, never VP meaning or free generation scores."""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from research.text_diffusion_v1 import pack_prompts,PAD,decode_bytes
from research.text_context_decoder_v2 import BOS
from text_language_engine_v6 import row_noise

CAUSAL_TEXTS=('A is taller than B.','B is taller than A.','C is heavier than D.','D is heavier than C.',
    'E is warmer than F.','F is warmer than E.','G is smaller than H.','H is smaller than G.')

def pack_ids(ys,length,device):
    ids=torch.full((len(ys),length),PAD,device=device,dtype=torch.long)
    for j,y in enumerate(ys):ids[j,:len(y)]=torch.tensor(y,device=device)
    return ids

@torch.no_grad()
def interface_probe(model,features,targets,budget,device):
    model.eval();saved=[];scores={str(s):{'loss_sum':0.,'correct':0,'symbols':0} for s in (0.,.15)}
    for start in range(0,len(features),4):
        rows=features[start:start+4];ys=targets[start:start+4];length=model.policy.target_bytes_with_eos
        ids=pack_ids([y['target_ids'] for y in ys],length,device);active=ids!=PAD
        tokens=pack_prompts([r['prompt'] for r in rows],model.policy.prompt_bytes,device)
        noise=row_noise(rows,length,model.core_config.z_dim,83,device)
        amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
        with amp:
            budget.charge('encode');enc=model.vp.encode(tokens);clean=model.vp.target(ids)
            prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1)
            for sigma in (0.,.15):
                latent=torch.where(active[...,None],clean+sigma*noise,noise)
                budget.charge('decoder');logits=model.decoder(prefix,model.decoder.context(enc[0],enc[1],latent)).float()
                loss=F.cross_entropy(logits.transpose(1,2),ids.masked_fill(~active,-100),ignore_index=-100,reduction='none')
                s=scores[str(sigma)];s['loss_sum']+=float(loss.double().sum());s['correct']+=int((logits.argmax(-1)[active]==ids[active]).sum());s['symbols']+=int(active.sum())
                saved.append({'ids':[r['id'] for r in rows],'sigma':sigma,'logits':logits.cpu(),'target_ids':ids.cpu()})
    for s in scores.values():s.update(NLL=s['loss_sum']/s['symbols'],accuracy=s['correct']/s['symbols'])
    return {'scope':'16 already-observed held development rows, clean/noisy GOLD codec interface only; not VP/semantic performance','scores':scores},saved

@torch.no_grad()
def causal_codec_probe(model,budget,device):
    model.eval();result=[];length=model.policy.target_bytes_with_eos
    for start in range(0,8,4):
        texts=CAUSAL_TEXTS[start:start+4];rows=[{'id':'oracle_state_'+str(start+j)} for j in range(len(texts))]
        ids=pack_ids([list(s.encode())+[256] for s in texts],length,device);active=ids!=PAD
        noise=row_noise(rows,length,model.core_config.z_dim,89,device)
        clean=model.vp.target(ids);latent=torch.where(active[...,None],clean,noise)
        prompt=latent.new_zeros(len(texts),1,latent.shape[-1]);mask=torch.ones(len(texts),1,dtype=torch.bool,device=device)
        amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
        with amp:generated=model.decoder.generate(model.decoder.context(prompt,mask,latent),budget.charge)
        for text,value in zip(texts,generated):
            raw=decode_bytes(value.cpu().tolist());result.append({'expected':text,'generated':raw,'exact':raw['text']==text and raw['eos']})
    return {'scope':'8 authored A-H opposite-order sentence ORACLE-codec controls, raw prompt zero; correct state-to-byte realization only, not inferred semantic capability','rows':result,'exact':sum(r['exact'] for r in result),'E2_final_accessed':False}
