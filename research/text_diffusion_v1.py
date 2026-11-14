"""Separate byte-text VP diffusion extension. Never changes the typed FDT core.

The copied refiner learns new byte latents; its old relation accuracy is not a
text-generation baseline. No oracle, entity roles, retrieval, or string templates
are used by this model or its sampler.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from fdt_local.model_masked import Config, Core, Language, RMS

EOS, PAD, VOCAB = 256, 257, 258

@dataclass(frozen=True)
class TextConfig:
    prompt_bytes: int = 160
    answer_bytes: int = 40
    steps: int = 16
    codec_seed: int = 282601

def pack_prompts(texts, limit=160, device='cpu'):
    out=torch.full((len(texts),limit),256,dtype=torch.long,device=device)
    for i,text in enumerate(texts):
        raw=text.encode('utf-8')
        if not raw or len(raw)>limit: raise ValueError('prompt byte capacity')
        out[i,:len(raw)]=torch.tensor(list(raw),device=device)
    return out

def pack_answers(texts, limit=40, device='cpu'):
    out=torch.full((len(texts),limit),PAD,dtype=torch.long,device=device)
    for i,text in enumerate(texts):
        raw=text.encode('utf-8')
        if not raw or len(raw)+1>limit: raise ValueError('answer byte capacity')
        out[i,:len(raw)]=torch.tensor(list(raw),device=device)
        out[i,len(raw)]=EOS
    return out

def decode_bytes(ids):
    ids=list(map(int,ids)); ended=EOS in ids
    prefix=ids[:ids.index(EOS)] if ended else ids
    special_before_eos=any(i>=256 for i in prefix)
    raw=bytes(i for i in prefix if 0<=i<256)
    try: text=raw.decode('utf-8',errors='strict'); valid=True
    except UnicodeDecodeError: text=raw.decode('utf-8',errors='replace'); valid=False
    return {'text':text,'raw_hex':raw.hex(),'utf8_valid':valid,
            'eos':ended,'special_before_eos':special_before_eos,
            'well_formed':valid and ended and not special_before_eos,'token_ids':ids}

class TextDiffusion(nn.Module):
    def __init__(self, core=Config(), text=TextConfig()):
        super().__init__(); self.core_config=core; self.text_config=text; d=core.z_dim
        if d<VOCAB: raise ValueError('fixed orthogonal byte codec requires z_dim>=258')
        self.language=Language(core)
        self.prompt_project=nn.Linear(core.lang_dim,d)
        self.positions=nn.Embedding(text.answer_bytes,d)
        self.time=nn.Sequential(nn.Linear(64,d),nn.SiLU(),nn.Linear(d,d,bias=False))
        self.refiner=nn.ModuleList([Core(core) for _ in range(core.diffusion_layers)])
        self.refine_norm=RMS(d); self.velocity=nn.Linear(d,d,bias=False)
        g=torch.Generator().manual_seed(text.codec_seed)
        codec=torch.linalg.qr(torch.randn(d,VOCAB,generator=g),mode='reduced').Q.T
        self.register_buffer('codec',codec)
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),32)))
        self.language.requires_grad_(False)

    def import_typed(self, state):
        selected={}
        for name in ('language','time','refiner','refine_norm','velocity'):
            module=getattr(self,name)
            values={k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')}
            module.load_state_dict(values,strict=True)
            selected[name]=sum(v.numel() for v in values.values())
        self.language.requires_grad_(False)
        return selected

    def encode(self, tokens):
        # All per-byte representations are retained; no entity labels or answers.
        valid=tokens!=256; n=tokens.shape[1]
        with torch.no_grad():
            x=self.language.embedding(tokens)
            pos=torch.arange(n,device=tokens.device)
            band=(pos[:,None]>=pos[None]) & (pos[:,None]-pos[None]<self.core_config.window)
            mask=band[None,None] & (valid[:,None,None,:] | torch.eye(n,device=tokens.device,dtype=torch.bool)[None,None])
            for block in self.language.blocks: x=block(x,mask,valid)
        context=self.prompt_project(x.detach())*valid[...,None]
        context=torch.cat((context,context.new_zeros(len(tokens),1,context.shape[-1])),1)
        mask=torch.cat((valid,torch.ones(len(tokens),1,device=tokens.device,dtype=torch.bool)),1)
        return context,mask

    def target(self, ids): return F.embedding(ids,self.codec)*math.sqrt(self.core_config.z_dim)

    def readout(self, clean):
        with torch.autocast(clean.device.type,enabled=False):
            return 8*(clean.float()@self.codec.float().T)/math.sqrt(self.core_config.z_dim)

    def denoise(self,z,t,enc):
        context,mask=enc
        phase=t.float()[:,None]*self.frequencies[None]
        time=self.time(torch.cat((phase.sin(),phase.cos()),-1))[:,None]
        pos=self.positions(torch.arange(z.shape[1],device=z.device))[None]
        x=z+time+pos
        query_mask=torch.ones(z.shape[:2],device=z.device,dtype=torch.bool)
        for block in self.refiner: x=block(x,context,query_mask,mask)
        return self.velocity(self.refine_norm(x))

    def objective(self, enc, ids, t, noise):
        clean=self.target(ids); angle=t[:,None,None]*math.pi/2
        a,s=angle.cos(),angle.sin(); z=a*clean+s*noise
        target_v=a*noise-s*clean
        v=self.denoise(z,t,enc).float(); predicted=a*z-s*v
        weights=torch.where(ids==PAD,.1,1.).float()
        norm=weights.sum()
        vmse=((v-target_v).square().mean(-1)*weights).sum()/norm
        cmse=((predicted-clean).square().mean(-1)*weights).sum()/norm
        ce=(F.cross_entropy(self.readout(predicted).transpose(1,2),ids,reduction='none')*weights).sum()/norm
        return vmse+.25*cmse+.5*ce, {'velocity_mse':vmse,'clean_mse':cmse,'byte_ce':ce}

    @torch.no_grad()
    def generate(self, tokens, noise, charge=lambda kind:None):
        charge('encode'); enc=self.encode(tokens); z=noise.float()
        schedule=torch.linspace(1,0,self.text_config.steps+1).tolist()
        for t,nxt in zip(schedule,schedule[1:]):
            charge('denoise'); v=self.denoise(z,z.new_full((len(z),),t),enc).float()
            a,s=(0.,1.) if t==1 else (math.cos(t*math.pi/2),math.sin(t*math.pi/2))
            clean=a*z-s*v; eps=s*z+a*v
            z=math.cos(nxt*math.pi/2)*clean+math.sin(nxt*math.pi/2)*eps
            if not torch.isfinite(z).all(): raise FloatingPointError('generation nonfinite')
        return self.readout(z).argmax(-1),z

    def specification(self):
        return {'core':asdict(self.core_config),'text':asdict(self.text_config),
                'parameters':sum(p.numel() for p in self.parameters()),
                'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'frozen_language':True,'external_roles':False,'fixed_closure':False,
                'runtime_answer_templates':False,'output':'raw UTF8 bytes + EOS; invalid output retained'}
