"""Raw-prompt research generation API. Caller owns time/call/process supervision.

No artifact row, query roles, gold, template, regex parser, or retrieval is read.
Returns uncorrected output and learned-factor diagnostics. Not a production API.
"""
from contextlib import nullcontext
import torch
from fdt_local.model_masked import Config
from research.text_diffusion_v1 import TextDiffusion,pack_prompts,decode_bytes
from research.text_semantic_reader_v3 import SemanticGenerator

def load_models(checkpoint,device='cuda'):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('schema')!='fdt_text_semantic_alignment_v3':raise ValueError('wrong research checkpoint')
    core=Config(**payload['core'])
    vp=TextDiffusion(core);vp.load_state_dict(payload['model']['generator'],strict=True)
    text=SemanticGenerator(core);text.load_state_dict(payload['model']['semantic_generator'],strict=True)
    return vp.requires_grad_(False).eval().to(device),text.requires_grad_(False).eval().to(device)

@torch.no_grad()
def generate_text(vp,text,prompt,seed=17,charge=lambda kind:None):
    # Caller must configure deterministic/math SDPA and a separately bounded run.
    from scripts.run_text_context_decoder_20261028 import proposal
    device=next(vp.parameters()).device
    tokens=pack_prompts([prompt],vp.text_config.prompt_bytes,device)
    rng=torch.Generator(device=device).manual_seed(seed)
    noise=torch.randn((1,vp.text_config.answer_bytes,vp.core_config.z_dim),device=device,generator=rng)
    amp=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()
    with amp:
        enc,latent=proposal(vp,tokens,noise,charge)
        if device.type=='cuda':latent=latent.half()
        charge('reader');context,logits=text.context(enc[0],enc[1],latent)
        ids=text.decoder.generate(context,charge)
    return {'prompt':prompt,'seed':seed,'generated':decode_bytes(ids[0].cpu().tolist()),
            'reader_pred':[int(v[0].argmax()) for v in logits],
            'reader_logits':[v[0].float().cpu().tolist() for v in logits],
            'interpretation':'Uncorrected learned hybrid output; semantic accuracy is limited.'}
