"""Discrete routing identity and numerical/cache agreement are separate axes."""
import torch
from fdt_rlm.runtime_contract import runtime_capabilities
from audit_fdt_v4_cache_integrity import local_state_diagnostics, anchor_state_diagnostics


def _error(a,b):
    return float((a.double()-b.double()).abs().max()) if a.numel() else 0.


@torch.inference_mode()
def audit_context(model,context,tolerance=3e-4):
    caps=runtime_capabilities(model.config)
    if caps['incremental_limit'] is None:
        return {'context':context,'status':'UNSUPPORTED','runtime':caps}
    if context+1>caps['incremental_limit']:
        return {'context':context,'status':'UNSUPPORTED_BOUNDARY','runtime':caps}
    device=next(model.parameters()).device
    ids=((torch.arange(context+1,device=device)*65537+29)%model.config.vocab_size).view(1,-1)
    captured={};originals=[]
    for layer,block in enumerate(model.blocks):
        if block.anchor is None:continue
        original=block.anchor.route;originals.append((block.anchor,original))
        def record(x,*args,_original=original,_layer=layer,**kwargs):
            result=_original(x,*args,**kwargs)
            captured[_layer]=(x.detach().cpu().clone(),result[3].detach().cpu().clone(),result[4].detach().cpu().clone())
            return result
        block.anchor.route=record
    try:
        pre,cache=model.prefill(ids[:,:context]);prefix=dict(captured);captured.clear()
        inc,cache=model.decode_step(ids[:,-1:],cache);step=dict(captured);captured.clear()
        full,full_cache=model.prefill(ids);recomputed=dict(captured)
        rows=[];identity=True;first=None
        for layer in sorted(recomputed):
            x,idx,mem=recomputed[layer]
            px,pi,pm=prefix[layer];sx,si,sm=step[layer]
            actual=torch.cat((pi,si),1); mismatch=actual.ne(idx).any(-1)
            loc=mismatch.nonzero()
            if loc.numel():
                token=int(loc[0,1]); identity=False
                point={'layer':layer,'token':token}
                if first is None or (token,layer)<(first['token'],first['layer']):first=point
            rows.append({'layer':layer,'selected_ids_order_mismatches':int(mismatch.sum()),
                'route_input_max_abs_error':_error(torch.cat((px,sx),1),x),
                'membership_max_abs_error':_error(torch.cat((pm,sm),1),mem)})
        local=local_state_diagnostics(cache,full_cache);anchor=anchor_state_diagnostics(cache,full_cache)
        logit_error=_error(inc['logits'][:,-1],full['logits'][:,-1])
        numerical=logit_error<=tolerance and local['max_abs_error']<=tolerance and anchor['raw_max_abs_error']<=tolerance
        route_values=all(r['route_input_max_abs_error']<=tolerance and r['membership_max_abs_error']<=tolerance for r in rows)
        return {'context':context,'status':'PASS' if identity and numerical and route_values else 'FAIL',
            'routing_identity':'PASS' if identity else 'FAIL','first_discrete_divergence':first,
            'numerical_logit_agreement':'PASS' if logit_error<=tolerance else 'FAIL',
            'logit_max_abs_error':logit_error,'route_layers':rows,'local_kv':local,'anchor_state':anchor,
            'tolerance':tolerance,'token_agreement':bool(inc['logits'][:,-1].argmax(-1).eq(full['logits'][:,-1].argmax(-1)).all())}
    finally:
        for anchor,original in originals:anchor.route=original
