"""Separate source-condition caches and answer-loss caches; frozen codec only."""
import torch
from research.memory_thought_diffusion_v94 import InputMeaning

STATIC=('anchors','prior','exact_keys','exact_values','exact_valid')
DYNAMIC=('source','contextual_long','contextual_recent')


def split_input(meaning,source_id,source_SHA):
    if meaning.exact_valid.any() or meaning.exact_keys.count_nonzero() or meaning.exact_values.count_nonzero():raise ValueError('Exact must remain NULL')
    static={k:getattr(meaning,k).detach().cpu().clone() for k in STATIC}
    dynamic={k:getattr(meaning,k).detach().cpu().clone() for k in DYNAMIC}
    return dict(schema='thought_static_memories_v95',model_epoch=meaning.model_epoch,**static),dict(
        schema='thought_input_only_v95',source_id=source_id,source_SHA=source_SHA,model_epoch=meaning.model_epoch,
        memory_revisions=meaning.memory_revisions,**dynamic)


def assemble_input(static,dynamic,device='cpu'):
    if set(static)!={'schema','model_epoch',*STATIC} or set(dynamic)!={'schema','source_id','source_SHA','model_epoch','memory_revisions',*DYNAMIC}:
        raise ValueError('strict SOURCE-only cache fields; no answer/target/roles')
    if static['schema']!='thought_static_memories_v95' or dynamic['schema']!='thought_input_only_v95' or static['model_epoch']!=dynamic['model_epoch']:
        raise ValueError('matching frozen memory epoch')
    values={k:(static if k in STATIC else dynamic)[k].detach().to(device).clone() for k in (*STATIC,*DYNAMIC)}
    if values['exact_valid'].any() or values['exact_values'].count_nonzero() or values['exact_keys'].count_nonzero():raise ValueError('Exact dormant cache')
    return InputMeaning(**values,model_epoch=static['model_epoch'],memory_revisions=tuple(dynamic['memory_revisions']))
