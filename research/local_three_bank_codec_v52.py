"""Read all three memory banks at each ordered local Commit slot, not only globally."""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.trainable_long_codec_v50 import TrainableLongCodec
from research.three_memory_codec_v33 import DifferentiableCodecCommit

class LocalThreeBankCodec(TrainableLongCodec):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.memory_codec.register_parameter('local_bank_gains',None)

    def enable_local_banks(self):
        if self.memory_codec.local_bank_gains is not None:raise ValueError('already enabled')
        self.memory_codec.local_bank_gains=nn.Parameter(self.decoder.embedding.weight.new_zeros(2))

    def _exact_index(self,tokens,h,valid,source_ids,source_versions,epoch):
        if any(not isinstance(s,str) or not s for s in (*source_ids,*source_versions)):
            raise ValueError('explicit nonempty source identity and version')
        return super()._exact_index(tokens,h,valid,source_ids,source_versions,epoch)

    def encode_input(self,tokens,*,source_ids,source_versions,model_epoch,disabled_memories=()):
        ep=super().encode_input(tokens,source_ids=source_ids,source_versions=source_versions,
            model_epoch=model_epoch,disabled_memories=disabled_memories)
        gains=self.memory_codec.local_bank_gains
        if gains is None:return ep
        z=ep.commit.reader_tensor();slots=z.shape[1]-2;chunk=self.chunk_bytes
        with torch.autocast(tokens.device.type,enabled=False):
            valid=tokens!=256;padding=slots*chunk-tokens.shape[1]
            mu=ep.diagnostics['token_memberships'].float()
            value=ep.diagnostics['token_values'].float()
            count=F.pad(valid,(0,padding)).reshape(len(z),slots,chunk).sum(-1)
            local_mu=F.pad(mu*valid[...,None],(0,0,0,padding)).reshape(len(z),slots,chunk,-1).sum(2)/count.clamp_min(1)[...,None]
            mean=F.pad(value*valid[...,None],(0,0,0,padding)).reshape(len(z),slots,chunk,-1).sum(2)/count.clamp_min(1)[...,None]
            roles=mean.new_zeros(*mean.shape[:2],16)
            query_z=self.core.condition(torch.cat((mean,roles),-1))
            kp=ep.memories.persistent.read(local_mu)
            # Input-only codepoint boundaries define a local retrieval neighborhood.
            # No target byte, gold span, requested answer or raw H reaches the decoder.
            candidates=[]
            for spans in ep.memories.exact.spans:
                by_slot=[[] for _ in range(slots)]
                for i,(start,end) in enumerate(spans):
                    center=start//chunk
                    for slot in range(max(0,center-1),min(slots,center+2)):by_slot[slot].append(i)
                candidates.append(by_slot)
            width=max(1,max(len(c) for row in candidates for c in row))
            ids=torch.tensor([[items+[0]*(width-len(items)) for items in row] for row in candidates],device=z.device)
            sizes=torch.tensor([[len(items) for items in row] for row in candidates],device=z.device)
            mask=torch.arange(width,device=z.device)[None,None]<sizes[...,None]
            index=ep.memories.exact
            batch=torch.arange(len(z),device=z.device)[:,None,None]
            keys=index.keys[batch,ids];values=index.values[batch,ids]
            query=F.normalize(self.memory_codec.exact_query(query_z),dim=-1)
            scores=(query[:,:,None]*F.normalize(keys,dim=-1)).sum(-1).masked_fill(~mask,-torch.inf)
            selected=scores.argsort(dim=-1,descending=True,stable=True)[...,:min(16,width)]
            chosen=scores.gather(-1,selected)
            weights=torch.cat((chosen,self.memory_codec.exact_null.expand(*chosen.shape[:-1],1)),-1).softmax(-1)
            picked=values.gather(2,selected[...,None].expand(-1,-1,-1,values.shape[-1]))
            ke=(weights[...,:-1,None]*picked).sum(2)
            if 'persistent' in disabled_memories:kp=torch.zeros_like(kp)
            if 'exact' in disabled_memories:ke=torch.zeros_like(ke)
            kpz=self.core.condition(torch.cat((kp,roles),-1));kez=self.core.condition(torch.cat((ke,roles),-1))
            scale=(128*gains).tanh()
            local=z[:,2:]+(scale[0]*kpz+scale[1]*kez)*(count>0)[...,None]
        diagnostics=dict(ep.diagnostics,local_exact_candidate_weights=weights,
            local_exact_candidate_ids=ids.gather(-1,selected),local_prior_norm=kp.norm(dim=-1),local_exact_norm=ke.norm(dim=-1))
        return replace(ep,commit=DifferentiableCodecCommit.from_semantic(torch.cat((z[:,:2],local),1),
            ep.commit.memory_revisions,ep.commit.model_epoch),diagnostics=diagnostics)
