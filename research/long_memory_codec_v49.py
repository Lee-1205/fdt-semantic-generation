"""Variable-length three-memory codec with scalable fuzzy slots and cached decoding."""
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from research.position_sensitive_codec_v46 import PositionSensitiveCodec
from research.three_memory_codec_v33 import (
    DifferentiableCodecCommit, SealedCodecCommit, ExactIndexSnapshot,
)
from research.three_memory_contract_v32 import ExactSource
from research.text_context_decoder_v2 import ContextDecoder, BOS
from research.text_diffusion_v1 import PAD
from research.utf8_codec_decode import generate_utf8


def attention_query(module, query, keys, values, mask=None):
    b,n,d = query.shape
    q = module.q(query).view(b,n,module.h,module.w).transpose(1,2)
    out = F.scaled_dot_product_attention(q,keys,values,attn_mask=mask,dropout_p=0.)
    return module.o(out.transpose(1,2).reshape(b,n,d))


def attention_memory(module, values):
    b,n,d = values.shape
    return (module.k(values).view(b,n,module.h,module.w).transpose(1,2),
            module.v(values).view(b,n,module.h,module.w).transpose(1,2))


class LongMemoryDecoder(ContextDecoder):
    def __init__(self, config, length=129, layers=2, chunk_bytes=16):
        super().__init__(config,length=length,layers=layers)
        self.chunk_bytes = chunk_bytes
        self.register_parameter('local_read_gain',None)
        self.register_parameter('local_slot_tag',None)

    def extend_positions(self, length):
        old = self.positions.weight.detach()
        if length < len(old):
            raise ValueError('capacity must preserve original position rows')
        index = torch.arange(length,device=old.device)
        weight = old[index.remainder(len(old)-1)].clone()
        frequency = torch.exp(-math.log(10000.)*torch.arange(old.shape[1],device=old.device)/old.shape[1])
        weight += .02*torch.sin(index[:,None]*frequency)
        weight[:len(old)] = old
        self.positions = nn.Embedding(length,old.shape[1],_weight=weight)
        self.length = length

    def enable_local_read(self, slots):
        if self.local_read_gain is not None:
            raise RuntimeError('local read already enabled')
        w = self.embedding.weight
        self.local_read_gain = nn.Parameter(w.new_zeros(len(self.blocks)))
        self.local_slot_tag = nn.Parameter(w.new_zeros(slots,w.shape[-1]))

    def local_mask(self, index, local_valid):
        slots = torch.arange(local_valid.shape[1],device=index.device)
        near = (index[:,None]//self.chunk_bytes-slots[None]).abs()<=1
        mask = near[None]&local_valid[:,None]
        mask = torch.where(mask.any(-1,keepdim=True),mask,local_valid[:,None])
        has = local_valid.any(-1)
        return mask|((~has)[:,None,None]&(slots==0)[None,None]),has

    def forward(self,prefix,context):
        if len(context)==2:
            return super().forward(prefix,context)
        values,valid,local,local_valid = context
        n = prefix.shape[1]
        index = torch.arange(n,device=prefix.device)
        x = self.embedding(prefix)+self.positions(index)[None]
        causal = torch.ones(n,n,device=x.device,dtype=torch.bool).tril()[None,None]
        mask,has = self.local_mask(index,local_valid)
        local = local+(128*self.local_slot_tag[:local.shape[1]]).tanh().to(local.dtype)[None]
        for i,block in enumerate(self.blocks):
            y = block.norms[0](x)
            x = x+block.self_attn(y,y,causal)
            q = block.norms[1](x)
            read = block.cross(q,values,valid[:,None,None,:])
            extra = block.cross(q,local,mask[:,None])
            x = x+read
            x = x+(128*self.local_read_gain[i]).tanh().to(x.dtype)*extra*has[:,None,None]
            x = x+block.ff(block.norms[2](x))
        return self.head(self.norm(x))

    def start_cache(self,context):
        global_values,valid = context[:2]
        memories = []
        for block in self.blocks:
            item = {'self_k':None,'self_v':None,
                    'global':attention_memory(block.cross,global_values)}
            if len(context)==4:
                local = context[2]+(128*self.local_slot_tag[:context[2].shape[1]]).tanh().to(context[2].dtype)[None]
                item['local'] = attention_memory(block.cross,local)
            memories.append(item)
        return {'position':0,'layers':memories,'context':context}

    def cached_step(self,token,cache):
        position = cache['position']
        context = cache['context']
        x = self.embedding(token)+self.positions.weight[position][None,None]
        if len(context)==4:
            mask,has = self.local_mask(torch.tensor([position],device=x.device),context[3])
        for i,(block,item) in enumerate(zip(self.blocks,cache['layers'])):
            normalized = block.norms[0](x)
            k,v = attention_memory(block.self_attn,normalized)
            item['self_k'] = k if item['self_k'] is None else torch.cat((item['self_k'],k),2)
            item['self_v'] = v if item['self_v'] is None else torch.cat((item['self_v'],v),2)
            x = x+attention_query(block.self_attn,normalized,item['self_k'],item['self_v'])
            q = block.norms[1](x)
            x = x+attention_query(block.cross,q,*item['global'],context[1][:,None,None,:])
            if len(context)==4:
                extra = attention_query(block.cross,q,*item['local'],mask[:,None])
                x = x+(128*self.local_read_gain[i]).tanh().to(x.dtype)*extra*has[:,None,None]
            x = x+block.ff(block.norms[2](x))
        cache['position'] += 1
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate_cached(self,context,charge=lambda kind:None,max_symbols=None):
        cache = self.start_cache(context)
        decoder = self
        class CachedCall:
            length = decoder.length if max_symbols is None else min(decoder.length,max_symbols)
            def __call__(self,prefix,unused):
                return decoder.cached_step(prefix[:,-1:],cache)
        return generate_utf8(CachedCall(),context,charge)


class LongMemoryCodec(PositionSensitiveCodec):
    def __init__(self,config,input_capacity=16384,chunk_bytes=16):
        super().__init__(config)
        self.input_capacity = input_capacity
        self.chunk_bytes = chunk_bytes
        old = self.decoder
        decoder = LongMemoryDecoder(config,length=old.length,layers=len(old.blocks),chunk_bytes=chunk_bytes)
        decoder.load_state_dict(old.state_dict(),strict=True)
        self.decoder = decoder

    def token_features(self,tokens):
        if tokens.dtype!=torch.long or tokens.ndim!=2 or not 1<=tokens.shape[1]<=self.input_capacity:
            raise ValueError('input exceeds declared byte capacity')
        if ((tokens<0)|(tokens>256)).any():
            raise ValueError('input byte range')
        valid = tokens!=256
        if not valid.any(1).all() or ((~valid).long().cumsum(1).bool()&valid).any():
            raise ValueError('nonempty contiguous input prefix')
        language = self.core.language
        x = language.embedding(tokens)
        pos = torch.arange(tokens.shape[1],device=tokens.device)
        band = (pos[:,None]>=pos[None])&(pos[:,None]-pos[None]<self.c.window)
        eye = torch.eye(tokens.shape[1],device=tokens.device,dtype=torch.bool)[None,None]
        mask = band[None,None]&(valid[:,None,None,:]|eye)
        for block in language.blocks:
            if self.c.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block,x,mask,valid,use_reentrant=False)
            else:
                x = block(x,mask,valid)
        weights = (x.float()@language.pool.float().T).masked_fill(~valid[...,None],-1e4).softmax(1)
        return x.float(),torch.einsum('blv,bld->bvd',weights,x.float()),valid

    def _exact_index(self,tokens,h,valid,source_ids,source_versions,epoch):
        if len(source_ids)!=len(h) or len(source_versions)!=len(h):
            raise ValueError('one source identity/version per input')
        sources,spans,starts,widths = [],[],[],[]
        for row,(identity,version) in enumerate(zip(source_ids,source_versions)):
            raw = bytes(tokens[row,valid[row]].detach().cpu().tolist())
            source = ExactSource.from_input(identity,version,raw)
            boundaries = []
            offset = 0
            for char in raw.decode('utf-8'):
                end = offset+len(char.encode())
                boundaries.append((offset,end))
                offset = end
            sources.append(source)
            spans.append(tuple(boundaries))
            starts.append(torch.tensor([a for a,b in boundaries],device=h.device))
            widths.append(torch.tensor([b-a for a,b in boundaries],device=h.device))
        revisions = {}
        for source in sources:
            identity = source.source_id,source.version
            if identity in revisions and revisions[identity]!=source.sha256:
                raise ValueError('conflicting bytes under the same source revision')
            revisions[identity] = source.sha256
        starts = nn.utils.rnn.pad_sequence(starts,batch_first=True)
        widths = nn.utils.rnn.pad_sequence(widths,batch_first=True)
        candidate_valid = widths>0
        max_width = int(widths.max())
        offset = torch.arange(max_width,device=h.device)
        positions = (starts[...,None]+offset).clamp_max(h.shape[1]-1)
        features = h[torch.arange(len(h),device=h.device)[:,None,None],positions]
        features = (features*(offset[None,None]<widths[...,None])[...,None]).sum(2)/widths.clamp_min(1)[...,None]
        keys = self.memory_codec.exact_key(features)
        roles = features.new_zeros(*features.shape[:2],16)
        values = self.core.value(torch.cat((features,roles),-1))
        return ExactIndexSnapshot(tuple(sources),tuple(spans),keys,values,candidate_valid,epoch)

    def enable_long_capacity(self):
        self.decoder.extend_positions(self.input_capacity+1)
        self.decoder.enable_local_read(math.ceil(self.input_capacity/self.chunk_bytes))

    def encode_input(self,tokens,*,source_ids,source_versions,model_epoch,disabled_memories=()):
        episode = super().encode_input(tokens,source_ids=source_ids,source_versions=source_versions,
                    model_epoch=model_epoch,disabled_memories=disabled_memories)
        if self.decoder.local_read_gain is None:
            return episode
        z = episode.commit.reader_tensor()
        slots = math.ceil(tokens.shape[1]/self.chunk_bytes)
        local = z.new_zeros(len(z),slots,z.shape[-1])
        if 'contextual' not in disabled_memories:
            with torch.autocast(tokens.device.type,enabled=False):
                valid = tokens!=256
                mu = episode.diagnostics['token_memberships'].float()
                values = episode.diagnostics['token_values'].float()
                padding = slots*self.chunk_bytes-tokens.shape[1]
                weight = F.pad(mu*valid[...,None],(0,0,0,padding)).reshape(len(z),slots,self.chunk_bytes,-1)
                value = F.pad(values,(0,0,0,padding)).reshape(len(z),slots,self.chunk_bytes,-1)
                mass = weight.sum(2)
                means = (weight.transpose(-1,-2)@value)/mass[...,None].clamp_min(1e-8)
                membership = 1-(1-weight).prod(2)
                rho = membership/membership.sum(-1,keepdim=True).clamp_min(1e-8)
                summary = (rho[...,None,:]@means).squeeze(-2)*membership.amax(-1,keepdim=True)
                local = self.core.condition(torch.cat((summary,summary.new_zeros(*summary.shape[:2],16)),-1))
                local = local*(mass.sum(-1)>0)[...,None]
        return replace(episode,commit=DifferentiableCodecCommit.from_semantic(torch.cat((z,local),1),
                       episode.commit.memory_revisions,episode.commit.model_epoch))

    def context(self,commit,mode='normal'):
        if type(commit) not in (DifferentiableCodecCommit,SealedCodecCommit):
            raise TypeError('decoder requires a codec Commit')
        z = commit.reader_tensor()
        if z.shape[1]==2:
            return super().context(commit,mode)
        if self.decoder.local_read_gain is None or z.shape[1]>2+len(self.decoder.local_slot_tag):
            raise ValueError('local Commit capacity')
        base = DifferentiableCodecCommit.from_semantic(z[:,:2],commit.memory_revisions,commit.model_epoch)
        values,valid = super().context(base,mode)
        local = z[:,2:].to(values.device)
        if mode=='swap':
            local = local.roll(max(1,len(local)//2),0)
        elif mode=='zero':
            local = torch.zeros_like(local)
        return values,valid,local,(local!=0).any(-1)

    def corrupt_teacher_prefix(self,target_ids):
        if not self.training or target_ids.shape[1]>self.input_capacity+1:
            raise ValueError('training-only dynamic prefix capacity')
        prefix = torch.cat((torch.full_like(target_ids[:,:1],BOS),target_ids[:,:-1]),1)
        selected = (torch.rand(prefix.shape,device=prefix.device)<.5)&(prefix<256)&(prefix>=0)
        eligible = (prefix>=0)&(prefix<256)
        return prefix.masked_fill(selected,PAD),selected,eligible

    def reconstruction_loss(self,tokens,target_ids,*,source_ids,source_versions,model_epoch,charge=lambda kind:None):
        if not self.training or target_ids.shape!=(len(tokens),tokens.shape[1]+1):
            raise ValueError('same-input dynamic reconstruction shape')
        expected = torch.full_like(target_ids,PAD)
        expected[:,:tokens.shape[1]] = tokens.masked_fill(tokens==256,PAD)
        expected.scatter_(1,(tokens!=256).sum(1,keepdim=True),256)
        if not torch.equal(expected,target_ids):
            raise ValueError('reconstruct the same input plus EOS')
        charge('encode')
        episode = self.encode_input(tokens,source_ids=source_ids,source_versions=source_versions,model_epoch=model_epoch)
        charge('decoder')
        clean = self.teacher_logits(episode.commit,target_ids)
        prefix,selected,eligible = self.corrupt_teacher_prefix(target_ids)
        charge('decoder')
        corrupt = self.decoder(prefix,self.context(episode.commit)).float()
        active = target_ids!=PAD
        labels = target_ids.masked_fill(~active,-100)
        clean_loss = F.cross_entropy(clean.transpose(1,2),labels,ignore_index=-100,reduction='none').sum()/active.sum()
        corrupt_loss = F.cross_entropy(corrupt.transpose(1,2),labels,ignore_index=-100,reduction='none').sum()/active.sum()
        return .5*(clean_loss+corrupt_loss),episode

    @torch.no_grad()
    def reconstruct(self,commit,charge=lambda kind:None,mode='normal',max_symbols=None):
        return self.decoder.generate_cached(self.context(commit,mode),charge,max_symbols)
