"""Raw language -> input meaning -> denoised output meaning -> language.

First bounded comparison-domain composition, not a general semantic ontology.
Gold targets occur only in training/scoring functions, never in infer/realize.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from fdt_local.model_masked import Language,Core,RMS
from research.text_context_decoder_v2 import ContextDecoder,BOS
from research.text_diffusion_v1 import PAD

PHASES=('input','meaning','language')

class InputEncoder(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.language=Language(c);self.project=nn.Linear(c.lang_dim,c.z_dim)
        self.language.requires_grad_(False)
        for b in self.language.blocks[-2:]:b.requires_grad_(True)

    def forward(self,tokens):
        valid=tokens!=256;n=tokens.shape[1];pos=torch.arange(n,device=tokens.device)
        band=(pos[:,None]>=pos[None])&(pos[:,None]-pos[None]<self.c.window)
        mask=band[None,None]&(valid[:,None,None,:]|torch.eye(n,device=tokens.device,dtype=torch.bool)[None,None])
        with torch.no_grad():
            x=self.language.embedding(tokens)
            for b in self.language.blocks[:-2]:x=b(x,mask,valid)
        for b in self.language.blocks[-2:]:x=b(x,mask,valid)
        x=self.project(x)*valid[...,None]
        return torch.cat((x,x.new_zeros(len(x),1,x.shape[-1])),1),torch.cat((valid,torch.ones(len(x),1,device=x.device,dtype=torch.bool)),1)

class InputMeaning(nn.Module):
    """Four predicted pointers: query S/O and premise S/O; premise operator and language."""
    def __init__(self,c,capacity=160):
        super().__init__();d=c.z_dim
        self.queries=nn.Parameter(torch.randn(6,d)*.02)
        self.blocks=nn.ModuleList([Core(c) for _ in range(2)])
        self.pointer_query=nn.Linear(d,d,bias=False);self.pointer_key=nn.Linear(d,d,bias=False)
        self.byte_positions=nn.Embedding(capacity,d)
        self.operator=nn.Linear(d,4);self.language=nn.Linear(d,2)

    def forward(self,encoded,tokens):
        prompt,mask=encoded;n=tokens.shape[1]
        if n>self.byte_positions.num_embeddings:raise ValueError('finite pointer capacity')
        candidates=(tokens>=65)&(tokens<=72)&mask[:,:n]
        if not candidates.any(-1).all():raise ValueError('finite A-H scope')
        h=self.queries[None].expand(len(tokens),-1,-1)
        for b in self.blocks:h=b(h,prompt,torch.ones(h.shape[:2],device=h.device,dtype=torch.bool),mask)
        keys=self.pointer_key(prompt[:,:n]+self.byte_positions(torch.arange(n,device=h.device))[None])
        logits=torch.einsum('brd,bnd->brn',self.pointer_query(h[:,:4]).float(),keys.float())/math.sqrt(h.shape[-1])
        logits=logits.masked_fill(~candidates[:,None],-torch.inf)
        p=logits.softmax(-1);ids=(tokens-65).clamp(0,7)
        entities=p.new_zeros(len(tokens),4,8).scatter_add(-1,ids[:,None].expand(-1,4,-1),p)
        op=self.operator(h[:,4]).float();lang=self.language(h[:,5]).float()
        return {'entities':entities,'operator':op.softmax(-1),'language':lang.softmax(-1),
            'pointer_logits':logits,'operator_logits':op,'language_logits':lang,'candidate_mask':candidates}

    def loss(self,read,spans,labels):
        return (F.cross_entropy(read['pointer_logits'].flatten(0,1),spans.flatten(),reduction='none').mean()+
            F.cross_entropy(read['operator_logits'],labels[:,4],reduction='none').mean()+
            .25*F.cross_entropy(read['language_logits'],labels[:,5],reduction='none').mean())

class OutputMeaning(nn.Module):
    """Predict the answer's bound entities/relation; condition ONLY on extracted meaning."""
    def __init__(self,c):
        super().__init__();self.c=c;d=c.z_dim
        self.values=nn.ModuleList([nn.Embedding(k,d) for k in (8,8,8,8,4)])
        self.types=nn.Parameter(torch.randn(5,d)*.02)
        self.role_project=nn.Linear(16,d,bias=False)
        self.time=nn.Sequential(nn.Linear(64,d),nn.SiLU(),nn.Linear(d,d,bias=False))
        self.refiner=nn.ModuleList([Core(c) for _ in range(c.diffusion_layers)])
        self.refine_norm=RMS(d);self.velocity=nn.Linear(d,d,bias=False)
        g=torch.Generator().manual_seed(3901)
        self.register_buffer('basis',torch.linalg.qr(torch.randn(d,21,generator=g),mode='reduced').Q.T)
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),32)))

    def condition(self,read):
        probs=[read['entities'][:,j] for j in range(4)]+[read['operator']]
        values=torch.stack([p.float()@v.weight.float()+self.types[j] for j,(p,v) in enumerate(zip(probs,self.values))],1)
        return values,read['entities'][:,:2].flatten(1)

    def target(self,labels):
        # [query subject8, query object8, OUTPUT relation4, constant1].
        fields=torch.cat((F.one_hot(labels[:,0],8),F.one_hot(labels[:,1],8),F.one_hot(labels[:,2],4),torch.ones_like(labels[:,:1])),-1).float()
        return (fields@self.basis.float()*math.sqrt(self.c.z_dim/4))[:,None]

    def logits(self,z):
        fields=4*(z[:,0].float()@self.basis.float().T)/math.sqrt(self.c.z_dim/4)
        return fields[:,:8],fields[:,8:16],fields[:,16:20]

    def denoise(self,z,t,condition):
        values,roles=condition;phase=t.float()[:,None]*self.frequencies[None]
        time=self.time(torch.cat((phase.sin(),phase.cos()),-1))[:,None]
        x=z+time+self.role_project(roles)[:,None]
        for b in self.refiner:x=b(x,values,torch.ones(x.shape[:2],device=x.device,dtype=torch.bool),torch.ones(values.shape[:2],device=x.device,dtype=torch.bool))
        return self.velocity(self.refine_norm(x))

    def loss(self,read,labels,t,noise):
        clean=self.target(labels);a=(t[:,None,None]*math.pi/2).cos();s=(t[:,None,None]*math.pi/2).sin()
        z=a*clean+s*noise;v=self.denoise(z,t,self.condition(read)).float();out=a*z-s*v
        vmse=(v-(a*noise-s*clean)).square().mean();scores=self.logits(out)
        ce=[F.cross_entropy(score,labels[:,j],reduction='none').mean() for j,score in enumerate(scores)]
        return vmse+.5*ce[2]+.25*(ce[0]+ce[1]),{'velocity_mse':vmse,'output_relation_CE':ce[2]}

    @torch.no_grad()
    def infer(self,read,noise,charge=lambda kind:None,steps=16):
        z=noise.float();condition=self.condition(read)
        schedule=torch.linspace(1,0,steps+1).tolist()
        for t,nxt in zip(schedule,schedule[1:]):
            charge('denoise');v=self.denoise(z,z.new_full((len(z),),t),condition).float()
            a,s=(0.,1.) if t==1 else (math.cos(t*math.pi/2),math.sin(t*math.pi/2))
            clean=a*z-s*v;eps=s*z+a*v
            z=math.cos(nxt*math.pi/2)*clean+math.sin(nxt*math.pi/2)*eps
            if not torch.isfinite(z).all():raise FloatingPointError('semantic diffusion nonfinite')
        scores=self.logits(z)
        return {'labels':torch.stack([x.argmax(-1) for x in scores]+[read['language'].argmax(-1)],-1),
            'logits':torch.cat(scores,-1),'latent':z,'language_probabilities':read['language']}

class LanguageRealizer(nn.Module):
    """Receives committed OUTPUT meaning only; no original prompt or input facts."""
    def __init__(self,c,length=40):
        super().__init__();d=c.z_dim
        self.values=nn.ModuleList([nn.Embedding(n,d) for n in (8,8,4,2)])
        self.types=nn.Parameter(torch.randn(4,d)*.02);self.decoder=ContextDecoder(c,length)

    def context(self,committed):
        if committed.ndim!=2 or committed.shape[1]!=4:raise ValueError('committed output meaning only')
        factors=torch.stack([v(committed[:,i])+self.types[i] for i,v in enumerate(self.values)],1)
        # There is no raw context or byte-proposal latent memory in this path.
        mask=torch.ones(factors.shape[:2],device=factors.device,dtype=torch.bool)
        return self.decoder.context(factors,mask,factors.new_empty(len(factors),0,factors.shape[-1]))

    def loss(self,committed,answer_ids):
        prefix=torch.cat((torch.full_like(answer_ids[:,:1],BOS),answer_ids[:,:-1]),1)
        logits=self.decoder(prefix,self.context(committed)).float();active=answer_ids!=PAD
        return F.cross_entropy(logits.transpose(1,2),answer_ids.masked_fill(~active,-100),ignore_index=-100,reduction='none').sum()/active.sum()

    @torch.no_grad()
    def generate(self,committed,charge=lambda kind:None):
        return self.decoder.generate(self.context(committed),charge)

class SemanticPipeline(nn.Module):
    def __init__(self,c,length=40):
        super().__init__();self.c=c;self.encoder=InputEncoder(c);self.input=InputMeaning(c)
        self.meaning=OutputMeaning(c);self.realizer=LanguageRealizer(c,length)
        self.potential_trainable=[n for n,p in self.named_parameters() if p.requires_grad]
        self.set_phase('input')

    def set_phase(self,phase):
        if phase not in PHASES:raise ValueError('explicit phase')
        self.phase=phase
        for n,p in self.named_parameters():
            p.requires_grad_(n in self.potential_trainable and
                (n.startswith(('encoder.','input.')) if phase=='input' else n.startswith('meaning.') if phase=='meaning' else n.startswith('realizer.')))

    def extract(self,tokens,charge=lambda kind:None):
        charge('encode');encoded=self.encoder(tokens);charge('reader');return self.input(encoded,tokens)

    @torch.no_grad()
    def infer(self,tokens,noise,charge=lambda kind:None):
        read=self.extract(tokens,charge);committed=self.meaning.infer(read,noise,charge)
        return read,committed

    def warmstart(self,language_state,span_state,typed_state):
        self.encoder.language.load_state_dict({k[len('vp.language.'):]:v for k,v in language_state.items() if k.startswith('vp.language.')},strict=True)
        self.encoder.project.load_state_dict({k[len('vp.prompt_project.'):]:v for k,v in language_state.items() if k.startswith('vp.prompt_project.')},strict=True)
        old={k[len('reader.'):]:v for k,v in span_state.items() if k.startswith('reader.')}
        for name in ('blocks','pointer_query','pointer_key','byte_positions'):
            getattr(self.input,name).load_state_dict({k[len(name)+1:]:v for k,v in old.items() if k.startswith(name+'.')},strict=True)
        with torch.no_grad():
            self.input.queries[:2].copy_(old['queries'][:2]);self.input.queries[4:].copy_(old['queries'][2:])
        self.input.language.load_state_dict({k[len('semantic_heads.1.'):]:v for k,v in old.items() if k.startswith('semantic_heads.1.')},strict=True)
        for name in ('time','refiner','refine_norm','velocity','role_project'):
            getattr(self.meaning,name).load_state_dict({k[len(name)+1:]:v for k,v in typed_state.items() if k.startswith(name+'.')},strict=True)
        with torch.no_grad():self.meaning.basis.copy_(typed_state['basis']);self.meaning.frequencies.copy_(typed_state['frequencies'])
        self.realizer.decoder.load_state_dict({k[len('decoder.'):]:v for k,v in span_state.items() if k.startswith('decoder.')},strict=True)
        with torch.no_grad():
            for j in (0,1,3):self.realizer.values[j].weight.copy_(old[f'values.{j}.weight'])
            self.realizer.values[2].weight[:3].copy_(old['values.2.weight'][[2,0,1]])
            self.realizer.values[2].weight[3].copy_(old['values.2.weight'].mean(0));self.realizer.types.copy_(old['types'])
        return {'language':'V9 language/project only','extractor':'V4 pointer/blocks/language plus new premise pointers/operator',
            'output_denoiser':'original typed basis/time/refiner/norm/velocity/role_project; new semantic condition embeddings',
            'realizer':'V4 learned factors/decoder, relation order remapped; no V9 byte-index prior',
            'optimizer':'fresh composed-model AdamW; all source optimizers preserved separately'}
