"""Three-memory reference/access contract for the language-codec stage.

Not a whole vNext implementation or a trained neural selector. The tensor bank
operations are differentiable references; the final byte-sealed Commit is an
inference boundary, not an end-to-end training wrapper. No denoising is executed.
"""
from dataclasses import dataclass
import hashlib
import torch

PHASE='language_meaning_language_codec_only'
REASONING_PREFIXES=('core.slot_router.','core.rel_left','core.rel_right','core.rel_score.',
    'core.rel_message.','core.time.','core.refiner.','core.refine_norm.','core.velocity.')

def assert_codec_training_only(named_parameters,denoise_calls=0):
    if denoise_calls!=0:raise ValueError('denoising is deferred until language codec is stable')
    if any(p.requires_grad and n.startswith(REASONING_PREFIXES) for n,p in named_parameters):
        raise ValueError('reasoning/denoising parameters must remain frozen in codec stage')

def _finite(x):
    if not torch.isfinite(x).all():raise ValueError('nonfinite memory')

def fuzzy_weights(mu):
    _finite(mu)
    if ((mu<0)|(mu>1)).any():raise ValueError('fuzzy membership bounds')
    mass=mu.sum(-1,keepdim=True)
    rho=torch.where(mass>1e-8,mu/mass.clamp_min(1e-8),torch.zeros_like(mu))
    return rho,mu.amax(-1)

@dataclass(frozen=True)
class PersistentFuzzySnapshot:
    anchors:torch.Tensor
    prior:torch.Tensor
    certified_N:torch.Tensor
    certified_D:torch.Tensor
    model_epoch:str

    @classmethod
    def prior_only(cls,anchors,prior,model_epoch):
        if anchors.ndim!=2 or prior.ndim!=2 or len(anchors)!=len(prior):raise ValueError('canonical anchor dimensions')
        _finite(anchors);_finite(prior)
        return cls(anchors.clone(),prior.clone(),torch.zeros_like(prior,dtype=torch.float64),torch.zeros(len(prior),1,dtype=torch.float64,device=prior.device),model_epoch)

    def read(self,mu):
        # No certified writer is supplied by this reference. Prior is learned
        # conditioning, never provenance or truth certification.
        rho,gamma=fuzzy_weights(mu)
        if self.certified_D.count_nonzero():raise NotImplementedError('certified-record merge/writer requires its own verified contract')
        return gamma[...,None]*(rho@self.prior.float())

@dataclass(frozen=True)
class ContextualFuzzyBank:
    long_N:torch.Tensor
    long_D:torch.Tensor
    recent_N:torch.Tensor
    recent_D:torch.Tensor
    decay:torch.Tensor
    source_id:str
    source_version:str

    @classmethod
    def empty(cls,batch,anchors,values,decay,source_id,source_version):
        if decay.shape!=(anchors,) or not ((decay>0)&(decay<1)).all():raise ValueError('anchor decay')
        n=torch.zeros(batch,anchors,values,dtype=torch.float64,device=decay.device);d=torch.zeros(batch,anchors,1,dtype=torch.float64,device=decay.device)
        return cls(n,d,n.clone(),d.clone(),decay.double().clone(),source_id,source_version)

    def write_after_read(self,mu,value,valid):
        if mu.shape!=self.long_D.shape[:2] or value.shape!=(mu.shape[0],self.long_N.shape[-1]) or valid.shape!=(len(mu),) or valid.dtype!=torch.bool:raise ValueError('context write shape/mask')
        fuzzy_weights(mu);_finite(value)
        weight=mu.double()*valid.double()[:,None];increment=weight[...,None]*value.double()[:,None]
        decay=torch.where(valid[:,None,None],self.decay[None,:,None],torch.ones_like(self.decay[None,:,None]))
        # Pure functional update: previous prefix bank remains unchanged. A
        # caller must perform its prefix read before invoking this operation.
        values=(self.long_N+increment,self.long_D+weight[...,None],
            decay*self.recent_N+increment,decay*self.recent_D+weight[...,None])
        for x in values:_finite(x)
        return ContextualFuzzyBank(*values,self.decay.clone(),self.source_id,self.source_version)

    def read(self,mu,long_weight=.5):
        if not 0<=long_weight<=1:raise ValueError('long/recent mixture')
        rho,gamma=fuzzy_weights(mu)
        def mean(n,d):return torch.where(d>1e-8,n/d.clamp_min(1e-8),torch.zeros_like(n))
        values=long_weight*mean(self.long_N,self.long_D)+(1-long_weight)*mean(self.recent_N,self.recent_D)
        # An empty contextual cell stays ZERO; it is never filled with prior.
        return (gamma[...,None].double()*(rho.double()@values)).float()

@dataclass(frozen=True)
class ExactSource:
    source_id:str
    version:str
    raw:bytes
    sha256:str

    @classmethod
    def from_input(cls,source_id,version,raw):
        raw=bytes(raw);raw.decode('utf-8')
        return cls(source_id,version,raw,hashlib.sha256(raw).hexdigest())

    def span(self,start,end):
        if type(start)is not int or type(end)is not int or not 0<=start<end<=len(self.raw):raise ValueError('source span bounds')
        # UTF8 byte boundaries, without normalizing accents or numbers.
        self.raw[:start].decode('utf-8');self.raw[start:end].decode('utf-8');self.raw[end:].decode('utf-8')
        if hashlib.sha256(self.raw).hexdigest()!=self.sha256:raise ValueError('source bytes changed')
        return self.raw[start:end]

@dataclass(frozen=True)
class CopySelection:
    source_id:str
    version:str
    start:int
    end:int
    decision_origin:str

@dataclass(frozen=True)
class CopyPolicy:
    stage:str='codec_learning'
    allow_full_source:bool=False

    def __post_init__(self):
        if self.stage not in ('codec_learning','post_semantic_commit'):raise ValueError('copy policy stage')
        if self.allow_full_source and self.stage!='post_semantic_commit':raise ValueError('no automatic full-source copy during codec learning')

@dataclass(frozen=True)
class CopyPermit:
    source_id:str
    version:str
    start:int
    end:int
    raw:bytes
    sha256:str
    decision_origin:str

def issue_permit(source,selection,policy=CopyPolicy()):
    """Mechanical source/span validation. DOES NOT learn or infer copy intent.

    CPU callers use explicit fixture selections; these are not neural success.
    A future semantic stage must supply the selection, and the realizer cannot
    create one by traversing source positions or consulting gold output spans.
    """
    if (source.source_id,source.version)!=(selection.source_id,selection.version):raise ValueError('source identity/version mismatch')
    whole=selection.start==0 and selection.end==len(source.raw)
    if whole and not (policy.stage=='post_semantic_commit' and policy.allow_full_source and selection.decision_origin=='semantic_commit'):
        raise ValueError('whole-source copy not authorized in current stage')
    raw=source.span(selection.start,selection.end)
    return CopyPermit(source.source_id,source.version,selection.start,selection.end,bytes(raw),hashlib.sha256(raw).hexdigest(),selection.decision_origin)

@dataclass(frozen=True)
class ThreeMemoryBundle:
    persistent:PersistentFuzzySnapshot
    contextual:ContextualFuzzyBank
    exact:tuple
    model_epoch:str

    def __post_init__(self):
        if self.model_epoch!=self.persistent.model_epoch:raise ValueError('mixed model epochs')
        if not isinstance(self.exact,tuple) or not self.exact:raise ValueError('explicit Exact source snapshot required')
        if not any((e.source_id,e.version)==(self.contextual.source_id,self.contextual.source_version) for e in self.exact):raise ValueError('context/exact provenance mismatch')

@dataclass(frozen=True)
class CodecCommit:
    shape:tuple
    semantic_bytes:bytes
    memory_revisions:tuple
    permits:tuple
    copy_stage:str

    @classmethod
    def seal(cls,z,bundle,permits=(),policy=CopyPolicy()):
        if not isinstance(bundle,ThreeMemoryBundle):raise TypeError('three-memory input required')
        if z.ndim!=3:raise ValueError('semantic slots')
        _finite(z)
        covered={}
        for permit in permits:
            sources=[e for e in bundle.exact if (e.source_id,e.version)==(permit.source_id,permit.version)]
            if len(sources)!=1 or sources[0].span(permit.start,permit.end)!=permit.raw:raise ValueError('permit not bound to snapshot')
            source=sources[0]
            selection=CopySelection(permit.source_id,permit.version,permit.start,permit.end,permit.decision_origin)
            if issue_permit(source,selection,policy)!=permit:raise ValueError('permit policy mismatch')
            covered.setdefault((source.source_id,source.version),set()).update(range(permit.start,permit.end))
        for source in bundle.exact:
            if len(covered.get((source.source_id,source.version),()))==len(source.raw) and not (policy.stage=='post_semantic_commit' and policy.allow_full_source and all(p.decision_origin=='semantic_commit' for p in permits)):
                raise ValueError('fragment tiling cannot bypass current whole-source restriction')
        raw=z.detach().float().cpu().contiguous().numpy().tobytes()
        revisions=(bundle.model_epoch,bundle.contextual.source_id,bundle.contextual.source_version)+tuple((e.source_id,e.version,e.sha256) for e in bundle.exact)
        return cls(tuple(z.shape),bytes(raw),revisions,tuple(permits),policy.stage)

    def reader_tensor(self):
        # Independent storage; returned tensors never alias immutable bytes.
        return torch.frombuffer(bytearray(self.semantic_bytes),dtype=torch.float32).reshape(self.shape).clone()

def realizer_inputs(commit):
    if not isinstance(commit,CodecCommit):raise TypeError('realizer accepts Commit only, not live memory stores or input H')
    return commit.reader_tensor(),commit.permits

def emit_copy(permit):
    if not isinstance(permit,CopyPermit) or hashlib.sha256(permit.raw).hexdigest()!=permit.sha256:raise ValueError('invalid sealed copy permit')
    return permit.raw
