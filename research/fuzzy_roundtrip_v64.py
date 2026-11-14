"""Language -> fuzzy-memory Commit -> language. Exact is dormant in this stage.

No current copy request is handled. A future model-internal exact-copy request
policy is separate work. Ordinary input text is never persisted in Exact, nor
automatically indexed there as per-codepoint neural keys/values.
"""
from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from research.prefix_independent_codec_v53 import PrefixIndependentCodec
from research.three_memory_codec_v33 import ExactIndexSnapshot, DifferentiableCodecCommit, SealedCodecCommit
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout, fuzzy_role_reads

EXACT_POLICY='DORMANT_UNTIL_FUTURE_EXPLICIT_INTERNAL_COPY_REQUEST'


@dataclass(frozen=True)
class SourceRevisionOnly:
    source_id:str
    version:str
    sha256:str


class FuzzyCodecV64(PrefixIndependentCodec):
    def _exact_index(self,tokens,h,valid,source_ids,source_versions,epoch):
        if len(source_ids)!=len(tokens) or len(source_versions)!=len(tokens):raise ValueError('source provenance')
        revisions=[];seen={}
        for i,(identity,version) in enumerate(zip(source_ids,source_versions)):
            if not isinstance(identity,str) or not identity or not isinstance(version,str) or not version:raise ValueError('source identity/version')
            # Ephemeral input hashing; no raw/text/length/character span in Exact.
            digest=hashlib.sha256(bytes(tokens[i,valid[i]].detach().cpu().tolist())).hexdigest()
            key=(identity,version)
            if key in seen and seen[key]!=digest:raise ValueError('conflicting source revision')
            seen[key]=digest
            revisions.append(SourceRevisionOnly(identity,version,digest))
        # One invalid NULL sentinel keeps legacy attention well-defined. It is
        # not a source entry. No input-dependent key, value or span is written.
        return ExactIndexSnapshot(tuple(revisions),tuple(() for _ in revisions),
            h.new_zeros(len(tokens),1,self.c.key_dim),h.new_zeros(len(tokens),1,self.c.value_dim),
            torch.zeros(len(tokens),1,device=h.device,dtype=torch.bool),epoch)


class FuzzyRoundTripV64(nn.Module):
    def __init__(self,encoder,readout):
        super().__init__();self.encoder=encoder;self.readout=readout
        self.exact_policy=EXACT_POLICY

    def extract(self,tokens,*,source_ids,source_versions,model_epoch,disabled_memories=()):
        episode=self.encoder.encode_input(tokens,source_ids=source_ids,source_versions=source_versions,model_epoch=model_epoch,disabled_memories=disabled_memories)
        exact=episode.memories.exact
        if exact.valid.any() or exact.keys.count_nonzero() or exact.values.count_nonzero() or any(exact.spans):
            raise ValueError('Exact write prohibited during fuzzy roundtrip')
        if any(set(vars(s))!={'source_id','version','sha256'} for s in exact.sources):raise ValueError('Exact contains more than provenance')
        # Binding remains INSIDE extraction. The language reader receives only
        # the resulting Commit, not token H, input bytes or unbound diagnostics.
        roles=fuzzy_role_reads(episode.diagnostics['token_memberships'],episode.diagnostics['token_values'],tokens!=256)
        if 'contextual' in disabled_memories:roles=torch.zeros_like(roles)
        commit=self.readout.bind_commit(episode.commit,roles)
        return commit,episode

    def verbalize(self,commit):
        if type(commit) not in (DifferentiableCodecCommit,SealedCodecCommit):raise TypeError('Commit-only language boundary')
        return self.readout.from_commit(commit)

    def forward(self,tokens,*,source_ids,source_versions,model_epoch):
        commit,episode=self.extract(tokens,source_ids=source_ids,source_versions=source_versions,model_epoch=model_epoch)
        return self.verbalize(commit),commit,episode


def enable_cycle_training(model):
    """Train language extraction and language realization jointly, not reasoning."""
    model.requires_grad_(False)
    last=len(model.encoder.core.language.blocks)
    prefixes=tuple(f'encoder.core.language.blocks.{i}.' for i in range(max(0,last-2),last))
    prefixes+=('encoder.core.value.','encoder.core.condition.','readout.')
    names=[]
    for name,value in model.named_parameters():
        if name.startswith(prefixes):value.requires_grad_(True);names.append(name)
    if not names:raise ValueError('no trainable encoder/decoder')
    return names
