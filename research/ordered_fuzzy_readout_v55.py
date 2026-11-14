"""Input-role binding of the existing fuzzy N/D read, then Commit-only language.

No raw byte, source H, target length, or answer prefix crosses the Commit boundary.
Each role is a contribution to the SAME local fuzzy read used by V49--V53.
This is a codec candidate, not a meaning denoiser or a losslessness proof.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit, SealedCodecCommit
from research.three_memory_contract_v32 import PHASE

EOS_LOG_BIAS = 5.628366021279994  # actual V54 TRAIN loss weights, no DEV fitting


def fuzzy_role_reads(mu, values, valid, chunk=16):
    """[B,S,role,V]. Sum over role equals the original fuzzy local summary."""
    if mu.ndim != 3 or values.shape[:2] != mu.shape[:2] or valid.shape != mu.shape[:2]:
        raise ValueError('fuzzy role shape')
    with torch.autocast(mu.device.type, enabled=False):
        slots = (mu.shape[1]+chunk-1)//chunk
        padding = slots*chunk-mu.shape[1]
        w = F.pad(mu.float()*valid[...,None], (0,0,0,padding)).reshape(len(mu),slots,chunk,-1)
        v = F.pad(values.float(), (0,0,0,padding)).reshape(len(mu),slots,chunk,-1)
        mass = w.sum(2)
        membership = 1-(1-w).prod(2)
        rho = membership/membership.sum(-1,keepdim=True).clamp_min(1e-8)
        role_mass = (w*(rho/mass.clamp_min(1e-8))[:,:,None,:]).sum(-1)
        return v*role_mass[...,None]*membership.amax(-1)[:,:,None,None]


@dataclass(frozen=True)
class PatchCommit:
    local: torch.Tensor
    global_views: torch.Tensor
    terminal: torch.Tensor
    phase: str = PHASE


class OrderedFuzzyReadout(nn.Module):
    def __init__(self, dimension=512, value_dimension=256, hidden=512, chunk=16):
        super().__init__()
        if dimension % chunk: raise ValueError('disjoint equal role subspaces')
        self.chunk=chunk
        self.role_projection=nn.Linear(value_dimension,dimension//chunk,bias=False)
        nn.init.zeros_(self.role_projection.weight)
        self.readout=CommitPatchReadout(dimension,hidden,chunk)

    def bind_patch(self, local, global_views, terminal, fuzzy_roles):
        if fuzzy_roles.shape != (len(local),self.chunk,self.role_projection.in_features):
            raise ValueError('ordered fuzzy role shape')
        ordered=self.role_projection(fuzzy_roles*self.chunk).flatten(1)
        return PatchCommit(local+ordered,global_views,terminal)

    def decode_patch(self, commit):
        if type(commit) is not PatchCommit or commit.phase != PHASE:
            raise TypeError('language readout accepts only a codec Commit')
        return self.readout(commit.local,commit.global_views,commit.terminal)

    def forward(self, local, global_views, terminal, fuzzy_roles):
        return self.decode_patch(self.bind_patch(local,global_views,terminal,fuzzy_roles))

    def bind_commit(self, commit, fuzzy_roles):
        if type(commit) not in (DifferentiableCodecCommit,SealedCodecCommit) or commit.phase != PHASE:
            raise TypeError('frozen three-memory Commit required')
        z=commit.reader_tensor().to(self.role_projection.weight.device)
        if fuzzy_roles.shape[:3] != (len(z),z.shape[1]-2,self.chunk):
            raise ValueError('input Commit/role correspondence')
        code=self.role_projection(fuzzy_roles*self.chunk).flatten(-2)
        return DifferentiableCodecCommit.from_semantic(torch.cat((z[:,:2],z[:,2:]+code),1),
            commit.memory_revisions,commit.model_epoch)

    def from_commit(self, commit):
        return self.readout.from_commit(commit)

    def import_v54_head(self, state):
        self.readout.load_state_dict(state,strict=True)
        with torch.no_grad():
            self.readout.net[-1].bias.reshape(self.chunk,258)[:,256].sub_(EOS_LOG_BIAS)
