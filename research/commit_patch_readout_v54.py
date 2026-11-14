"""Shared prefix-free language readout over a frozen three-memory Commit.

This is a decoder probe, not a replacement of FDT's encoder/memories/denoiser.
It receives semantic slots and a structural terminal slot, never source bytes.
"""
import torch
from torch import nn
from research.three_memory_codec_v33 import DifferentiableCodecCommit,SealedCodecCommit
from research.three_memory_contract_v32 import PHASE
from research.utf8_codec_decode import generate_utf8

class CommitPatchReadout(nn.Module):
    def __init__(self,dimension=512,hidden=512,chunk=16):
        super().__init__();self.dimension=dimension;self.chunk=chunk
        self.terminal=nn.Parameter(torch.randn(dimension)*.02)
        self.norm=nn.LayerNorm(3*dimension)
        self.net=nn.Sequential(nn.Linear(3*dimension,hidden),nn.SiLU(),nn.Linear(hidden,chunk*258))

    def forward(self,local,global_views,terminal):
        if local.shape[-1]!=self.dimension or global_views.shape!=(len(local),2,self.dimension):
            raise ValueError('Commit slot dimensions')
        if terminal.dtype!=torch.bool or terminal.shape!=(len(local),):raise ValueError('structural terminal role')
        local=local+terminal[:,None]*self.terminal
        features=torch.cat((local,global_views.flatten(1)),-1)
        return self.net(self.norm(features)).reshape(len(local),self.chunk,258)

    def from_commit(self,commit):
        if type(commit) not in (DifferentiableCodecCommit,SealedCodecCommit) or commit.phase!=PHASE:
            raise TypeError('readout accepts only a codec Commit')
        z=commit.reader_tensor().to(self.terminal.device)
        if len(z)!=1 or z.shape[1]<=2:raise ValueError('one variable-length local Commit')
        local=torch.cat((z[0,2:],z.new_zeros(1,self.dimension)),0)
        terminal=torch.zeros(len(local),device=local.device,dtype=torch.bool);terminal[-1]=True
        return self(local,z[:,:2].expand(len(local),-1,-1),terminal)

@torch.no_grad()
def decode_fixed_logits(logits,max_symbols=16385):
    """UTF8 FSM reads neural scores only. No source length/bytes or forced EOS."""
    logits=logits.detach().float().cpu().reshape(-1,258)
    class Scores:
        length=min(len(logits),max_symbols)
        def __call__(self,prefix,context):return logits[prefix.shape[1]-1][None,None]
    ids=generate_utf8(Scores(),(torch.zeros(1,1,1),))[0].tolist()
    before=ids[:ids.index(256)] if 256 in ids else ids
    raw=bytes(i for i in before if i<256)
    return dict(ids=ids,text=raw.decode('utf-8'),EOS=256 in ids,UTF8=True,output_bytes=len(raw))
