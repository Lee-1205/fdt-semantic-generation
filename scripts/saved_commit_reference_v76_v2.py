"""Pinned one-level saved neural Commit resolver for V76 retention evaluation.

V75 observed-panel packets may reference a V74 packet without source fields.
Such legacy packets are accepted ONLY behind a strict metadata-bearing pinned
referencer with identical targets, known schema, and dormant Exact policy. No recursive reference,
source-text retrieval, reconstruction target, or Exact entry is a decoder input.
New TRAIN rows and new evaluation rows still require actual source encoding.
"""
import hashlib
from pathlib import Path
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def resolve_saved_commit(packet,row,embedding_sha,pins,root,hidden_dim):
    expected_source=hashlib.sha256(row['prompt'].encode()).hexdigest()
    def identity(p):
        if p.get('source_id')!=row['id'] or p.get('source_SHA')!=expected_source or p.get('embedding_sha')!=embedding_sha or p.get('Exact_entries')!=0:
            raise ValueError('saved Commit source/upstream/Exact identity')
    identity(packet)
    direct='fuzzy_commit' in packet;referenced='fuzzy_commit_reference' in packet
    if direct==referenced:raise ValueError('exactly one saved Commit representation')
    provenance=None
    if referenced:
        ref=packet['fuzzy_commit_reference']
        if set(ref)!={'path','SHA','key','embedding_sha'} or ref['key']!='fuzzy_commit' or ref['embedding_sha']!=embedding_sha:
            raise ValueError('saved Commit reference schema/upstream')
        path=Path(ref['path']).resolve();base=Path(root).resolve()
        if not path.is_relative_to(base):raise ValueError('saved Commit reference outside root')
        if pins.get(str(path))!=ref['SHA'] or sha(path)!=ref['SHA']:raise ValueError('saved Commit reference not pinned')
        other=torch.load(path,map_location='cpu',weights_only=True)
        metadata={'source_id','source_SHA','embedding_sha'}
        if metadata.issubset(other):
            identity(other)
        elif not metadata.intersection(other):
            # Known V74 direct packets predate per-packet source metadata.
            # Source/upstream identity comes from the STRICT verified V75/V76
            # referencer plus exact pinned file SHA and identical labels.
            legacy={'logits','targets','fuzzy_commit','Exact_entries','Exact_policy','evaluation_mode'}
            policy='DORMANT_UNTIL_FUTURE_EXPLICIT_INTERNAL_COPY_REQUEST'
            if (set(other)!=legacy or other['Exact_entries']!=0 or other['Exact_policy']!=policy
                    or packet.get('Exact_policy')!=policy or not isinstance(other['evaluation_mode'],str)):
                raise ValueError('unknown legacy saved Commit schema/Exact policy')
        else:
            raise ValueError('partial legacy source metadata rejected')
        if 'fuzzy_commit_reference' in other or 'fuzzy_commit' not in other:raise ValueError('one-level direct saved Commit required')
        if not torch.equal(other['targets'],packet['targets']):raise ValueError('saved Commit target integrity')
        source=other;provenance=dict(ref)
    else:source=packet
    z=source['fuzzy_commit'];slots=(len(row['prompt'].encode())+15)//16
    if (not torch.is_tensor(z) or z.device.type!='cpu' or z.shape!=(1,slots+2,hidden_dim)
            or not z.is_floating_point() or not torch.isfinite(z).all()):raise ValueError('saved Commit tensor schema/numeric')
    return z,provenance
