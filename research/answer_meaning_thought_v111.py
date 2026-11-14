"""Conditional NEW answer-meaning learning. Codec/memories/D remain frozen.

Source meaning is the condition; E(answer), never E(input), is the target.
This new schema prevents claiming an identity checkpoint is an exact resume.
"""
import hashlib
import random
from research.latent_identity_thought_v110 import LatentIdentityEngine,LatentIdentityObjective


def answer_curriculum(selected_rows, epochs=16):
    if len(selected_rows) != 134 or epochs != 16 or len({r['id'] for r in selected_rows}) != 134:
        raise ValueError('fixed134TRAIN sixteen finite epochs')
    pairs = {}
    for row in selected_rows:
        pairs.setdefault(row['pair_key'], []).append(row['id'])
    real = [dict(ids=sorted(v), real_reverse=True, key=k) for k,v in sorted(pairs.items()) if len(v)==2]
    singles = sorted(v[0] for v in pairs.values() if len(v)==1)
    if len(real)!=58 or len(singles)!=18 or any(len(v)>2 for v in pairs.values()):
        raise ValueError('58 actual pairs +18 singleton rows; do not complete missing pairs')
    # Nine scheduling blocks are NOT semantic pairs. No pair-consistency loss
    # is applied to them. Two-row blocks preserve actual pairs across batches.
    units = real + [dict(ids=singles[i:i+2], real_reverse=False, key='single_schedule:'+str(i)) for i in range(0,18,2)]
    output=[]
    for epoch in range(epochs):
        order=list(units);random.Random(111000+epoch).shuffle(order);rows=[]
        for unit in order:
            slot=(epoch+int(hashlib.sha256(unit['key'].encode()).hexdigest()[:8],16))%16
            rows.extend(dict(id=i,t=1-slot/16,real_reverse_pair=unit['real_reverse']) for i in unit['ids'])
        output.append(rows)
    return output


class ConditionalAnswerObjective(LatentIdentityObjective):
    """Endpoint and VP losses on the supplied E(Y), no source-target shortcut."""


class AnswerMeaningEngine(LatentIdentityEngine):
    """Same pure-latent loss, different task/source metadata/fullstate schema.

The inherited objective consumes an explicit frozen answer packet; it never
constructs a target from the source. The worker supplies ONLY E(Y) packets.
"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.objective=ConditionalAnswerObjective(self.thought,self.update,self.codec_SHA,self.latent_policy)

    def payload(self,*,audit=True):
        value=super().payload(audit=audit)
        value.update(schema='conditional_answer_meaning_fullstate_v111',target_is='E(answer)_NOT_E(input)')
        return value

    def restore(self,value,*,audit=True):
        if value['schema']!='conditional_answer_meaning_fullstate_v111' or value.get('target_is')!='E(answer)_NOT_E(input)':
            raise ValueError('conditional answer task checkpoint only')
        super().restore(dict(value,schema='latent_identity_fullstate_v110'),audit=audit)
