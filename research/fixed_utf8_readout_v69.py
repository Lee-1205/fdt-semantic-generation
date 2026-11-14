"""Linear-time equivalent of the fixed-logit UTF8 FSM, no source or prefix.

The old reference repeatedly concatenates an ever-growing unused prefix.
This implementation uses only the current score row and UTF8 state.
"""
import numpy as np
import torch
from research.codec_commit_stop_v56 import guard_codec_stop

def decode_long_commit(logits,guard=lambda:None):
    scores=guard_codec_stop(logits).detach().float().cpu().reshape(-1,258).numpy()
    ids=[];remaining=0;low=128;high=191;cap=len(scores)
    for i,row in enumerate(scores):
        if i%4096==0:guard()
        if remaining:
            byte=low+int(row[low:high+1].argmax())
        else:
            legal=np.zeros(258,dtype=bool);legal[:128]=True;legal[256]=True
            left=cap-i
            if left>=2:legal[194:224]=True
            if left>=3:legal[224:240]=True
            if left>=4:legal[240:245]=True
            byte=int(np.where(legal,row,-np.inf).argmax())
        if not np.isfinite(row[byte]):raise FloatingPointError('no finite legal UTF8 score')
        ids.append(byte)
        if remaining:remaining-=1
        elif 194<=byte<=223:remaining=1
        elif 224<=byte<=239:remaining=2
        elif 240<=byte<=244:remaining=3
        low=160 if byte==224 else 144 if byte==240 else 128
        high=159 if byte==237 else 143 if byte==244 else 191
        if byte==256:break
    if remaining:raise RuntimeError('cap inside UTF8 character')
    raw=bytes(x for x in ids if x<256)
    return dict(ids=ids,text=raw.decode('utf-8'),EOS=bool(ids and ids[-1]==256),UTF8=True,output_bytes=len(raw))
