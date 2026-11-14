"""TRAIN-only bounded class balance for a fuzzy sentence reconstruction cycle."""
import math
OBJECTIVE='TRAIN_row_mass_balanced_byte_CE_bounded_.5_to4_plus_.1_EOS'

def class_id(byte):
    return 0 if 65<=byte<=90 else 1 if 97<=byte<=122 else 2 if 48<=byte<=57 else 3

def training_weights(rows):
    if not rows:raise ValueError('TRAIN pool required')
    mass=[0.]*4
    for row in rows:
        raw=row['prompt'].encode();counts=[0]*4
        for b in raw:counts[class_id(b)]+=1
        for k in range(4):mass[k]+=counts[k]/(len(raw)+1)/len(rows)
    if not all(mass):raise ValueError('all four TRAIN classes required')
    # Fix expected per-window byte gradient scale; bound rare-class emphasis.
    lo,hi=0.,16.
    for _ in range(80):
        a=(lo+hi)/2;weights=[min(4.,max(.5,a/math.sqrt(q))) for q in mass]
        if sum(q*w for q,w in zip(mass,weights))>sum(mass):hi=a
        else:lo=a
    a=(lo+hi)/2;weights=[min(4.,max(.5,a/math.sqrt(q))) for q in mass]
    lookup=[weights[class_id(b)] for b in range(256)]+[1.,0.]
    return lookup,dict(objective=OBJECTIVE,mean_per_window_byte_mass=mass,class_weights=weights,EOS_weight=1.,PAD_weight=0.,
        normalization=sum(q*w for q,w in zip(mass,weights))/sum(mass),TRAIN_only=True)
