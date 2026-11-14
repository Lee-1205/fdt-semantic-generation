"""Prospective, small controlled meanings; labels are scoring-only, never features."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
import text_generation_support_v1 as support

def main():
    out=ROOT/'research/artifacts/semantic_controls_20261111_v58_cpu'
    if out.exists():raise FileExistsError('immutable diagnostic data')
    names=('Mina','Owen','Nora','Liam','Sara','Noah','Lena','Adam')
    objects=('book','cup','key','ball');features=[];targets=[];groups=[]
    for kind in ('roles','negation','quantity'):
        for i in range(16):
            a,b=names[i%8],names[(i%8+1+i//8)%8];obj=objects[i%4]
            if kind=='roles':
                text=f'{a} gave {b} a {obj}.'
                paraphrase=f'{b} received a {obj} from {a}.'
                contrary=f'{b} gave {a} a {obj}.'
                labels=[dict(agent=a,recipient=b,object=obj,polarity=1)]*2+[dict(agent=b,recipient=a,object=obj,polarity=1)]
            elif kind=='negation':
                text=f'{a} did not open the gate.'
                paraphrase=f'It is not true that {a} opened the gate.'
                contrary=f'{a} opened the gate.'
                labels=[dict(agent=a,action='open gate',polarity=0)]*2+[dict(agent=a,action='open gate',polarity=1)]
            else:
                n=i+1;words=('one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen')
                text=f'The box contains {n} red tokens.'
                paraphrase=f'There are {words[i]} red tokens in the box.'
                contrary=f'The box contains {n+1} red tokens.'
                labels=[dict(container='box',quantity=n,item='red tokens')]*2+[dict(container='box',quantity=n+1,item='red tokens')]
            gid=f'{kind}:{i:02d}';ids=[]
            for variant,prompt,label in zip(('canonical','paraphrase','changed','layout'),(text,paraphrase,contrary,'\n  '+text+'  \n'),labels+[labels[0]]):
                rid=gid+':'+variant;features.append(dict(id=rid,prompt=prompt));targets.append(dict(id=rid,group=gid,kind=kind,variant=variant,meaning=label));ids.append(rid)
            assert labels[0]==labels[1] and labels[0]!=labels[2]
            if kind=='roles':assert sorted(text.encode())==sorted(contrary.encode())
            groups.append(dict(id=gid,kind=kind,ids=ids))
    assert len(features)==192 and len(groups)==48 and all(set(f)=={'id','prompt'} for f in features)
    # Repeated negation/name facts are intentionally correlated. No independent
    # 192-case or broad natural-language benchmark claim is made.
    manifest=dict(groups=48,feature_rows=192,duplicates=len(features)-len({f['prompt'] for f in features}),
        max_UTF8_bytes=max(len(f['prompt'].encode()) for f in features),source='root-authored controlled diagnostic',
        no_model_training=True,no_E2_access=True,labels_are_score_only=True,
        distance_spaces=['global_three_memory_Commit','contextual_anchor_means','anchor_activation','mean_ordered_local_Commit','byte_bag_control'],
        metric='same-meaning paraphrase/layout should rank closer than meaning change; a metric alone neither proves nor disproves semantics',
        intervention='first12 role groups: same global/local Commit shape and byte multiset; swap global2 vs all local slots',
        purpose='separate semantic organization from byte-position reconstruction; no denoising/optimizer',
        cost=dict(encode=192,readout=36,total=228),max_model_calls=240)
    out.mkdir()
    for name,value in [('features',features),('targets',targets),('groups',groups),('manifest',manifest)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    proof=dict(all192_feature_target_ids_match=[f['id'] for f in features]==[t['id'] for t in targets],
        all48_equivalence_contrast_labels_checked=True,role_reversals_same_byte_multiset=True,
        no_gold_in_feature_keys=True,duplicate_rows_explicit=manifest['duplicates'],CPU_only=True)
    (out/'root_CPU_contract.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(manifest=manifest,manifest_sha=support.sha(out/'manifest.json'))))
if __name__=='__main__':main()
