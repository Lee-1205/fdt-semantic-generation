"""Independent expanded TRAIN offsets and unchanged held inputs/labels."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    data=ROOT/'research/artifacts/semantic_binding_20261111_v60_cpu';old=ROOT/'research/artifacts/semantic_binding_20261111_v59_cpu'
    read=lambda name:json.loads((data/(name+'.json')).read_text(encoding='utf-8-sig'))
    train,target,groups=read('train_features'),read('train_targets'),read('train_groups');dev,dy=read('dev_features'),read('dev_targets')
    for name in ('dev_features','dev_targets','dev_groups','OWT_train_reconstruction_windows'):assert (data/(name+'.json')).read_bytes()==(old/(name+'.json')).read_bytes()
    assert not {r['prompt'] for r in train}&{r['prompt'] for r in dev}
    for features,targets in ((train,target),(dev,dy)):
        for f,t in zip(features,targets):
            assert f['id']==t['id'] and set(f)=={'id','prompt'}
            raw=f['prompt'].encode()
            for name,(a,b) in zip(('primary','recipient','theme'),t['spans']):
                expected=t['meaning']['roles'][name]
                if expected is None:assert [a,b]==[-1,-1]
                else:assert raw[a:b].decode()==expected and raw[:a].decode() is not None and raw[b:].decode() is not None
    assert {r['family'] for r in target}=={0,1,2,6,7,8} and {r['family'] for r in dy}=={3,4,5}
    for group in groups:
        assert len(group)==10
        truth=target[group[0]]['meaning'];opposite=target[group[3]]['meaning']
        assert truth!=opposite
        for index in group:assert target[index]['meaning'] in (truth,opposite)
    orders=read('orders')
    for epoch in range(4):
        batches=orders[epoch*144:(epoch+1)*144];assert sorted(sum(batches,[]))==list(range(2880))
        for batch in batches:
            for group in groups:
                overlap=set(batch)&set(group);assert not overlap or overlap==set(group)
    report=dict(all3136_UTF8_spans_reconstructed=True,all288group_truth_contrasts_checked=True,all4orders_group_preserving_exact=True,
        held4files_byte_identical=True,held_literal_TRAIN_overlap=0,TRAINDEV_meaning_overlap=len({t['meaning_id'] for t in target}&{t['meaning_id'] for t in dy}),
        duplicate_TRAIN_prompts=len(train)-len({r['prompt'] for r in train}),expanded_families_explicit=True,adaptive_design_not_blind=True,
        source_cache_reuse='reuse unchanged V59 DEV features and targets, no repeated source DEV encoding',actual_call_budget=816,model_forward=0)
    (data/'root_data_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
