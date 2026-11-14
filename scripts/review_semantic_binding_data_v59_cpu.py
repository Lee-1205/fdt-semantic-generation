"""Independent source offsets/splits/finite group orders/corpus provenance review."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import text_generation_support_v1 as support

def main():
    data=ROOT/'research/artifacts/semantic_binding_20261111_v59_cpu'
    read=lambda name:json.loads((data/(name+'.json')).read_text(encoding='utf-8-sig'))
    tf,ty,tg=read('train_features'),read('train_targets'),read('train_groups');df,dy,dg=read('dev_features'),read('dev_targets'),read('dev_groups')
    old=json.loads((ROOT/'research/artifacts/semantic_controls_20261111_v58_cpu/features.json').read_text(encoding='utf-8-sig'))
    for features,targets,groups in ((tf,ty,tg),(df,dy,dg)):
        assert [r['id'] for r in features]==[r['id'] for r in targets]
        assert sorted(sum(groups,[]))==list(range(len(features)))
        for f,t in zip(features,targets):
            raw=f['prompt'].encode();roles=t['meaning']['roles']
            assert set(f)=={'id','prompt'}
            for name,(a,b) in zip(('primary','recipient','theme'),t['spans']):
                if roles[name] is None:assert (a,b)==(-1,-1)
                else:assert 0<=a<b<=len(raw) and raw[a:b].decode()==roles[name] and raw[:a].decode() is not None and raw[b:].decode() is not None
            assert t['meaning']['relation'] in (0,1,2) and t['meaning']['polarity'] in (0,1) and 0<=t['meaning']['quantity']<=24
        for indices in groups:
            a,b,c,d=(targets[i] for i in indices)
            assert a['meaning']==b['meaning']==c['meaning'] and a['meaning']!=d['meaning']
    assert not {r['prompt'] for r in tf}&{r['prompt'] for r in df+old}
    assert not {r['meaning_id'] for r in ty}&{r['meaning_id'] for r in dy}
    assert {r['family'] for r in ty}=={0,1,2} and {r['family'] for r in dy}=={3,4,5}
    orders=read('orders')
    for epoch in range(8):
        batches=orders[epoch*72:(epoch+1)*72]
        assert sorted(sum(batches,[]))==list(range(1152))
        for batch in batches:
            for group in tg:
                overlap=set(batch)&set(group)
                assert not overlap or overlap==set(group)
    original=json.loads((ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu/train.json').read_text(encoding='utf-8-sig'))
    byid={r['id']:r for r in original};windows=read('OWT_train_reconstruction_windows')
    for row in windows:
        original_id=row['id'].rsplit(':',1)[0];source=byid[original_id]['prompt'].encode()
        assert row['prompt'].encode()==source[row['source_window_byte_start']:row['source_window_byte_end']]
    report=dict(all1408_gold_spans_UTF8_independent=True,all352_group_facts_equivalence_contrast=True,
        train_dev_literals_and_meanings_disjoint=True,V58_literal_train_overlap=0,held_template_families=True,
        all8_epoch_orders_once_per_row_group_intact=True,source_features_no_gold=True,
        duplicate_TRAIN_prompts=len(tf)-len({r['prompt'] for r in tf}),duplicate_DEV_prompts=len(df)-len({r['prompt'] for r in df}),
        quantity_labels_TRAIN=sorted({r['meaning']['quantity'] for r in ty}),
        OWT_windows_reconstructed=len(windows),OWT_short_windows=sum(r['selection']=='short' for r in windows),
        OWT_windows_not_semantic_gold=True,model_forward=0)
    (data/'root_data_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
