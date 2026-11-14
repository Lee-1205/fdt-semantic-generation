"""Separate source span, finite exposure and held-data audit; no NN forwards."""
import collections,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from research.fuzzy_factual_objective_v70 import training_weights

def main():
    data=ROOT/'research/artifacts/factual_cycle_20261112_v70_cpu_v2';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    rows=read(data/'train_features.json');targets=read(data/'train_targets.json');dev=read(data/'dev_features.json');manifest=read(data/'manifest.json');orders=read(data/'orders.json')
    assert len(rows)==len(targets)==512 and sorted(i[0] for i in orders)==list(range(512)) and all(len(i)==1 for i in orders)
    role=collections.Counter();counts=collections.Counter()
    for i,(row,target) in enumerate(zip(rows,targets)):
        assert set(row)=={'id','prompt'} and row['id']==target['id'];raw=row['prompt'].encode();seen=set()
        for span in target['critical_spans']:
            a,b=span['start'],span['end'];assert 0<=a<b<=len(raw) and raw[a:b].decode()==span['text'] and not seen&set(range(a,b));seen.update(range(a,b))
            counts[span['kind']]+=b-a
            for pos in range(a,b):role[pos%16]+=1
        if i<384:
            assert all(s['text'].isdigit() or s['text'] in manifest['TRAIN_names'] for s in target['critical_spans'])
            for event in target['authored_events']:
                text=raw[event['byte_start']:event['byte_end']].decode()
                assert all(str(event[k]) in text for k in ('actor','recipient','quantity','reserve','item'))
                assert 'temporal_order' not in event or event['temporal_order'].lower() in text.lower()
    assert set(role)==set(range(16)) and sum(role.values())==manifest['critical_bytes']==38480
    assert sum(len(r['prompt'].encode()) for r in rows)==940237 and max(len(r['prompt'].encode()) for r in rows)==16384
    forbidden=read(ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json')+read(ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu/features.json')
    assert not any(x['prompt'] in r['prompt'] for x in forbidden for r in rows)
    owt=json.loads((ROOT/'research/artifacts/english_memory_cycle_20261112_v68_cpu/OWT_features.json').read_text());assert rows[384:]==owt
    olddev=read(ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu/dev.json')
    canonical=lambda x:re.search(r'OWT:(?:train|validation):[0-9]+',x['id']).group()
    assert not {canonical(x) for x in owt}&{canonical(x) for x in olddev}
    assert not {r['prompt'] for r in rows}&{r['prompt'] for r in dev}
    held_name_hits={name:sum(bool(re.search(r'\b'+name+r'\b',r['prompt'])) for r in rows) for name in manifest['DEV_names']}
    lookup,objective=training_weights(rows)
    # Retention is the same observed English-only panel; Korean remains deferred.
    english=[r for r in forbidden[:52] if r['language']=='English'];assert len(english)==36
    retained=data/'retained_English.json';assert not retained.exists();retained.write_text(json.dumps(english,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    proof=dict(status='ROOT_V70_DATA_REVIEW_PASS',rows512_onefinite_epoch=True,target_bytes=940237,critical_bytes=38480,critical_role_positions=dict(role),
        critical_kinds=dict(counts),new_DEV_name_hits_in_entire_TRAIN=held_name_hits,objective_report=objective,
        features_id_prompt_only=True,goldspans_loss_only=True,V66_V69_never_TRAIN=True,OWT_original_source_and_split_exact=True,
        metadata_revision='rev1 kept unused temporal order for clauses without temporal assertion; rev2 omits that label, inputs identical; both originals and builder snapshot preserved; no GPU in either prep',
        prior_research=['https://aclanthology.org/2021.eacl-main.235/','https://arxiv.org/abs/2411.02083'],
        interpretation='papers motivate inspecting entities/numbers separately, not proof of chosen .25 weight; no new auxiliary head. Number proximity alone is insufficient for exact source quantities; apply categorical content CE through actual codec.',
        E2_registry_access=0,new_model_forwards=0)
    out=data/'root_data_review.json';assert not out.exists();out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
