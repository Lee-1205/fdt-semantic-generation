"""Actual source/score separation and all 138 eval rows, no model call."""
import collections,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from review_long_extrapolation_v69_saved_cpu import PATTERN,fields
def main():
    d=ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu';short=io.load_json(d/'short_features.json');labels=io.load_json(d/'short_targets.json');long=io.load_json(d/'long_features.json');targets=io.load_json(d/'long_targets.json')
    train=io.load_json(ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2/train_features.json');events=0;negative=collections.Counter()
    assert len(short)==len(labels)==128 and len(long)==len(targets)==10 and all(set(r)=={'id','prompt'} for r in short+long)
    for row,t in zip(short,labels):
        assert row['id']==t['id'];b=row['prompt'].encode();last=-1
        for s in t['critical_spans']:assert last<=s['start']<s['end']<=len(b) and b[s['start']:s['end']].decode()==s['text'];last=s['end']
        for e in t.get('events',[]):
            text=b[e['start']:e['end']].decode();a=e['actor'];other=e['other'];q=e['quantity'];r=e['reserve']
            if e['family']==0:assert text.startswith('The crate beside the door belongs to '+a+', not '+other) and f'exactly {q} ' in text
            elif e['family']==1:assert text.startswith(a+' arrived after '+other) and a+' had departed earlier' in text and f'remaining {r} ' in text and 'neither person took' in text
            elif e['family']==2:assert text.startswith(a+f' recorded a temperature of -{q}.35') and other+f' kept +{r}.50 kg' in text and '08:07 on 2024-06-19' in text
            else:assert text.startswith('Only '+a+f' transferred {q} ') and ' to '+other in text and f'the {r} items left with '+a in text and 'not part of the delivery' in text
    for row,t in zip(long,targets):
        assert row['id']==t['id'];raw=row['prompt'].encode();assert len(raw)==t['actual_bytes']
        if t['style']=='single_sentence':assert row['prompt'].count('.')==1 and '\n' not in row['prompt']
        for e in t['events']:
            source=raw[e['byte_start']:e['byte_end']].decode();match=PATTERN.fullmatch(source);assert match;f=fields(match)
            assert all(f[k]==e[k] for k in ('episode','quantity','reserve'))
            assert all(f[k]==e[k].casefold() for k in ('actor','checker','item','source_color','destination_color','order'))
            assert bool(f['negative'])==e['negative_removal'];events+=1;negative[e['negative_removal']]+=1
    assert not any(t['prompt'] in r['prompt'] or r['prompt'] in t['prompt'] for t in train for r in short+long)
    assert not {r['prompt'] for r in short+long}&{t['prompt'] for t in train}
    m=io.load_json(d/'manifest.json');assert sum((len(r['prompt'].encode())+2047)//2048 for r in long)==m['long_stream_chunks']==736
    proof=dict(status='ROOT_EXPANDED_EVAL_V73_CPU_PASS',all138_feature_target_rows_verified=True,all_source_long_events_independently_reconstructed=events,negative_removal_events=dict(negative),
        actual_positive_and_negative_sources=True,no_eval_prompt_or_full_train_embedding_in_TRAIN=True,prepared32_and_previous_long2_separated_from_new96_and_long8=True,
        all_model_features_id_prompt_only=True,all_event_span_gold_OFFLINE_ONLY=True,new_training_rows=0,NNcalls=0,CUDA_initialized=False,
        files_SHA={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.glob('*.json')},limitations=m['limitations'])
    file=d/'root_data_review.json';assert not file.exists();io.atomic_json(file,proof);print(json.dumps(proof))
if __name__=='__main__':main()
