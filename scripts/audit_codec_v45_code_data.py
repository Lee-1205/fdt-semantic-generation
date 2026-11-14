"""Audit V45 input provenance and the codec's two-vector decoder interface."""
import hashlib
import importlib.util
import json
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from codec_resume_v45_contract import RUN, read
from learning_io_v2 import atomic_json


def key(text):
    return unicodedata.normalize('NFKC', ' '.join(text.split())).casefold()


def inventory(rows):
    prompts = [row['prompt'] for row in rows]
    lengths = sorted(len(s.encode()) for s in prompts)
    return {
        'rows': len(rows), 'bytes_median': statistics.median(lengths),
        'bytes_p10': lengths[len(lengths)//10], 'bytes_p90': lengths[9*len(lengths)//10],
        'non_ascii_rows': sum(not s.isascii() for s in prompts),
        'digit_rows': sum(any(ch.isdigit() for ch in s) for s in prompts),
        'leading_non_alnum_rows': sum(not s[0].isalnum() for s in prompts),
        'terminal_non_sentence_punctuation': sum(s[-1] not in '.!?' for s in prompts),
        'normalized_unique_rows': len(set(map(key, prompts))),
    }


def main():
    plan = read(RUN/'plan.json')
    corpus = Path(plan['artifact'])
    train = Path(plan['training_artifact'])
    v44 = ROOT/'research/artifacts/three_memory_codec_20261111_v44'
    documents, originals, gold = (read(corpus/(name+'.json')) for name in ('documents','features','targets'))
    features, targets, orders = (read(train/(name+'.json')) for name in ('features','targets','orders'))
    parent_train = read(v44/'features.json')[:8192]
    old_docs = {d['id']:d for d in documents['train']}
    dev_docs = {d['id']:d for d in documents['dev']}
    assert not (old_docs.keys() & dev_docs.keys())
    assert not ({d['normalized_sha256'] for d in old_docs.values()} &
                {d['normalized_sha256'] for d in dev_docs.values()})
    for f, t in zip(features, targets, strict=True):
        assert f['id']==t['id'] and f['prompt']==t['target']
        assert t['target_ids']==list(f['prompt'].encode())+[256]
        assert 1<=len(f['prompt'].encode())<=128
    spec = importlib.util.spec_from_file_location('sentence_rules',
        ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23/executed_builder_snapshot.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    old_map = {r['id']:r for r in gold['train']}
    for f, t in zip(originals['train'], gold['train'], strict=True):
        assert f['id']==t['id'] and f['prompt']==t['target'] and t['document_id'] in old_docs
    assert [t['document_id'] for t in gold['train']] == [d['id'] for d in documents['train'] for _ in range(4)]
    for subset, provenance in [(features[:4096], read(train/'fresh_provenance.json')),
                               (parent_train, read(v44/'fresh_provenance.json'))]:
        grouped = {}
        for f, row in zip(subset, provenance, strict=True):
            assert f['id']==row['id']
            doc = row['document_id']
            if doc not in grouped:
                grouped[doc]=list(rules.split_spans(old_docs[doc]['text']))
            assert f['prompt']==grouped[doc][row['span']]
            assert rules.acceptable_new_span(f['prompt'])
    assert {row['id'] for row in features[4096:8192]} <= {r['id'] for r in parent_train}
    assert {old_map[row['id']]['document_id'] for row in features[8192:]} == old_docs.keys()
    dev_keys = {key(row['prompt']) for row in originals['dev']}
    all_dev_sentences = {key(s) for d in dev_docs.values() for s in rules.split_spans(d['text'])}
    new_keys = {key(row['prompt']) for row in features[:8192]}
    assert not (new_keys & all_dev_sentences)
    assert not ({key(row['prompt']) for row in features} & dev_keys)
    assert sorted(i for batch in orders for i in batch)==list(range(len(features)))
    assert all(Counter(0 if i<4096 else 1 if i<8192 else 2 for i in batch)=={0:2,1:2,2:4}
               for batch in orders)
    report = {
        'provenance_verified': len(features), 'train_documents': len(old_docs),
        'dev_documents': len(dev_docs), 'exact_normalized_train_dev_overlap': 0,
        'training': inventory(features), 'dev256': inventory(originals['dev']),
        'held16': inventory([originals['dev'][4*i] for i in range(16)]),
        'fresh': inventory(features[:4096]), 'dataset_parent': str(corpus),
        'source_rows_never_added_to_training': 'development split and E2 final',
    }
    import torch
    from fdt_local.model_masked import Config
    from research.three_memory_codec_v33 import ThreeMemoryCodec, DifferentiableCodecCommit
    from research.text_context_decoder_v2 import BOS
    torch.set_num_threads(3)
    state = torch.load(RUN/'model/checkpoints/state_2048.pt', map_location='cpu', weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = ThreeMemoryCodec(Config(**state['core']))
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval()
    activation = torch.load(RUN/'held_final_logits.pt', map_location='cpu', weights_only=True)['semantic_commit']
    sample = activation[:4].float()
    prefix = torch.full((len(sample), 1), BOS, dtype=torch.long)
    with torch.inference_mode():
        def logits(views):
            return model.decoder(prefix, model.context(DifferentiableCodecCommit.from_semantic(views, (), 'audit')))
        original = logits(sample)
        swapped = logits(sample.flip(1))
        row_swapped = logits(sample.roll(1, 0))
    report['code'] = {
        'commit_shape': list(activation.shape),
        'slot_swap_BOS_logit_max_difference': float((original-swapped).abs().max()),
        'different_sentence_BOS_logit_max_difference': float((original-row_swapped).abs().max()),
        'view_cosine_mean': float(torch.nn.functional.cosine_similarity(activation[:,0],activation[:,1]).mean()),
        'decoder_memory_position_tags': False,
        'checkpoint_modified': False, 'CUDA_initialized': torch.cuda.is_initialized(),
    }
    atomic_json(RUN/'root_code_data_audit_20261111.json',report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
