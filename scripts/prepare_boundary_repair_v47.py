"""Replace suspect abbreviation cuts with complete spans from the same split."""
import hashlib
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import codec_resume_v46_contract as c

OUT = c.ROOT/'research/artifacts/codec_boundary_repair_20261111_v47'
END = re.compile(r'[.!?]["”’)\]]*(?=\s|$)')
CLOSERS = '"”’)]'


def key(text):
    return unicodedata.normalize('NFKC', ' '.join(text.split())).casefold()


def candidates(document, rules):
    text = ' '.join(document['text'].split())
    begin = 0
    for match in END.finditer(text):
        piece = text[begin:match.end()].strip()
        clean_end = piece.rstrip(CLOSERS)
        if rules.abbreviation_end(clean_end):
            continue
        begin = match.end()
        if (rules.acceptable_new_span(clean_end) and len(piece.encode()) <= 128
                and piece.count('“') == piece.count('”')
                and piece.count('"') % 2 == 0
                and piece.count('(') == piece.count(')')):
            yield piece


def save(name, obj):
    with (OUT/name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, separators=(',', ':'))


def replacement(text, document, options, forbidden):
    length = len(text.encode())
    unique = set(options)
    legal = [row for row in unique if key(row) not in forbidden]
    if not legal:
        return None
    chosen = min(legal, key=lambda row:(abs(len(row.encode())-length), row))
    row_id = hashlib.sha256(f"V47:{document}:{chosen}".encode()).hexdigest()
    return {'id':row_id, 'prompt':chosen}


def target(row):
    return {'id':row['id'], 'target':row['prompt'], 'target_ids':list(row['prompt'].encode())+[256]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = c.read(c.RUN/'plan.json')
    corpus = Path(plan['artifact'])
    docs = c.read(corpus/'documents.json')
    original = c.read(corpus/'targets.json')
    dev = c.read(corpus/'features.json')['dev']
    original_map = {r['id']:r['document_id'] for split in original.values() for r in split}
    spec = importlib.util.spec_from_file_location('sentence_rules',
        c.ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23/executed_builder_snapshot.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    audit = c.read(c.RUN/'root_position_boundary_audit.json')
    train_issues = {r['id'] for r in audit['abbreviation_tail_candidates']['original_replay']}
    dev_issues = {r['id'] for r in audit['abbreviation_tail_candidates']['BOS64']}
    train = c.read(c.TRAIN_DATA/'features.json')
    train_sources = {d['id']:d for d in docs['train']}
    dev_sources = {d['id']:d for d in docs['dev']}
    dev_keys = {key(r['prompt']) for r in dev}
    for d in dev_sources.values():
        dev_keys.update(key(s) for s in rules.split_spans(d['text']))
        dev_keys.update(key(s) for s in candidates(d,rules))
    used = {key(r['prompt']) for r in train}
    counts = Counter(key(r['prompt']) for r in train)
    edits, seen = [], set()
    dev_overlap_before = sum(key(r['prompt']) in dev_keys for r in train)
    fallback = ((doc['id'], span) for doc in docs['train'] for span in candidates(doc,rules))
    for index,row in enumerate(train):
        k = key(row['prompt'])
        duplicate = k in seen
        seen.add(k)
        overlap = k in dev_keys
        if row['id'] not in train_issues and not duplicate and not overlap:
            continue
        document = original_map.get(row['id'])
        if document not in train_sources:
            raise ValueError('replacement must remain in the original TRAIN document')
        new = replacement(row['prompt'], document, candidates(train_sources[document],rules), used | dev_keys)
        source = document
        if new is None:
            source, span = next((d,s) for d,s in fallback if key(s) not in used and key(s) not in dev_keys)
            new = replacement(row['prompt'], source, (span,), ())
        reason = ('overlap with a sentence in a DEV document' if overlap else
                  'normalized duplicate' if duplicate else 'protected abbreviation before continuation')
        edits.append({'index':index,'old_id':row['id'],'new_id':new['id'],'old_document':document,
                      'new_document':source,'reason':reason})
        train[index] = new
        used.add(key(new['prompt']))
    assert not ({key(r['prompt']) for r in train} & dev_keys)
    held = [dev[i] for i in plan['full_BOS_indices']]
    held_edits = []
    forbidden = {key(r['prompt']) for r in train+dev}
    for index,row in enumerate(held):
        if row['id'] not in dev_issues:
            continue
        document = original_map[row['id']]
        new = replacement(row['prompt'],document,candidates(dev_sources[document],rules),forbidden)
        if new is None:
            continue
        held_edits.append({'index':index,'old_id':row['id'],'new_id':new['id'],'document':document})
        held[index] = new
        forbidden.add(key(new['prompt']))
    OUT.mkdir()
    save('train_features.json',train)
    save('train_targets.json',[target(r) for r in train])
    save('held64_features.json',held)
    save('held64_targets.json',[target(r) for r in held])
    save('manifest.json',dict(train_edits=edits,held64_edits=held_edits,
        dev_document_sentence_overlap_before=dev_overlap_before,
        original_audit_SHA256=c.sha256(c.RUN/'root_position_boundary_audit.json'),
        old_comparison_panels_preserved=True,train_rows=len(train),held_documents=len(held),
        remaining_normalized_train_duplicates=len(train)-len({key(r['prompt']) for r in train}),
        selection='Heuristic sentence boundaries, same-document replacements when available.'))
    print(json.dumps(dict(train_replacements=len(edits),dev_overlap_before=dev_overlap_before,
                          held_replacements=len(held_edits),held_candidates=len(dev_issues),
                          initial_duplicate_rows=sum(n-1 for n in counts.values())),indent=2))


if __name__=='__main__':
    main()
