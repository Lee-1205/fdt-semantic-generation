"""Compare identical V46/V47 DEV targets by position using saved outputs."""
import statistics
from pathlib import Path
import torch

from codec_resume_v47_contract import PARENT, RUN, read
from learning_io_v2 import atomic_json


def compare_teacher(tensors):
    targets = tensors['target_ids']
    logits = tensors['normal'].float()
    active = targets != 257
    pos = torch.arange(targets.shape[1])[None]
    prediction = logits.argmax(-1)
    nll = logits.logsumexp(-1)-logits.gather(-1,targets[...,None]).squeeze(-1)
    result = {}
    for name, mask in [('first_byte',active & (pos==0)), ('first16',active & (pos<16)),
                       ('after16',active & (pos>=16))]:
        result[name] = dict(accuracy=float((prediction[mask]==targets[mask]).float().mean()),
                            NLL=float(nll[mask].double().mean()), symbols=int(mask.sum()))
    result['all_symbols_correct_rows'] = int(((prediction==targets) | ~active).all(1).sum())
    return result


def compare_BOS(path):
    rows = read(path)
    scores = []
    for row in rows:
        expected = list(row['source'].encode('utf-8'))+[256]
        produced = row['generated']['token_ids']
        first = next((i for i,symbol in enumerate(expected) if i>=len(produced) or produced[i]!=symbol),len(expected))
        scores.append((row['id'], first, len(expected)-1))
    return dict(rows=len(scores), mean_correct_prefix=statistics.mean(score[1] for score in scores),
                median_correct_prefix=statistics.median(score[1] for score in scores),
                first_byte_correct=sum(score[1]>0 for score in scores),
                first16_correct=sum(score[1]>=16 for score in scores),
                per_row=[dict(id=id,correct_prefix=n,source_bytes=size) for id,n,size in scores])


def main():
    torch.set_num_threads(2)
    parent = torch.load(PARENT/'held_final_logits.pt', map_location='cpu', weights_only=True)
    current = torch.load(RUN/'held_final_logits.pt', map_location='cpu', weights_only=True)
    assert torch.equal(parent['target_ids'],current['target_ids'])
    before = compare_BOS(PARENT/'held64_reconstruction.json')
    after = compare_BOS(RUN/'held64_reconstruction.json')
    assert [x['id'] for x in before['per_row']] == [x['id'] for x in after['per_row']]
    report = dict(V46_teacher=compare_teacher(parent), V47_teacher=compare_teacher(current),
                  V46_BOS64=before,V47_BOS64=after,
                  V47_repaired_BOS64=compare_BOS(RUN/'held64_boundary_repaired.json'))
    atomic_json(RUN/'root_first_byte_comparison.json',report)
    print({name:{k:v for k,v in value.items() if k!='per_row'} for name,value in report.items()})


if __name__ == '__main__':
    main()
