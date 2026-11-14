"""Historical TRAIN teacher coverage audit only; no NN forward or data expansion."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.train_critical_frontier_v84 import select

def main():
    torch.set_num_threads(4);run=ROOT/'runs/update_retention_20261113_v80';p=io.load_json(run/'plan.json');envelope=torch.load(run/'model/runtime.pt',map_location='cpu',weights_only=True);state=envelope['runtime']
    data=Path(p['data']);features=io.load_json(data/'train_features.json');labels=io.load_json(data/'train_targets.json');ids={r['id'] for r in features};anchors=set(state['anchor_indices']);selected=[];ct=collections.Counter();pins={}
    assert len(features)==len(state['cache_refs'])==len(ids)==1024
    for i,ref in enumerate(state['cache_refs']):
        file=Path(ref['targets']);sha=support.sha(file);assert sha==ref['targets_SHA'];pins[str(file)]=sha;t=torch.load(file,map_location='cpu',weights_only=True);row=features[i];raw=row['prompt'].encode();y=t['labels'];source=torch.full_like(y.flatten(),257);source[:len(raw)]=torch.tensor(list(raw));source[len(raw)]=256
        assert t['source_id']==row['id']==ref['source_id'] and torch.equal(y.flatten(),source)
        # Independent reconstruction from authored TRAIN critical spans.
        expected=torch.zeros(y.numel(),dtype=torch.bool)
        for span in labels[i]['critical_spans']:expected[span['start']:span['end']]=True
        assert torch.equal(t['critical_mask'].flatten(),expected)
        item=select(t,row['id'],ids,precision_policy='historical_V75_BF16_teacher',expected_precision_policy='historical_V75_BF16_teacher',max_margin=1.,max_patches=512)
        weak=t['critical_mask']&t['teacher_correct']&(t['teacher_margin']<=1.)&(y!=257)
        assert item['critical_positions']==weak.nonzero().tolist() and item['patch_indices']==weak.any(-1).nonzero().flatten().tolist()
        n=int(weak.sum());ct['TRAIN_critical_bytes']+=int(expected.sum());ct['initial_critical_wrong']+=item['initial_wrong_critical_bytes'];ct['weak_correct_critical']+=n
        ct['weak_in_old100']+=n if i in anchors else 0;ct['weak_outside_old100']+=n if i not in anchors else 0;ct['sources_with_weak']+=n>0;ct['weak_sources_outside_old100']+=(n>0 and i not in anchors)
        ct['selected_patch_rows']+=len(item['patch_indices'])
        if n:selected.append(dict(item,source_index=i,in_old100=i in anchors))
    assert ct['weak_correct_critical']==213 and ct['weak_in_old100']==24 and ct['weak_outside_old100']==189 and ct['weak_sources_outside_old100']==149
    assert ct['selected_patch_rows']<=512 and not torch.cuda.is_initialized()
    a=ROOT/'research/artifacts/critical_frontier_20261113_v84_cpu_v1';a.mkdir(exist_ok=False)
    io.atomic_json(a/'historical_BF16_TRAIN_selection.json',selected)
    for f in (run/'model/runtime.pt',data/'train_features.json',data/'train_targets.json',ROOT/'research/train_critical_frontier_v84.py',ROOT/'tests/test_train_critical_frontier_v84.py',Path(__file__).resolve()):pins[str(f.resolve())]=support.sha(f)
    io.atomic_json(a/'source_pins.json',pins)
    report=dict(status='ROOT_V84_CPU_CRITICAL_FRONTIER_CONTRACT_READY_NOT_TRAINED',counts=dict(ct),historical_teacher_policy='V75 BF16 at V80 teacher pass',
        total_source_teacher_files_verified=1024,source_and_critical_spans_independently_reconstructed=True,old100_weak_coverage_fraction=24/213,
        new_selector_historical_weak_coverage_fraction=1.,actual_parent_model_forwards=0,optimizer_steps=0,CUDA_initialized=False,selected_feature_bytes_FP32=ct['selected_patch_rows']*512*4,
        all_old_files_preserved=True,new_rows=0,DEV_training=0,E2_access=0,
        caveat='213 are saved BF16 initial correct weak critical byte positions, not independent semantic facts or measured currentFP32 weak positions. Historical selector output is diagnostic only: must rebuild under actual new parent and fullFP32 teacher/prefix/runtime before any training. Strong margins/unseen documents remain outside this guarantee.',
        next='Future guard bank: old TRAIN broad top1 reference plus ALL TRAIN weak critical positions, selected by loss-only labels/margins; features separate. Test under same fullFP32 arithmetic. Do not merely strengthen all ordinary byte margin constraints or replay identical SGD. No actual next driver/plan/seal/launch yet.')
    io.atomic_json(a/'root_cpu_review.json',report);print(json.dumps(report))

if __name__=='__main__':main()
