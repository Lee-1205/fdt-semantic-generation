"""Independent saved-tensor distances and literal intervention readings, no NN."""
import collections,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(4);run=ROOT/'runs/semantic_controls_20261111_v58';p=io.load_json(run/'plan.json')
    terminal=io.load_json(run/'final_status.json');native=io.load_json(run/'root_terminal_native.json')
    assert terminal['status']=='COMPLETE' and terminal['exit']['all_owned_processes_gone'] and native['fresh_python_count']==0
    assert support.sha(p['source'])==p['source_sha'] and support.sha(p['head_source'])==p['head_sha']
    data=Path(p['data']);features=io.load_json(data/'features.json');targets=io.load_json(data/'targets.json');groups=io.load_json(data/'groups.json')
    byid={r['id']:r for r in features};gold={r['id']:r for r in targets}
    packet=torch.load(run/'raw_semantic_packets.pt',weights_only=True,map_location='cpu');saved=io.load_json(run/'semantic_distances.json')
    def cosine_distance(a,b):
        x,y=a.double().flatten(),b.double().flatten();return 1-float(torch.dot(x,y)/(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y)))
    ranks={}
    for kind in ('roles','negation','quantity'):
        ranks[kind]={}
        for space in next(iter(packet.values()))['spaces']:
            para=layout=0
            for i,g in enumerate(g for g in groups if g['kind']==kind):
                a,b,c,d=(packet[r]['spaces'][space] for r in g['ids'])
                distances=[cosine_distance(a,x) for x in (b,c,d)]
                actual=saved[kind][space]['rows'][i]
                for value,key in zip(distances,('paraphrase','changed','layout')):assert abs(value-actual[key])<1e-8
                def rank(x,y):return .5 if abs(x-y)<1e-8 else float(x<y)
                para+=rank(distances[0],distances[1]);layout+=rank(distances[2],distances[1])
                assert gold[g['ids'][0]]['meaning']==gold[g['ids'][1]]['meaning']==gold[g['ids'][3]]['meaning']
                assert gold[g['ids'][0]]['meaning']!=gold[g['ids'][2]]['meaning']
                assert set(byid[g['ids'][0]])=={'id','prompt'}
            ranks[kind][space]=dict(paraphrase_closer=para/16,layout_closer=layout/16)
    interventions=io.load_json(run/'interventions.json');group_by_id={g['id']:g for g in groups};counts={}
    for condition in ('normal','swap_global','swap_local'):
        records=[r for r in interventions if r['condition']==condition];canonical=changed=exact=EOS=0
        for row in records:
            g=group_by_id[row['group']];a=gold[g['ids'][0]]['meaning'];b=gold[g['ids'][2]]['meaning']
            match=re.match(r'^([A-Za-z]+) gave ([A-Za-z]+) a (book|cup|key|ball)\.',row['output'])
            if match:
                frame=dict(agent=match[1],recipient=match[2],object=match[3],polarity=1)
                canonical+=int(frame==a);changed+=int(frame==b)
            exact+=int(row['output']==byid[g['ids'][0]]['prompt'] or row['output']==byid[g['ids'][2]]['prompt']);EOS+=row['EOS']
        counts[condition]=dict(canonical_first_clause=canonical,reversed_first_clause=changed,full_sentence_exact=exact,EOS=EOS,total=len(records))
    train=io.load_json(ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu/train.json');lengths=[len(r['prompt'].encode()) for r in train]
    audit=dict(TRAIN_min_UTF8_bytes=min(lengths),TRAIN_max_UTF8_bytes=max(lengths),TRAIN_under42=sum(n<42 for n in lengths),
        diagnostic_max_UTF8_bytes=max(len(r['prompt'].encode()) for r in features),diagnostic_rows=192,
        duplicate_diagnostic_rows=192-len({r['prompt'] for r in features}),score_is_outside_new_head_short_EOS_training_coverage=True)
    report=dict(status='INDEPENDENT_V58_SAVED_REVIEW_PASS',raw_tensor_distances_recomputed=True,all48_fact_equivalence_contrasts_checked=True,
        inference_feature_keys_id_prompt_only=True,source_head_parent_SHA_unchanged=True,raw_packet_sha=support.sha(run/'raw_semantic_packets.pt'),
        ranks=ranks,intervention_first_clause_score_only=counts,length_coverage_audit=audit,
        important='Current ordered roles are byte positions, not agent/patient semantic roles. core.value gets16 zero role inputs in current codec.',
        interpretation='No positive semantic-invariance evidence under these controls; literal local code dominates readable first clauses. This does not prove all semantic information absent.',
        limitations=['48correlated authored groups/47duplicate rows','short41B controls vs new head TRAIN minimum658B','cosine can miss nonlinear/distributed semantics',
            'global2 is a selected branch, not a proven unique semantic variable','first-clause scoring ignores garbage suffix and is not full correct reconstruction'],
        extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(run/'root_saved_review.json',report);print(json.dumps(report))
if __name__=='__main__':main()
