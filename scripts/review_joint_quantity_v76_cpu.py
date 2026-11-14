"""Independent whole-clause/quartet/finite-order review plus saved-history replay."""
import collections,hashlib,itertools,json,random,re,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
D=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1'
def SHA(p):return hashlib.sha256(p.read_bytes()).hexdigest()
U=r'kg|degrees C|km'
PATTERNS=[
 r"For entry (?P<episode>\d+), (?P<actor>\w+) registered a (?P<kind>mass change|temperature change|displacement change) of (?P<A>[+-]\d+\.75 (?P<unit>kg|degrees C|km)); (?P<peer>\w+)'s (?P<predicate>separate stock weighed|separate chamber measured|separate route spanned) (?P<B>\+\d+\.25 (?P=unit))\.",
 r"For entry (?P<episode>\d+), (?P<peer>\w+)'s (?P<predicate>separate stock weighed|separate chamber measured|separate route spanned) (?P<B>\+\d+\.25 (?P<unit>kg|degrees C|km)); (?P<actor>\w+) registered a (?P<kind>mass change|temperature change|displacement change) of (?P<A>[+-]\d+\.75 (?P=unit))\. The two records describe separate quantities\.",
 r"For entry (?P<episode>\d+), (?P<actor>\w+) shipped (?P<A>\d+) steel rings; (?P<peer>\w+) kept (?P<B>\d+) steel rings\.",
 r"For entry (?P<episode>\d+), (?P<peer>\w+) kept (?P<B>\d+) steel rings; (?P<actor>\w+) shipped (?P<A>\d+) steel rings\. The two records describe separate quantities\."
]
def main():
    torch.set_num_threads(2)
    rows=io.load_json(D/'train_features.json');targets=io.load_json(D/'train_targets.json');orders=io.load_json(D/'orders.json');m=io.load_json(D/'manifest.json')
    assert len(rows)==len(targets)==len(orders)==1024 and sorted(o[0] for o in orders)==list(range(1024)) and all(len(o)==1 for o in orders)
    groups=collections.defaultdict(list);cells=collections.defaultdict(set);where={o[0]:i for i,o in enumerate(orders)}
    ending=collections.Counter();spancount=collections.Counter()
    kinds={'kg':('mass change','separate stock weighed'),'degrees C':('temperature change','separate chamber measured'),'km':('displacement change','separate route spanned')}
    for i,(r,t) in enumerate(zip(rows,targets)):
        assert set(r)=={'id','prompt'} and r['id']==t['id'];raw=r['prompt'].encode();previous=-1
        for s in t['critical_spans']:
            a,b=s['start'],s['end'];assert previous<=a<b<=len(raw) and raw[a:b].decode()==s['text'];previous=b;spancount[s['kind']]+=1
        if i>=896:continue
        assert all(ord(c)<128 for c in r['prompt'])
        assert raw[t['clause_start']:].decode()==t['clause']
        matches=[(n,q) for n,pat in enumerate(PATTERNS) if (q:=re.fullmatch(pat,t['clause']))]
        assert len(matches)==1;(pat,match)=matches[0];g=match.groupdict()
        assert t['layout']==('end' if pat in (0,2) else 'middle')
        assert int(g['episode'])==t['episode'] and g['actor']==t['actor'] and g['peer']==t['peer']
        if pat<2:
            assert t['mode']=='signed' and t['unit']==g['unit']
            assert (g['kind'],g['predicate'])==kinds[g['unit']]
            assert g['A']==f"{t['sign']}{t['A']}.75 {t['unit']}" and g['B']==f"+{t['B']}.25 {t['unit']}"
            ending[t['unit'],t['layout']]+=1
        else:
            assert t['mode']=='unsigned' and t['unit']=='items'
            assert int(g['A'])==t['A'] and int(g['B'])==t['B']
        for name in ('A','B'):
            span=t[name+'_span'];assert span['text']==g[name]
            assert span['start']==t['clause_start']+match.start(name) and span['end']==t['clause_start']+match.end(name)
            assert raw[span['start']:span['end']].decode()==span['text']
        names={(x.start(),x.end(),x.group()) for x in re.finditer(r'\b(?:'+'|'.join(m['TRAIN_names'])+r')\b',r['prompt'])}
        assert names=={(s['start'],s['end'],s['text']) for s in t['critical_spans'] if s['kind']=='name'}
        numbers={(x.start(),x.end(),x.group()) for x in re.finditer(r'[+-]?\d+(?:\.\d+)?(?: degrees C| kg| km)?',r['prompt'])}
        assert numbers=={(s['start'],s['end'],s['text']) for s in t['critical_spans'] if s['kind']=='number_expression'}
        assert len(raw)%16==t['EOS_position']
        cells[t['unit'],t['layout'],t['sign']].add(t['EOS_position']);groups[t['quartet_id']].append(i)
    assert len(groups)==224 and len(cells)==14 and all(s==set(range(16)) for s in cells.values())
    for indices in groups.values():
        assert len(indices)==4 and {tuple(targets[i]['member']) for i in indices}==set(itertools.product(range(2),repeat=2))
        positions=sorted(where[i] for i in indices);assert positions==list(range(positions[0],positions[0]+4))
        norm=[];q={}
        for i in indices:
            t=targets[i];raw=rows[i]['prompt']
            for name in sorted(('A','B'),key=lambda n:t[n+'_span']['start'],reverse=True):
                s=t[name+'_span'];raw=raw[:s['start']]+'<'+name+'>'+raw[s['end']:]
            norm.append(raw);q[tuple(t['member'])]=t
        assert len(set(norm))==1
        for i,j in itertools.product(range(2),repeat=2):
            assert q[i,j]['A']==q[0,0]['A']+i and q[i,j]['B']==q[0,0]['B']+j
    # JSON next-cursor is enough for this DATA order only; not worker/scaler recovery.
    for cursor in (0,1,3,4,895,896,1023,1024):
        state=json.loads(json.dumps(dict(cursor=cursor,order=orders)))
        assert state['order'][state['cursor']:]==orders[cursor:]
    old=ROOT/'research/artifacts/critical_retention_20261112_v75_cpu'
    assert rows[896:]==io.load_json(old/'train_features.json')[512:] and targets[896:]==io.load_json(old/'train_targets.json')[512:]
    for filename,expected in m['held_exclusion_files_SHA'].items():
        file=Path(filename);assert SHA(file)==expected
        assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in io.load_json(file) for r in rows)
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in rows for n in m['forbidden_observed_held_names'])
    assert sum(len(r['prompt'].encode()) for r in rows)==m['source_UTF8_bytes']==1010943
    assert sum(s['end']-s['start'] for t in targets for s in t['critical_spans'])==m['critical_bytes']==45460
    # Replay scalar weights at SAVED V75 losses, not gradients/model updates.
    env=torch.load(ROOT/'runs/critical_retention_20261112_v75/model/runtime.pt',weights_only=True,map_location='cpu')
    hist=env['runtime']['history'];assert len(hist)==640
    oldwrong=sum(.25*h['wrong_critical_CE'] for h in hist);oldtotal=sum(h['loss'] for h in hist);scaled=[]
    for h in hist:
        c,w=h['critical_positions'],h['wrong_critical_positions']
        factor=w/max(w,c/8) if w else 0.
        scaled.append(.25*h['wrong_critical_CE']*factor)
    newwrong=sum(scaled);replayed=oldtotal-oldwrong+newwrong
    replay=dict(source='saved V75 losses, unchanged logits and history',old_wrong_loss_fraction=oldwrong/oldtotal,
        hypothetical_capped_wrong_loss_fraction=newwrong/replayed,old_wrong_term_sum=oldwrong,hypothetical_wrong_term_sum=newwrong,
        max_relative_position_weight=8.,rows_with_wrong=sum(h['wrong_critical_positions']>0 for h in hist),
        no_parameter_gradient_or_accuracy_prediction=True,no_model_forward_optimizer=True)
    proof=dict(status='ROOT_V76_DATA_AND_CAPPED_LOSS_CPU_REVIEW_PASS',all1024_features_targets_spans_checked=True,
        all896_authored_clauses_independently_parsed=True,all224_quartets_only_two_values_change=True,
        independent_two_by_two_facts_not_one_sign_pair=True,unit_layout_sign_EOS_cross_cells=len(cells),
        all16_EOS_roles_in_every_cell=True,unit_layout_counts={str(k):v for k,v in ending.items()},
        original_OWT128_exact=True,finite1024_order_quartet_adjacency_and_JSON_nextcursor=True,
        observed_held_prompts_names_not_TRAIN=True,manifest_FILE_SHA=SHA(D/'manifest.json'),
        files_SHA={f.name:SHA(f) for f in D.glob('*.json')},focused_loss_CPU_tests=7,
        loss_replay=replay,model_forward_calls=0,optimizer_steps=0,CUDA_initialized=torch.cuda.is_initialized(),
        actual_worker_full_restore_or_plan_ready=False,actual_performance_improvement=False)
    assert not proof['CUDA_initialized']
    io.atomic_json(D/'root_CPU_review.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='files_SHA'}))
if __name__=='__main__':main()
