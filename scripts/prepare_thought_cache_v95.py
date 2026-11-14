"""Source-only frozen cache purpose, independent data review, actual preflight."""
import argparse,collections,itertools,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_thought_cache_v95 import validate,SCHEMA,OUTPUT,START,SPLITS
from research.thought_cache_v95 import assemble_input

A=ROOT/'research/artifacts/thought_cache_20261113_v95_cpu_v1';PLAN=A/'plan_unsealed.json'
WORKER=ROOT/'scripts/run_thought_cache_v95.py';OWNER=ROOT/'scripts/supervise_thought_cache_v95.py'


def independent_data(data):
    # Independently compare explicit orders and equality, not the builder's
    # integer-world function. Gold information is loss-side ONLY.
    union=set();ans={};stats={}
    for split in SPLITS:
        ff=io.load_json(data/(split+'_features.json'));tt=io.load_json(data/(split+'_targets.json'))
        assert len(ff)==len(tt)==(384 if split=='train' else 32)
        pairs=collections.defaultdict(list);labels=collections.Counter()
        for f,t in zip(ff,tt):
            assert set(f)=={'id','prompt'} and f['id']==t['id']==support.hashlib.sha256(f['prompt'].encode()).hexdigest()
            assert f['id'] not in union;union.add(f['id'])
            op=t['source_fact']['operator'];possibilities={'greater':{1},'less':{-1},'not greater':{-1,0},'not less':{0,1}}[op]
            outcomes={-x if t['reverse'] else x for x in possibilities}
            label=('greater' if outcomes=={1} else 'less' if outcomes=={-1} else 'unknown')
            a,b=t['source_fact']['a'],t['source_fact']['b']
            if t['reverse']:a,b=b,a
            text='It is unknown.' if label=='unknown' else a+' is '+('larger' if label=='greater' else 'smaller')+' than '+b+'.'
            assert t['label']==label and t['answer']==text
            ans[support.hashlib.sha256(text.encode()).hexdigest()]=text;labels[label]+=1;pairs[t['pair_key']].append(f['id'])
        assert all(len(x)==2 for x in pairs.values())
        stats[split]=dict(rows=len(ff),labels=dict(labels))
        if split=='train':train={f['id'] for f in ff};train_pairs={frozenset(v) for v in pairs.values()}
    assert ans==io.load_json(data/'answers_loss_only.json') and len(ans)==31
    for order in io.load_json(data/'four_finite_orders.json'):
        assert len(order)==192 and {frozenset(v) for v in order}==train_pairs
        assert {v for pair in order for v in pair}==train and len([v for pair in order for v in pair])==384
    return dict(rows480=len(union)==480,answers31=True,orders4=True,feature_fields_only_id_prompt=True,finite_relation_negation_checked=True,statistics=stats)


def main(mode):
    if mode=='prepare':
        A.mkdir(exist_ok=False)
        old=(ROOT/'scripts/supervise_missing_assessment_v93.py').read_text(encoding='utf-8')
        prefix,suffix=old.split("    state=dict(status='PREPARING'",1)
        prefix=prefix.replace('run_missing_assessment_v93.py','run_thought_cache_v95.py')
        start=prefix.index("    if (p['schema']");end=prefix.index('    for name,digest',start)
        prefix=prefix[:start]+"    if (p['schema']!='"+SCHEMA+"' or p['wall_seconds']!=300 or p['max_calls']!=542 or p['expected_input_rows']!=480 or p['expected_answers']!=31 or p['max_output_bytes']!=128*1024**2):\n        raise ValueError('fixed source-cache542 NOoptimizer NOthought scope')\n"+prefix[end:]
        prefix=prefix.replace("p['deadline']-p['start']!=180","p['deadline']-p['start']!=300").replace("p['cutoff']!=p['deadline']-45","p['cutoff']!=p['deadline']-60")
        prefix=prefix.replace('108*1024**3+32*1024**2','108*1024**3+128*1024**2').replace('108.03125GiB start:100 protected+32MiB owned+8 headroom','108.125GiB start:100 protected+128MiB owned+8 headroom')
        with OWNER.open('x',encoding='utf-8',newline='\n') as f:f.write(prefix+"    state=dict(status='PREPARING'"+suffix)
        selection=ROOT/'configs/frozen_thought_codec_20261113_v94.json';s=io.load_json(selection)
        data=ROOT/'research/artifacts/memory_thought_data_20261113_v95_cpu_v1'
        io.atomic_json(A/'root_independent_data.json',independent_data(data))
        # Code-only closure, runtime+mandatory standalone and this task's data.
        # Do not rehash2453 irrelevant historical raw files or rerun baselines.
        files=[f for folder in ('research','scripts','src/fdt_local') for f in (ROOT/folder).glob('*.py')]
        files+=list(data.glob('*.json'))+[selection,Path(s['selected_runtime']),Path(s['mandatory_standalone_base']),Path(s['selected_runtime']).with_suffix('.json'),A/'root_independent_data.json']
        pins={str(f.resolve()):support.sha(f) for f in files}
        p=dict(schema=SCHEMA,authorized=False,root=str(ROOT),output=str(ROOT/'runs/thought_cache_20261113_v95'),selection=str(selection),source_codec_SHA=s['selected_SHA'],data=str(data),
            files=pins,max_calls=542,expected_input_rows=480,expected_answers=31,max_output_bytes=OUTPUT,start_free_bytes=START,wall_seconds=300,optimizer_steps=0,denoise_calls=0,answer_capacity=33,
            purpose='ONE readonly frozenE(X)+source-only3memory cache; separate LOSS_ONLY E(answer) and D target-ceiling. NOT thought learning. No source gold condition, ExactNULL.')
        io.atomic_json(PLAN,p);validate(PLAN,Path(p['output']));print(json.dumps(dict(unsealed=str(PLAN),pins=len(pins),actual_GPU=0)));return
    p,_,_=validate(PLAN,Path(io.load_json(PLAN)['output']));out=Path(p['output'])
    pre=io.load_json(A/'cpu_preflight_owned/final_status.json')
    assert pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['worker_exit_code']==0
    report=pre['report'];assert not report['CUDA_initialized'] and report['all_codec_parameters_buffers_unchanged'] and report['actual_encoder_forwards']==2 and report['actual_readout_forwards']==1
    assert report['source_cache_excludes_answer_memory'] and not report['thought_model_instantiated'] and report['optimizer_steps']==0
    if mode=='review':
        r=io.load_json(A/'cpu_preflight_owned/runtime/cache_records.json')
        def load(ref):
            assert support.sha(ref['path'])==ref['SHA'];return torch.load(ref['path'],map_location='cpu',weights_only=True)
        m=assemble_input(load(r['static']),load(r['source']))
        target=load(r['target']['target']);trace=load(r['target']['ceiling_trace'])
        assert target['source_codec_SHA']==p['source_codec_SHA'] and target['semantic_target'].shape==(1,35,512)
        assert target['valid_extent_mask'].shape==(1,35) and target['labels'].shape==(trace['patch_count'],16)
        labels=target['labels'].flatten();raw=r['target']['expected'].encode()
        assert labels[:len(raw)].tolist()==list(raw) and int(labels[len(raw)])==256 and (labels[len(raw)+1:]==257).all()
        assert not m.exact_valid.any() and m.exact_values.count_nonzero()==0 and m.exact_keys.count_nonzero()==0
        old=(ROOT/'scripts/supervise_missing_assessment_v93.py').read_text(encoding='utf-8');new=OWNER.read_text(encoding='utf-8')
        assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
        assert shutil.disk_usage(ROOT).free>=START
        io.atomic_json(A/'root_independent_validate.json',dict(status='ROOT_FROZEN_CACHE_V95_READY',actual_CPU=report,data=independent_data(Path(p['data'])),
            source_only_cache_independently_loaded=True,target_separate_shape35x512=True,unchanged_postPopen_OS_cleanup=True,
            actual_fixed_cost=dict(encode=511,readout=31,total=542,optimizer=0,denoise=0),Cfree=shutil.disk_usage(ROOT).free,whole_output128MiB=True))
        print('ROOT_FROZEN_CACHE_V95_READY');return
    review=io.load_json(A/'root_independent_validate.json');assert mode=='seal' and review['status']=='ROOT_FROZEN_CACHE_V95_READY' and shutil.disk_usage(ROOT).free>=START
    for f in (A/'cpu_preflight_owned/final_status.json',A/'root_independent_validate.json'):p['files'][str(f)]=support.sha(f)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+240,deadline=now+300);out.mkdir(exist_ok=False);io.atomic_json(out/'plan.json',p);validate(out/'plan.json',out)
    snap=out/'code_snapshot';snap.mkdir()
    for f in (Path(__file__).resolve(),WORKER,OWNER,ROOT/'research/thought_cache_v95.py'):(snap/f.name).write_bytes(f.read_bytes())
    print(json.dumps(dict(planSHA=support.sha(out/'plan.json'),output=str(out),deadline=p['deadline'],calls=542,optimizer=0)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','review','seal']);main(a.parse_args().mode)
