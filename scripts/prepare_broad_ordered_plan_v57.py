"""Concrete new one-epoch scope: all original TRAIN sources, no huge role cache."""
import ast,collections,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_broad_ordered_20261111_v57 import validate,SOURCE_SHA,PARENT_SHA

def main():
    artifact=ROOT/'research/artifacts/broad_ordered_20261111_v57_cpu';path=artifact/'plan_unsealed.json'
    if path.exists():raise FileExistsError('new immutable plan')
    oldrun=ROOT/'runs/balanced_fuzzy_20261111_v56';old=json.loads((oldrun/'plan.json').read_text(encoding='utf-8-sig'))
    data=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu'
    train=json.loads((data/'train.json').read_text(encoding='utf-8-sig'))
    names=('upper','lower','digit','other')
    table=bytes(0 if 65<=i<=90 else 1 if 97<=i<=122 else 2 if 48<=i<=57 else 3 for i in range(256))
    counts=collections.Counter()
    for row in train:counts.update(row['prompt'].encode().translate(table))
    total=sum(counts.values());weights={k:math.sqrt(total/(4*counts[k])) for k in range(4)}
    norm=sum(counts[k]*weights[k] for k in range(4))/total;weights={k:v/norm for k,v in weights.items()}
    lookup=[weights[table[i]] for i in range(256)]+[1.,0.]
    source=Path(old['source']);head=oldrun/'model/checkpoints/state_3624.pt';output=ROOT/'runs/broad_ordered_20261111_v57'
    pins={k:v for k,v in old['files'].items() if Path(k).suffix=='.py'}
    files=[ROOT/'scripts/broad_ordered_engine_v57.py',ROOT/'scripts/run_broad_ordered_20261111_v57.py',
        ROOT/'scripts/supervise_broad_ordered_v57.py',ROOT/'scripts/test_broad_ordered_v57_cpu.py',Path(__file__),
        artifact/'root_CPU_contract.json',source,head,oldrun/'root_saved_review.json',
        data/'train.json',data/'dev.json',data/'manifest.json']
    for file in files:pins[str(file)]=support.sha(file)
    for name,digest in pins.items():
        if support.sha(name)!=digest:raise ValueError('pin changed')
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    p=dict(schema='fdt_broad_ordered_v57',authorized=False,root=str(ROOT),output=str(output),data=str(data),
        source=str(source),source_sha=SOURCE_SHA,head_source=str(head),head_sha=PARENT_SHA,files=pins,
        capacity=16384,batch=1,attempts=8192,epochs=1,amp_dtype='bfloat16',wall_seconds=3300,
        start=None,cutoff=None,deadline=None,max_calls=17000,max_output_bytes=1024**3,cost=16416,
        raw_indices=[0,4,8,12],class_weights=lookup,class_counts_TRAIN={names[k]:v for k,v in counts.items()},
        class_weights_TRAIN={names[k]:v for k,v in weights.items()},early_patch_multiplier=4,
        exposure=dict(TRAIN_documents=8192,target_UTF8_bytes=40416999,document_EOF=7859,input_window_cut_EOS=333),
        objective='one doc loss mean with TRAIN-class inverse-sqrt and first32position4; EOS1/PAD0; no answer-prefix feature',
        optimizer='same8Adam/scaler/allRNG from V56 step7248, cosine8192steps to .1 initialLR; source96Adam frozen in immutable V53',
        recovery='head8Adam/scaler/allRNG/schedule/docorder/cursor/history/full source references; no growing role cache',
        checkpoints=[0,1,2048,4096,6144,8192],maximum_additional_STOP_checkpoint=1,
        denoise_calls=0,source_copy_bytes=0,source_encoder_optimizer_updates=0,automatic_promotion=False,
        start_free_bytes=110*1024**3,stop_free_bytes=101*1024**3,protected_free_bytes=100*1024**3)
    path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(path,output)
    print(json.dumps(dict(actual_schema_PASS=True,cost=16416,weights=p['class_weights_TRAIN'],TRAIN_bytes=total)))
if __name__=='__main__':main()
