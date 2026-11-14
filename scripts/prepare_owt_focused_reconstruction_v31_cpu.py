"""Prospective training-only OWT curriculum; no model construction or execution plan."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
A=R/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final'
O=R/'research/artifacts/owt_focused_reconstruction_20261029_v31_cpu'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    import random
    assert not O.exists();f,y,d=map(lambda n:read(A/n),('features.json','targets.json','documents.json'))
    # Fixed hash ordering, only the conservatively selected newer TRAIN documents.
    selected=sorted(d['train'][4096:],key=lambda doc:(doc['normalized_sha256'],doc['id']))[:64]
    indices=[]
    for doc in selected:
        options=[i for i in range(16384,len(f['train'])) if y['train'][i]['document_id']==doc['id']]
        assert len(options)==4
        indices.append(min(options,key=lambda i:(len(f['train'][i]['prompt'].encode('utf-8')),f['train'][i]['id'])))
    features=[f['train'][i] for i in indices];targets=[y['train'][i] for i in indices]
    assert len({t['document_id'] for t in targets})==64 and all(set(row)=={'id','prompt'} for row in features)
    assert all(row['prompt']==target['target'] and target['target_ids']==list(row['prompt'].encode())+[256] for row,target in zip(features,targets))
    assert not {row['id'] for row in features}&{row['id'] for row in f['dev']}
    assert not {doc['normalized_sha256'] for doc in selected}&{doc['normalized_sha256'] for doc in d['dev']}
    orders=[]
    for seed in range(391,647):
        ids=list(range(64));random.Random(seed).shuffle(ids)
        orders.extend(ids[i:i+8] for i in range(0,64,8))
    assert len(orders)==2048 and all(sorted(i for batch in orders[j:j+8] for i in batch)==list(range(64)) for j in range(0,2048,8))
    O.mkdir();write('features.json',features);write('targets.json',targets);write('orders.json',orders)
    byte_sum=sum(len(row['prompt'].encode()) for row in features)
    manifest={'status':'CPU_TRAINING_CURRICULUM_ONLY_NO_RUNNER_OR_EXECUTION',
        'purpose':'Can the unchanged fuzzy input-meaning-input cycle fit a small declared subset after broad OWT training? This is a learned reconstruction capacity check, not a blind LM benchmark or unique semantic-vector ground truth.',
        'source_artifact':str(A),'source_manifest_sha256':sha(A/'manifest.json'),
        'builder_sha256':sha(Path(__file__)),'selected_train_indices':indices,
        'selected_documents':[{'id':doc['id'],'normalized_sha256':doc['normalized_sha256']} for doc in selected],
        'selection':'64 lowest normalized document hashes among newer conservative training4096docs; shortest existing span per doc, SHA tie break; no output-dependent filtering',
        'training_documents':64,'unique_training_spans':64,'finite_epochs':256,'attempts':2048,'batch':8,
        'physical_target_UTF8_bytes':byte_sum,'row_selections':16384,'target_UTF8_byte_exposures':byte_sum*256,
        'held_rows_moved_to_train':0,'new_documents_downloaded':0,'full_broad_corpus_preserved':True,
        'same_fuzzy_method_and_inference_proposed':True,'reasoning_training_proposed':False,
        'source_learning_state':'V30 final8192/8188/global100287; all81 populated Adam/RNG retained, no newparameters or furtherunfreezing',
        'file_SHA256':{n:sha(O/n) for n in ('features.json','targets.json','orders.json')},
        'inference_features':['id','prompt'],'targets_training_score_only':True,
        'new_model_construction_forward_optimizer_calls':0,'GPU_training_performed':False,'E2_final_accessed':False,
        'limitations':['Repeated64existing spans are not16384new documents or general LM learning.','The source selection favors short existing spans and is not a blind evaluation.','Even exact reconstruction of these training sentences would not prove abstract semantic extraction or successful reasoning denoising.','New actual schema/driver/preflight/fullstate/OSowner/cost/resource review required; current Cfree is below40GiB startfloor.']}
    write('manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('status','physical_target_UTF8_bytes','target_UTF8_byte_exposures','training_documents','attempts','new_model_construction_forward_optimizer_calls')}))
if __name__=='__main__':main()
