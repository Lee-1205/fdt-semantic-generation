"""Independent data review + tiny ownership + full-parent tensor warm copy."""
import collections,copy,gc,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.fuzzy_extraction_v73 import enable_fixed_readout_extraction
from research.text_diffusion_v1 import pack_prompts
from research.frozen_verbalizer_objective_v72 import decoder_losses
from research.fuzzy_factual_objective_v70 import critical_mask

def independent_data(d):
    meta=io.load_json(d/'manifest.json');rows=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');dev=io.load_json(d/'dev_features.json');devlabels=io.load_json(d/'dev_targets.json');orders=io.load_json(d/'orders.json')
    assert len(rows)==len(labels)==960 and len(dev)==len(devlabels)==32 and len(orders)==960 and sorted(v[0] for v in orders)==list(range(960))
    groups=collections.defaultdict(list);primary=collections.Counter();coverage=collections.defaultdict(collections.Counter);allnames=meta['TRAIN_names']+meta['new_DEV_names']
    for feature,target in zip(rows+dev,labels+devlabels):
        assert set(feature)=={'id','prompt'} and feature['id']==target['id'];b=feature['prompt'].encode();last=-1
        for span in target['critical_spans']:
            assert last<=span['start']<span['end']<=len(b);last=span['end'];assert b[span['start']:span['end']].decode()==span['text']
            if feature in rows:
                for pos in range(span['start'],span['end']):coverage[chr(b[pos])][str(pos%16)]+=1
        if target['source_kind']=='prospective_authored':
            assert target['actor_position_role']==target['clause_start']%16 and feature['prompt'][target['clause_start']:].startswith(target['clause'])
            clause=target['clause'];a=target['actor'];other=target['recipient'];q=target['primary_value'];left=target['reserve'];family=target['family']
            if family==0:
                match=re.fullmatch(r'(\w+) recorded \+(\d+)\.45 degrees C at (\d{2}:\d{2}); (\w+) kept -(\d+)\.80 kg in the green container\. Neither person changed the label\.',clause)
                assert match and match[1]==a and int(match[2])==q and match[4]==other and int(match[5])==left
            elif family==1:
                match=re.fullmatch(r'(\w+) paid \$(\d+),(\d{3})\.25 to (\w+) on (2024-\d{2}-\d{2}); the refund was -\$(\d+)\.80\. The earlier amount was not paid twice\.',clause)
                assert match and match[1]==a and int(match[2])==100+q and match[4]==other and int(match[6])==left
            elif family==2:
                match=re.fullmatch(r'(\w+) moved (\d+) blue folders to (\w+) before the bell; (\w+) kept (\d+) folders and did not move them\.',clause)
                assert match and match[1]==a and int(match[2])==q and match[3]==match[4]==other and int(match[5])==left
            else:
                match=re.fullmatch(r'(\w+) signed a note: (\w+) kept (\d+) wooden tokens after the meeting, and (\w+) did not take the remaining (\d+) tokens\.',clause)
                assert match and match[1]==match[4]==a and match[2]==other and int(match[3])==q and int(match[5])==left
            names=[(m.start(),m.end()) for m in re.finditer(r'\b(?:'+'|'.join(allnames)+r')\b',feature['prompt'])]
            assert names==[(s['start'],s['end']) for s in target['critical_spans'] if s['kind']=='name']
    for i,t in enumerate(labels[:832]):groups[t['pair_id']].append(i);primary[t['primary_value']]+=1
    position={r[0]:j for j,r in enumerate(orders)}
    for pair,indices in groups.items():
        assert len(indices)==2 and [labels[i]['member'] for i in indices]==[0,1] and abs(position[indices[0]]-position[indices[1]])==1
        a,b=(labels[i] for i in indices);assert b['primary_value']==a['primary_value']+1 and all(a[k]==b[k] for k in ('family','actor','recipient','reserve','clause_start','actor_position_role'))
    assert len(groups)==416 and set(primary)==set(range(100)) and min(primary.values())==8 and max(primary.values())==9
    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZz+:':assert set(coverage[c])=={str(i) for i in range(16)}
    assert sum(len(r['prompt'].encode()) for r in rows)==meta['source_UTF8_bytes'] and sum(s['end']-s['start'] for t in labels for s in t['critical_spans'])==meta['critical_bytes']
    old=io.load_json(ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu/train_features.json');assert rows[832:]==old[384:512]
    held=[]
    for f in ('fresh_fuzzy_eval_20261112_v66_cpu/rows.json','long_extrapolation_20261112_v69_cpu/features.json','factual_cycle_20261112_v70_cpu_v2/dev_features.json','frozen_verbalizer_20261112_v72_cpu/dev_features.json'):
        held+=io.load_json(ROOT/'research/artifacts'/f)
    assert not any(h['prompt'] in row['prompt'] for h in held for row in rows)
    assert not any(re.search(r'\b'+n+r'\b',row['prompt']) for n in meta['new_DEV_names']+['Elric','Mavis','Torin','Yvette','Stellan','Brynn','Lucan','Odette','Inez'] for row in rows)
    assert not {r['prompt'] for r in rows}&{r['prompt'] for r in dev}
    return dict(all992_source_target_span_rows_verified=True,all_authored_clauses_facts_independently_parsed=True,complete416_pairs_adjacent_finite_orders=True,
        primary_values_0_to99_each8or9=True,all_capitals_z_plus_colon_all16positions=True,old128_OWT_TRAIN_features_exact=True,known_held_prompt_not_TRAIN=True,new_held_name_whole_tokens_not_TRAIN=True,
        source_UTF8_bytes=meta['source_UTF8_bytes'],critical_bytes=meta['critical_bytes'],files_SHA={p.name:support.sha(p) for p in d.glob('*.json')})

def candidate():
    torch.set_num_threads(2);torch.manual_seed(73);c=Config.smoke()
    enc=FuzzyCodecV64(c,input_capacity=256);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks();m=FuzzyRoundTripV64(enc,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=128))
    names=enable_fixed_readout_extraction(m);frozen=support.state_sha({k:v for k,v in m.state_dict().items() if k not in names});named=dict(m.named_parameters())
    optimizer=torch.optim.AdamW([named[n] for n in names],lr=1e-6);scaler=torch.amp.GradScaler('cpu',init_scale=1024)
    text='Yara kept 7 wooden tokens; Isolde kept 6. Neither person took the remaining objects.';x=pack_prompts([text],len(text.encode()),'cpu')
    commit,episode=m.extract(x,source_ids=('tiny73',),source_versions=('v1',),model_epoch='fixture73')
    # reader_tensor intentionally clones on each access. Inspect the stored
    # training tensor only in this CPU test; normal verbalize still clones it.
    semantic=commit._semantic;semantic.retain_grad();z=m.verbalize(commit)
    y=torch.full(z.shape[:2],257,dtype=torch.long);y.flatten()[:len(text)]=torch.tensor(list(text.encode()));y.flatten()[len(text)]=256
    mask=critical_mask(y,[dict(start=0,end=4),dict(start=10,end=11)],len(text));p=z.detach().softmax(-1).gather(-1,y[...,None]).squeeze(-1);correct=z.detach().argmax(-1)==y
    weights=torch.ones(258);weights[257]=0;loss,_=decoder_losses(z,y,weights,mask,p,correct);scaler.scale(loss).backward()
    assert semantic.grad is not None and semantic.grad.count_nonzero() and all(named[n].grad is not None for n in names)
    assert all(v.grad is None for n,v in named.items() if n not in names) and not episode.memories.exact.valid.any()
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);assert support.state_sha({k:v for k,v in m.state_dict().items() if k not in names})==frozen
    assert len(optimizer.state)==23 and scaler.is_enabled()
    return dict(tiny_actual_boundCommit_loss_gradient_reaches_upstream23=True,tiny_final_readout7_and_all_other_frozen_tensors_unchanged=True,tiny_enabled_scaler23Adam_oneupdate=True,
        tiny_encode_calls=1,tiny_readout_calls=1,limitation='ownership/gradient only; not whole worker/cache/phase/ALLRNG recovery or model capability')

def parent_warm():
    file=ROOT/'runs/frozen_verbalizer_20261112_v72/model/checkpoints/state_0640.pt';assert support.sha(file)=='74531e0baa91f8f3fd0bfb26aeedc73871c9bc4afbd5a896ad6c25d4158fe864'
    parent=torch.load(file,map_location='cpu',weights_only=True);before=support.state_sha(parent);assert before=='d2b59076b393dcfcafef768da2cc377cc95f47bba8ac874247409dabc4eb2e09'
    enc=FuzzyCodecV64(Config(**parent['core']),input_capacity=16384);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks()
    head=OrderedFuzzyReadout();model=FuzzyRoundTripV64(enc,head);model.load_state_dict(parent['model'],strict=True);assert support.state_sha(model.state_dict())==support.state_sha(parent['model'])
    names=enable_fixed_readout_extraction(model);active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert active==14769664 and len(names)==23 and len(parent['optimizer']['state'])==7 and support.state_sha(parent)==before
    assert not torch.cuda.is_initialized()
    return dict(full_parent_all_tensors_exact_CPU_warm=True,parent_SHA=support.sha(file),parent_recursive_digest_unchanged=before,parent7Adam_scaler_ALLRNG_preserved=True,
        actual_upstream_active_parameters=active,actual_upstream_tensor_count=len(names),actual_readout7_frozen=True,actual_parent_model_forward=0,actual_optimizer_trial=0,CUDA_initialized=False)

def main():
    d=ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2';data=independent_data(d);tiny=candidate();warm=parent_warm()
    for name,value in [('root_data_review',data),('root_candidate_review',dict(status='V73_CPU_DATA_CANDIDATE_READY',**tiny,**warm))]:
        path=d/(name+'.json');assert not path.exists();io.atomic_json(path,value)
    print(json.dumps(dict(data={k:v for k,v in data.items() if k!='files_SHA'},candidate=tiny,warm=warm)))
if __name__=='__main__':main()
