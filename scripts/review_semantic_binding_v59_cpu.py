"""Independent saved bank/role/factor checks and explicit bypass diagnosis."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(4);run=ROOT/'runs/semantic_binding_20261111_v59';p=io.load_json(run/'plan.json')
    terminal=io.load_json(run/'final_status.json');native=io.load_json(run/'root_terminal_native.json')
    assert terminal['status']=='COMPLETE' and terminal['exit']['all_owned_processes_gone'] and native['fresh_python_count']==0
    states={}
    for directory,steps in [('seed',[0]),('initial',[0]),('model',[1,288,576])]:
        for step in steps:
            path=run/directory/'checkpoints'/f'state_{step:04d}.pt';meta=io.load_json(path.with_suffix('.json'))
            assert support.sha(path)==meta['sha256'];s=torch.load(path,weights_only=True,map_location='cpu')
            assert support.state_sha(s)==meta['algorithm_sha256'];states[(directory,step)]=s
            assert s['attempts']==s['updates']==step and not s['skips']
            assert all(int(v['step'])==step for v in s['optimizer']['state'].values())
    final=states[('model',576)];initial=states[('initial',0)];data=Path(p['data'])
    assert final['cursor']==9216 and final['exposure']=={'row_selections':9216,'cached_source_UTF8_bytes':465512}
    assert [h['rows'] for h in final['history']]==final['orders'] and final['scaler']['_growth_tracker']==576 and final['scaler']['scale']==1024
    for key in ('torch_rng','cuda_rng','python_rng','numpy_rng'):
        assert support.state_sha({key:initial[key]})==support.state_sha({key:final[key]})
    assert torch.equal(initial['binder']['persistent_prior'],final['binder']['persistent_prior'])
    for name,digest in final['cache_refs'].items():assert support.sha(name)==digest,name
    assert support.sha(p['source'])==p['source_sha'] and support.sha(p['head_source'])==p['head_sha']
    features=torch.load(run/'dev_features.pt',weights_only=True,map_location='cpu');gold=io.load_json(data/'dev_targets.json');rows=io.load_json(data/'dev_features.json')
    raw=torch.load(run/'raw_final.pt',weights_only=True,map_location='cpu');scores={};examples=[]
    for kind in range(3):
        indices=[i for i,t in enumerate(gold) if t['meaning']['relation']==kind];fact=role_good=role_total=0;pol=qty=0
        for i in indices:
            source=rows[i]['prompt'].encode();predspans=raw['spans'][i].tolist()
            selected=[None if a<0 else source[a:b].decode() for a,b in predspans]
            predicted=dict(relation=int(raw['relation_probabilities'][i].argmax()),polarity=int(raw['polarity_probabilities'][i].argmax()),
                quantity=int(raw['quantity_probabilities'][i].argmax()),roles=dict(zip(('primary','recipient','theme'),selected)))
            truth=gold[i]['meaning'];fact+=predicted==truth;pol+=predicted['polarity']==truth['polarity'];qty+=predicted['quantity']==truth['quantity']
            for key in truth['roles']:
                if truth['roles'][key] is not None:role_total+=1;role_good+=predicted['roles'][key]==truth['roles'][key]
            if len([e for e in examples if e['kind']==kind])<6:examples.append(dict(kind=kind,source=rows[i]['prompt'],predicted=predicted,gold=truth,exact=predicted==truth))
        scores[str(kind)]=dict(rows=len(indices),exact_facts=fact,present_role_correct=role_good,present_role_total=role_total,polarity_correct=pol,quantity_correct=qty)
    # Actual role bank N/D reconstructed from stored neural degrees and values.
    degree=raw['role_degree'].double();mu=features['mu'].double();v=raw['projected_values'].double();valid=features['valid']
    weight=degree[...,None]*mu[:,None]*valid[:,None,:,None]
    expected_n=torch.einsum('brta,btv->brav',weight,v);expected_d=weight.sum(2)[...,None]
    assert torch.allclose(raw['role_N'],expected_n,atol=1e-8) and torch.allclose(raw['role_D'],expected_d,atol=1e-10)
    controls={}
    for bank in ('persistent','contextual','exact'):
        other=torch.load(run/('raw_read_off_'+bank+'.pt'),weights_only=True,map_location='cpu')
        unchanged={key:float((other[key]-raw[key][:64]).abs().max()) for key in ('start_logits','end_logits','relation_logits','negative_logits','quantity_logits')}
        controls[bank]=dict(factor_and_span_logit_deltas=unchanged,semantic_vector_delta=float((other['semantic_vector']-raw['semantic_vector'][:64]).abs().max()))
    training={name:sum(h['losses'][name] for h in final['history'][-72:])/72 for name in final['history'][0]['losses']}
    report=dict(status='INDEPENDENT_V59_SAVED_REVIEW_PASS',all5_states_SHA_safeload_digest=True,Adam_steps576_scaler_ALLRNG_prior_orders_verified=True,
        all_cache_raw_SHA_verified=True,actual_role_ND_independently_reconstructed=True,old_source_codec_SHA_unchanged=True,
        per_relation=scores,last_epoch_losses=training,read_channel_controls=controls,examples=examples,
        important='V59 factor heads use pre-bank token pooling. Read-channel controls change semantic vector but cannot change factor/span predictions by construction. Factor gains are not proof that canonical role K_C is read.',
        next='Separate bound-memory extraction state and memory-only factor readout; include a learned event/cue fuzzy role so predicate/negation/quantity are not lost by entity-only spans.',
        extra_model_calls=0,CUDA_initialized=torch.cuda.is_initialized())
    io.atomic_json(run/'root_saved_review.json',report);print(json.dumps({k:v for k,v in report.items() if k!='examples'}))
if __name__=='__main__':main()
