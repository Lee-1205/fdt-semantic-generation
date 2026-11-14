"""No model forwards: independent saved logits, raw bytes and fullstates review."""
from collections import Counter
import gc,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import text_generation_support_v1 as support
torch.set_num_threads(2)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def gram_stats(text):
    words=text.split();grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return {'words':len(words),'repeated_4gram_fraction':1-len(set(grams))/len(grams) if grams else 0.,
        'the_word_fraction':words.count('the')/len(words) if words else 0.}


def main():
    out=ROOT/'runs/text_language_continuation_20261028_v6';plan=read(out/'plan.json');result=read(out/'result.json')
    terminal=read(out/'supervisor/final_status.json');native=read(out/'code_snapshot/root_native_terminal_check.json')
    assert terminal['status']=='COMPLETE' and terminal['exit']['all_owned_processes_gone'] and native['native_python_count']==0
    assert support.sha(out/'plan.json')==terminal['plan_sha256']
    artifact=Path(plan['artifact']);features=read(artifact/'features.json');targets=read(artifact/'targets.json')
    raw_review={};nll_review={}
    for phase in ('baseline','final'):
        summary=read(out/(phase+'_NLL.json'))
        payload=torch.load(out/(phase+'_NLL_logits.pt'),map_location='cpu',weights_only=True)
        logits=payload['logits'].double();ids=payload['target_ids'];active=ids!=257
        expected=torch.full_like(ids,257)
        for i,t in enumerate(targets['dev']):expected[i,:len(t['target_ids'])]=torch.tensor(t['target_ids'])
        assert torch.equal(ids,expected) and torch.isfinite(logits).all()
        # Independent stable log-sum-exp, no training/evaluation helper call.
        nll=logits.logsumexp(-1)-logits.gather(-1,ids[...,None]).squeeze(-1)
        pred=logits.argmax(-1);symbols=int(active.sum());ce=float(nll[active].sum())
        accuracy=float(((pred==ids)&active).sum())/symbols
        assert symbols==53150 and abs(ce/symbols-summary['NLL'])<2e-6
        assert abs(accuracy-summary['byte_accuracy'])<1e-12
        end_rows=[i for i,t in enumerate(targets['dev']) if t['source_eof']]
        end_correct=0;margins=[];ranks=[]
        for i in end_rows:
            position=len(targets['dev'][i]['target_ids'])-1;v=logits[i,position]
            end_correct+=int(pred[i,position]==256)
            other=torch.cat((v[:256],v[257:]));margins.append(float(v[256]-other.max()))
            ranks.append(int((v>v[256]).sum())+1)
        assert end_correct==summary['true_EOF_correct']
        for i,row in enumerate(summary['raw']):assert row['predicted_ids']==pred[i].tolist()
        nll_review[phase]={'NLL_reconstructed':ce/symbols,'byte_accuracy':accuracy,'symbols':symbols,
            'true_EOF_correct':end_correct,'true_EOF_count':len(end_rows),
            'EOF_gold_margin_mean':sum(margins)/len(margins),'EOF_gold_rank_median':sorted(ranks)[len(ranks)//2],
            'logits_FILE_sha256':support.sha(out/(phase+'_NLL_logits.pt'))}
        del logits,ids,payload,nll,pred;gc.collect()
        generated=read(out/(phase+'_generation.json'));raw_review[phase]={}
        for condition,rows in generated.items():
            decoded=[];stats=[];utf8=eos_count=specials=0
            for row in rows:
                item=row['generated'];values=item['token_ids'];ended=256 in values
                prefix=values[:values.index(256)] if ended else values
                bad=any(x<0 or x>=256 for x in prefix);raw=bytes(x for x in prefix if 0<=x<256)
                try:text=raw.decode('utf-8');valid=True
                except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');valid=False
                assert (text,ended,valid,bad)==(item['text'],item['eos'],item['utf8_valid'],item['special_before_eos'])
                assert raw.hex()==item['raw_hex']
                utf8+=valid;eos_count+=ended;specials+=bad;decoded.append(text);stats.append(gram_stats(text))
            raw_review[phase][condition]={'rows':len(rows),'UTF8':utf8,'EOS':eos_count,'special_before_EOS':specials,
                'unique_output_strings':len(set(decoded)),
                'repeated_4gram_fraction_mean':sum(s['repeated_4gram_fraction'] for s in stats)/len(stats),
                'the_word_fraction_mean':sum(s['the_word_fraction'] for s in stats)/len(stats),
                'length_bytes':[len(x.encode('utf-8')) for x in decoded]}
    source=torch.load(plan['source'],map_location='cpu',weights_only=True)
    assert support.sha(plan['source'])==plan['source_sha'] and support.state_sha(source)==plan['source_algorithm']
    imported=torch.load(out/'import/checkpoints/state_0149.pt',map_location='cpu',weights_only=True)
    keys=('model','optimizer','optimizer_parameter_names','scaler','order','cursor','attempts','updates','skips',
        'history','exposure','actual_calls','evaluations','frozen_sha','scheduler','gradients',
        'torch_rng','cuda_rng','python_rng','numpy_rng')
    import_hash=support.state_sha({k:source[k] for k in keys})
    assert import_hash==support.state_sha({k:imported[k] for k in keys})
    del imported;gc.collect()
    states=[];last=None
    for directory in ('import','model'):
        for marker in sorted((out/directory/'checkpoints').glob('*.json')):
            meta=read(marker);path=marker.parent/meta['file'];payload=torch.load(path,map_location='cpu',weights_only=True)
            assert support.sha(path)==meta['sha256'] and support.state_sha(payload)==meta['algorithm_sha256']
            names={n for ns in payload['optimizer_parameter_names'].values() for n in ns}
            frozen={k:v for k,v in payload['model'].items() if k not in names}
            assert support.state_sha(frozen)==payload['frozen_sha']==source['frozen_sha']
            assert payload['cursor']==min(8*payload['attempts'],3271)
            assert [i for row in payload['history'] for i in row['rows']]==payload['order'][:payload['cursor']]
            assert len(payload['history'])==payload['attempts']
            states.append({'file':str(path.relative_to(out)),'sha256':meta['sha256'],
                'algorithm_sha256':meta['algorithm_sha256'],'attempts':payload['attempts'],'updates':payload['updates'],
                'Adam_parameter_states':len(payload['optimizer']['state']),'scaler':payload['scaler']})
            if payload['attempts']==409:last=payload
            else:del payload
            gc.collect()
    assert last and (last['attempts'],last['updates'],last['cursor'],last['skips'])==(409,408,3271,0)
    assert [r['attempt'] for r in last['history'] if r['outcome']=='FAILED_ATTEMPT']==[149]
    assert len(last['optimizer']['state'])==144 and len(source['optimizer']['state'])==74
    step_counts=Counter(int(state['step']) for state in last['optimizer']['state'].values())
    assert dict(step_counts)=={408:74,260:70}
    assert torch.equal(last['model']['vp.language.embedding.weight'],source['model']['vp.language.embedding.weight'])
    assert not torch.equal(last['model']['vp.velocity.weight'],source['model']['vp.velocity.weight'])
    exposure={}
    for prefix,indices in (
        ('selected',[i for row in last['history'] for i in row['rows']]),
        ('backpropagated',[i for row in last['history'] if row['outcome']=='UPDATED' for i in row['rows']])):
        exposure[prefix]={'rows':len(indices),'target_bytes':sum(len(targets['train'][i]['target'].encode('utf-8')) for i in indices),
            'EOF':sum(targets['train'][i]['source_eof'] for i in indices),
            'documents':len({targets['train'][i]['document_id'] for i in indices})}
    assert exposure['selected']['rows']==3271 and exposure['selected']['target_bytes']==384675
    assert exposure['backpropagated']['rows']==3263
    assert sum(last['allocation_calls'].values())==12978 and sum(last['actual_calls'].values())==21153
    for path,h in plan['files'].items():assert support.sha(path)==h,path
    assert support.core_sha(ROOT)==support.CORE_SHA and not torch.cuda.is_initialized()
    review={'status':'ROOT_CPU_RAW_AND_FULLSTATE_VERIFIED','plan_sha256':support.sha(out/'plan.json'),
        'result_sha256':support.sha(out/'result.json'),'global_attempts':409,'global_updates':408,
        'new_updates':260,'prior_failed_attempt':149,'frozen_prefix_codec_unchanged':True,
        'import_BEFORE_AFTER_algorithm_sha256':import_hash,'all_source_optimizer_scaler_RNG_cursor_preserved':True,
        'fullstates':states,'final_Adam_step_counts':dict(step_counts),'required_gradient_coverage_corrected':True,
        'allocation_calls':last['allocation_calls'],'cumulative_v2_v6_calls':last['actual_calls'],
        'exposure':exposure,'worker_processed_exposure':last['exposure'],'NLL':nll_review,'generation':raw_review,
        'free_generation_semantic_review':'Root read all16 normal baseline/final prompts and outputs: baseline garbled old-relation fragments; final repetitive the/and strings, no coherent document continuation or EOS. Neither condition demonstrates useful story generation.',
        'goal_NLL_reduction_at_least_point1':True,'goal_free_generation_retention':False,'overall_development_goal':'FAIL',
        'capability':'Teacher-forced byte prediction improved across the combined408updates; not attributed only to V6 or an isolated diffusion effect.',
        'actual_additional_model_forwards':0,'CUDA_initialized':False,'all_owned_gone':True,
        'allocation_closed':True,'unused_budget_reusable':False,'E2_final_accessed':False}
    with (out/'root_result.json').open('x',encoding='utf-8') as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in review.items() if k not in ('fullstates',)},ensure_ascii=True))


if __name__=='__main__':main()
