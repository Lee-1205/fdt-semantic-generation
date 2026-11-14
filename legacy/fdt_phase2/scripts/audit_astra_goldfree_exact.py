"""Query-blind parser-assisted multi-record evaluation; labels only reach scoring."""
from pathlib import Path
import sys, json, random
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState
from fdt_rlm.runtime_contract import validate_generation_budget


def register_records(records, query, tokenizer, device):
    # Deliberately segmented tokenization is the input protocol, uniformly for
    # every record; we never locate a record by searching for the gold answer.
    ids=list(tokenizer.encode('Records:\n'))
    positions=[];payloads=[];source_map=[]
    for record in records:
        ids.extend(tokenizer.encode(record['name']+' = '))
        key=len(ids)-1
        value=list(tokenizer.encode(record['value']))
        if not value: raise ValueError('empty record value')
        positions.append(key);payloads.append(value)
        source_map.append({'name':record['name'],'key_position':key,'value':record['value']})
        ids.extend(value);ids.extend(tokenizer.encode('\n'))
    ids.extend(tokenizer.encode('\n'+query+'\nAnswer:'))
    ends=torch.arange(len(ids),device=device).view(1,-1)
    mask=torch.zeros(1,len(ids),device=device,dtype=torch.bool)
    width=max(map(len,payloads));padded=[]
    for key,value in zip(positions,payloads):
        mask[0,key]=True;ends[0,key+1:key+1+len(value)]=key+len(value)
        padded.append(value+[0]*(width-len(value)))
    meta={'span_end_positions':ends,'registered_key_mask':mask,
          'registered_key_positions':torch.tensor([positions],device=device),
          'registered_payload_ids':torch.tensor([padded],device=device),
          'registered_payload_lengths':torch.tensor([[len(v) for v in payloads]],device=device)}
    text='Records:\n'+''.join(r['name']+' = '+r['value']+'\n' for r in records)+'\n'+query+'\nAnswer:'
    return text,ids,meta,source_map


def cases(seed=20260916):
    rng=random.Random(seed);result=[]
    for count in (2,5,17):
        for kind in ('present','similar','duplicate','absent','noncopy'):
            for replicate in range(2):
                records=[{'name':f'item_{j:02d}','value':f'LV{rng.randrange(100000,999999)}'} for j in range(count)]
                target=replicate%count
                if kind=='similar':
                    base=records[target]['value']
                    for j,r in enumerate(records):
                        if j!=target:r['value']=base[:-2]+f'{j:02d}'
                if kind=='duplicate': records[(target+1)%count]['name']=records[target]['name']
                query=f"Copy the value of {records[target]['name']} exactly. End the answer after that value."
                if kind=='absent': query='Copy the value of item_missing exactly. If absent, do not copy any record.'
                if kind=='noncopy': query='Explain in one sentence what an identifier is. Do not copy any record.'
                gold=dict(records[target]) if kind in ('present','similar') else None
                rng.shuffle(records)
                result.append({'id':len(result),'kind':kind,'records':records,'query':query,'gold':gold})
    return result


@torch.inference_mode()
def generate(model,tokenizer,records,query,budget=96):
    device=next(model.parameters()).device;config=model.config
    prompt,ids,meta,source_map=register_records(records,query,tokenizer,device)
    validate_generation_budget(config,len(ids),budget)
    generated=torch.tensor([ids],device=device);output=model(generated,attention_mask=torch.ones_like(generated))
    memory=model.build_exact_memory(output['hidden'],generated,torch.ones_like(generated),source_length=len(ids),**meta)
    state=LexicalPointerDecodeState(source_length=len(ids),max_activation_steps=budget,max_copy_tokens=budget)
    selected=[];trace=[]
    for _ in range(budget):
        base=output['logits'][:,-1].float()
        logits,diag=state.prepare_logits(model.exact_pointer,base,output['hidden'],generated,torch.ones_like(generated),
            min_gate=0.,anchor_memory=memory,query_anchor_ids=model.exact_route_indices(output['hidden'])[:,-1],
            max_candidate_chunks=config.exact_pointer_candidate_chunks,
            full_scan_fallback=True,candidate_cap=config.exact_memory_candidate_cap,
            commit_threshold=config.exact_memory_commit_threshold,hard_copy=False)
        token=int(logits.argmax(-1));before=state.copied_tokens
        state.commit(token,diag)
        trace.append({'selected_id':token,'mode':diag.get('mode'),
            'source_positions':diag.get('source_positions'),
            'gate':diag.get('gate'),'pointer_confidence':diag.get('pointer_confidence'),
            'score_margin':diag.get('score_margin'),'committed':state.copied_tokens>before,
            'cursor_commit_eligible':diag.get('cursor_commit_eligible'),
            'copied_tokens':state.copied_tokens,'completed':state.completed})
        selected.append(token)
        if token==config.eos_token_id:break
        generated=torch.cat((generated,torch.tensor([[token]],device=device)),1)
        output=model(generated,attention_mask=torch.ones_like(generated))
    return {'prompt':prompt,'prompt_ids':ids,'source_map':source_map,'generated_ids':selected,
            'text':tokenizer.decode(selected,skip_special_tokens=True),'trace':trace,
            'eos':bool(selected and selected[-1]==config.eos_token_id),'budget':budget,
            'registration':'all-records-query-blind','hard_copy_enabled':False}


def score(case,run,tokenizer,eos_id):
    gold=case['gold'];trace=run['trace'];steps=[t for t in trace if t['committed']]
    first=trace[0] if trace else {};positions=first.get('source_positions') or [[]]
    wanted=next((r['key_position'] for r in run['source_map'] if gold and r['name']==gold['name'] and r['value']==gold['value']),None)
    answer_ids=list(tokenizer.encode(gold['value'])) if gold else None
    output=run['generated_ids'];clean=output[:-1] if output and output[-1]==eos_id else output
    correct_source=bool(wanted is not None and positions[0] and positions[0][0]==wanted)
    first_exact=bool(answer_ids and output and output[0]==answer_ids[0])
    whole=clean==answer_ids if gold else None
    cursor=[t for t in trace if t['mode']=='cursor']
    starts=[t for t in steps if t['mode']!='cursor']
    wrong_commit=any(not t.get('source_positions') or t['source_positions'][0][0]!=wanted for t in starts) if gold else bool(steps)
    gold_cursor_correct=all(j<len(answer_ids) and t['selected_id']==answer_ids[j] for j,t in enumerate(trace) if t['mode']=='cursor') if gold and cursor else None
    return {'source_top1':correct_source if gold else None,
        'source_topk':wanted in positions[0] if gold else None,
        'first_token_exact':first_exact if gold else None,
        'real_cursor_commit':bool(steps), 'real_cursor_continuation_steps':len(cursor),
        'cursor_transport_followed_selected_source':all(t['committed'] for t in cursor) if cursor else None,
        'cursor_continuation_correct':gold_cursor_correct,
        'selected_source_and_span_correct':correct_source if gold else None,
        'whole_string_exact':whole,'termination_correct':bool(whole and run['eos']) if gold else None,
        'whole_chain_success':bool(correct_source and first_exact and steps and whole and run['eos']) if gold else None,
        'false_commit':wrong_commit,
        'abstained_from_copy':not steps,'normal_eos':run['eos'],
        'max_token_exhaustion':not run['eos'] and len(output)==run['budget']}
