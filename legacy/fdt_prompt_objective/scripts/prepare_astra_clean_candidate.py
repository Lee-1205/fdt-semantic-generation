"""Local, immutable clean-data candidate. Never changes the original prepared_data.

This is a data repair artifact, not authorization to restart a quarantined pilot.
Known parent and aborted-pilot rows are excluded. Older lineage remains unknown.
"""
from pathlib import Path
import sys,json,argparse,ast,textwrap
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from astra_bounded_pilot import digest
from astra_evidence_io import save
from astra_data_contract import retrieval_contract,python_contract,json_contract
from prepare_fdt_v21_knowledge_scale import V21Writer,add_varied_retrieval,token_hash
from fdt_rlm.tokenization import load_tokenizer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume-existing',action='store_true')
    args=ap.parse_args();parent=args.plan_dir.resolve();out=args.out.resolve()
    if not out.is_relative_to(parent.parent) or out==parent or out.is_relative_to(parent):
        raise RuntimeError('candidate must be a new sibling under this workspace outputs')
    if out.exists() and not args.resume_existing:raise FileExistsError(out)
    if (out/'candidate_manifest.json').exists():raise FileExistsError('candidate is already complete')
    out.mkdir(exist_ok=args.resume_existing);torch.set_num_threads(3)
    plan=json.loads((parent/'data_plan_resume.json').read_text(encoding='utf-8'));t=load_tokenizer(plan['tokenizer_dir'])
    audit=parent/'data_semantics_audit';summaries={};dataset_dirs={}
    attempts=[json.loads(p.read_text(encoding='utf-8')) for p in (parent/'execution_v3/A/attempts').glob('*.json')]
    consumed_rows=[r for a in attempts for r in plan['rows'][a['starting_cursor']:a['starting_cursor']+8]]
    for name,info in plan['shards'].items():
        if digest(info['path'])!=info['sha256']:raise RuntimeError('original shard changed')
        d=torch.load(info['path'],map_location='cpu',weights_only=False,mmap=True)
        dest=out/name;dataset_dirs[name]=str(dest)
        if args.resume_existing and (dest/'manifest.json').exists():
            existing=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
            if digest(existing['train']['shard']).lower()!=existing['train']['shard_sha256'].lower():
                raise RuntimeError('completed candidate shard drift')
            if existing['semantic_checks']['semantic_failures']:raise RuntimeError('invalid completed category')
            summaries[name]=existing['semantic_checks'];print('REUSED VERIFIED',name,flush=True)
            continue
        if dest.exists() and any(dest.iterdir()):raise RuntimeError('partial category payload exists; preserve and investigate')
        dest.mkdir(exist_ok=args.resume_existing)
        states=info['initial_state'];order=torch.randperm(len(d['input_ids']),generator=torch.Generator().manual_seed(states['seed']))
        excluded=set(map(int,order[:states['cursor']].tolist()))
        excluded.update(r['row_id'] for r in consumed_rows if r['source']==name)
        diagnostics=json.loads((audit/(name+'.json')).read_text(encoding='utf-8'))['rows']
        if name=='retrieval':
            reference={token_hash(ids[mask.bool()].tolist()) for ids,mask in zip(d['input_ids'],d['attention_mask'])}
            writer=V21Writer(dest,t,512,reference)
            families=add_varied_retrieval(writer,20000,20260917)
            metadata=writer.save();payload=torch.load(metadata['shard'],map_location='cpu',weights_only=False,mmap=True)
            provenance={'method':'regenerated complete synthetic keyed records','seed':20260917,
                        'excluded_exact_original_retrieval_rows':len(reference),'families':families,
                        'rejections':dict(writer.rejections),'global_lineage_overlap':'UNKNOWN'}
        else:
            indices=[];reasons={}
            for i,r in enumerate(diagnostics):
                reason=None
                if i in excluded:reason='known_parent_or_pilot_consumption'
                elif name=='factual' and (r['supervised']>=49 or r['active']>=512):reason='possible_prompt_or_answer_truncation'
                elif name=='python':
                    if r['supervised']>=97 or r['active']>=512:reason='possible_prompt_or_answer_truncation'
                    elif r['status']!='PASS_SYNTAX_ONLY':reason='invalid_python_syntax'
                    else:
                        labels=d['labels'][i];text=t.decode(labels[labels.ge(0)].tolist(),skip_special_tokens=True)
                        try:
                            tree=ast.parse(textwrap.dedent(text).strip());compile(tree,'<candidate>','exec')
                            if not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in tree.body):reason='no_complete_function'
                        except (SyntaxError,ValueError):reason='invalid_python_context'
                elif name=='json' and (r['status']!='PASS_SYNTAX_ONLY' or r['active']>=512):reason='invalid_or_possible_truncated_json'
                if reason:reasons[reason]=reasons.get(reason,0)+1
                else:indices.append(i)
            index=torch.tensor(indices,dtype=torch.long)
            payload={k:v.index_select(0,index).clone() for k,v in d.items() if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==len(d['input_ids'])}
            if name=='natural':
                endings=payload['attention_mask'].sum(1).long()-1;row_ids=torch.arange(len(indices))
                if not payload['input_ids'][row_ids,endings].eq(t.eos_token_id).all():raise RuntimeError('unexpected natural ending')
                payload['input_ids'][row_ids,endings]=t.pad_token_id
                payload['labels'][row_ids,endings]=-100
                payload['attention_mask'][row_ids,endings]=0
                for k in ['grounding_mask','signature_mask']:
                    if k in payload:payload[k][row_ids,endings]=0
            shard_dir=dest/'shards/train';shard_dir.mkdir(parents=True)
            shard=shard_dir/'shard_00000.pt'
            with shard.open('xb') as f:torch.save(payload,f)
            metadata={'rows':len(indices),'shard':str(shard),'shard_sha256':digest(shard)}
            provenance={'method':'conservative local repair/filter','source_path':info['path'],
                        'source_sha256':info['sha256'],'original_row_ids':indices,
                        'excluded':reasons,'global_lineage_overlap':'UNKNOWN',
                        'natural_eos':'removed because document-end provenance was lost' if name=='natural' else None,
                        'answer_cap_saturation':'excluded conservatively; syntax alone is insufficient' if name=='python' else None}
        counts={'rows':len(payload['input_ids']),'active':int(payload['attention_mask'].sum()),'supervised':0,'semantic_failures':0}
        for ids,labels,mask in zip(payload['input_ids'],payload['labels'],payload['attention_mask']):
            valid=mask.bool()&labels.ne(-100)&labels.ne(t.pad_token_id);valid[0]=False
            pos=valid.nonzero().flatten();counts['supervised']+=len(pos)
            prompt=t.decode(ids[:int(pos[0])].tolist(),skip_special_tokens=True)
            answer=t.decode(labels[valid].tolist(),skip_special_tokens=True)
            if name=='retrieval' and retrieval_contract(prompt,answer)!='PASS':counts['semantic_failures']+=1
            if name=='python' and python_contract(answer)!='PASS_SYNTAX_ONLY':counts['semantic_failures']+=1
            if name=='json' and json_contract(answer)!='PASS_SYNTAX_ONLY':counts['semantic_failures']+=1
            if name=='natural' and labels[valid][-1]==t.eos_token_id:counts['semantic_failures']+=1
        if counts['semantic_failures']:raise RuntimeError('candidate semantic failure: '+name)
        save(dest/'manifest.json',{'schema_version':'astra_clean_data_candidate_v1','split':'train','train':metadata,
             'source':name,'max_length':512,'semantic_checks':counts,'provenance':provenance})
        summaries[name]=counts;print(name,json.dumps(counts),flush=True)
    save(out/'candidate_manifest.json',{'status':'CPU_VALIDATED_DATA_CANDIDATE_NOT_TRAINED',
        'source_pilot_stop_sha256':digest(parent/'DATA_SEMANTICS_STOP.json'),
        'tokenizer_dir':plan['tokenizer_dir'],'tokenizer_sha256':plan['tokenizer_sha256'],
        'dataset_dirs':dataset_dirs,'summary':summaries,'original_prepared_data_modified':False,
        'training_allowed_by_this_manifest':False,
        'remaining_gates':['fresh token-normalized plan with source-epoch checks','holdout overlap check',
            'source-specific retention references on valid targets','paired training budget including all prior attempts',
            'variable-prefix and termination diagnostics before promotion'],
        'limitations':['factual truth not globally verified','prior full-lineage overlap UNKNOWN',
            'natural document boundaries unavailable; removed ambiguous EOS','Python subset now favors short complete functions']})


if __name__=='__main__':main()
