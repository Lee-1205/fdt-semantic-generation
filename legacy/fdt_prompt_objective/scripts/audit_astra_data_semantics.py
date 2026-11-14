"""Read-only full-shard semantic audit; no optimizer or prepared-data mutation."""
from pathlib import Path
import sys,json,argparse,collections,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from astra_data_contract import retrieval_contract,python_contract,json_contract
from astra_bounded_pilot import digest
from astra_evidence_io import save
from fdt_rlm.tokenization import load_tokenizer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-dir',type=Path,required=True);args=ap.parse_args()
    plan=json.loads((args.plan_dir/'data_plan_resume.json').read_text());t=load_tokenizer(plan['tokenizer_dir'])
    torch.set_num_threads(3);out=args.plan_dir/'data_semantics_audit';out.mkdir(exist_ok=True)
    steps=[json.loads(p.read_text()) for p in (args.plan_dir/'execution_v3/A').glob('step_*.json')]
    used={(r['source'],r['row_id']) for s in steps for r in s['rows']}
    chosen={(r['source'],r['row_id']) for r in plan['rows']}
    planned_rows={(r['source'],r['row_id']):r for r in plan['rows']};all_summary={}
    for source,info in plan['shards'].items():
        payload=torch.load(info['path'],map_location='cpu',weights_only=False,mmap=True)
        report=[]
        for i in range(len(payload['input_ids'])):
            mask=payload['attention_mask'][i].bool();ids=payload['input_ids'][i][mask]
            labels=payload['labels'][i][mask];valid=labels.ne(-100)&labels.ne(t.pad_token_id);valid[0]=False
            positions=valid.nonzero().flatten();start=int(positions[0]);target=labels[valid].tolist()
            prompt=t.decode(ids[:start].tolist(),skip_special_tokens=True)
            answer=t.decode(target,skip_special_tokens=True)
            status={'retrieval':retrieval_contract,'python':lambda p,a:python_contract(a),
                    'json':lambda p,a:json_contract(a)}.get(source,lambda p,a:'NOT_SEMANTICALLY_SCORED')(prompt,answer)
            row={'row_id':i,'status':status,'active':int(mask.sum()),'supervised':int(valid.sum()),
                 'terminal_eos':target[-1]==t.eos_token_id,'selected_in_plan':(source,i) in chosen,
                 'observed_optimizer_step':(source,i) in used}
            if status in {'SOURCE_NOT_VISIBLE','VISIBLE_SOURCE_TARGET_MISMATCH','INVALID_PYTHON_SYNTAX','INVALID_JSON'}:
                row['prompt']=prompt;row['answer']=answer
            report.append(row)
        summary={}
        for name,subset in [('all',report),('planned',[r for r in report if r['selected_in_plan']]),
                            ('observed_steps',[r for r in report if r['observed_optimizer_step']])]:
            by_status={}
            for status in sorted({r['status'] for r in subset}):
                group=[r for r in subset if r['status']==status]
                by_status[status]={'rows':len(group),'active':sum(r['active'] for r in group),
                    'supervised':sum(planned_rows[(source,r['row_id'])]['supervised']
                                     if name=='planned' else r['supervised'] for r in group)}
            summary[name]={'rows':len(subset),'by_status':by_status,
                           'terminal_eos_rows':sum(r['terminal_eos'] for r in subset)}
        all_summary[source]=summary
        save(out/(source+'.json'),{'shard':info,'rows':report,'summary':summary})
        print(source,json.dumps(summary),flush=True)
    save(out/'summary.json',{'source_shards':all_summary,
        'limitation':'Syntax pass is not proof of complete/correct code; factual answer truncation and global lineage require source comparison. Natural random-span EOS provenance is a code-level finding.',
        'code_sha256':{p.name:digest(p) for p in [Path(__file__),ROOT/'scripts/astra_data_contract.py']}})


if __name__=='__main__':main()
