"""Resumable inference-only evidence runner. No optimizer is constructed."""
from pathlib import Path
import sys,json,hashlib,time,random,argparse,importlib.util,gc
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_rlm.models import build_model
from fdt_rlm.config import ModelConfig
from fdt_rlm.tokenization import load_tokenizer
from fdt_rlm.runtime_contract import runtime_capabilities,validate_generation_budget
from audit_astra_goldfree_exact import cases,generate as exact_generate,score
from astra_evidence_io import save, completed, ensure_run_identity


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    model=build_model(ModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model_state_dict'],strict=True)
    return model.to(device='cuda',dtype=torch.float32).eval()


PROMPTS=[
 'The gardener placed the seedlings beside the window. Each morning she checked',
 'Rain had washed the dust from the village road. As the bus arrived,',
 'A library keeps books in a consistent order so that visitors can',
 'During winter, the lake freezes first near the shore because',
 'The engineer tested the bridge with a small load before',
 'The photographer waited until the sun was low. The softer light',
 'To compare two explanations fairly, a researcher should',
 'A good map shows both distances and landmarks. Without a scale,',
 'A seed needs water, suitable temperature, and oxygen to germinate. Water helps the seed',
 'Ice floats on liquid water because its solid structure occupies more space. Therefore its density',
 'The Moon reflects sunlight rather than producing its own light. Its appearance changes because',
 'A bicycle brake slows the wheel by friction. When the rider squeezes the lever,',
 'A computer stores a file as data. To retrieve that file later, the system needs',
 'A recipe calls for two cups of flour for four servings. To make eight servings,',
 'An experiment has a control group and a treatment group. The control group allows',
 'A dictionary explains word meanings, while a timetable lists departures. To find when a train leaves,',
 'Record: Mira is 74 years old. Question: How old is Mira? Answer:',
 'Record: The access identifier is LV320072. Question: What is the access identifier? Answer:',
 'Record: The parcel belongs to Nadia, not Omar. Question: Who owns the parcel? Answer:',
 'Record: The blue box contains 17 screws. Question: How many screws are in the blue box? Answer:',
 'Record: The meeting starts at 09:40. Question: When does the meeting start? Answer:',
 'Record: The station is named Elmbrook. Question: What is the station called? Answer:',
 'Record: The invoice currency is EUR. Question: What is the invoice currency? Answer:',
 'Record: The server port is 8123. Question: Which port does the server use? Answer:',
 'Input record: name=Mira, age=74. The equivalent JSON object is',
 'Input record: status=ready, count=3. The equivalent JSON object is',
 'The list is [2, 4, 6]. Adding one to each entry gives',
 'A Python function that returns the sum of two numbers begins:\ndef add(a, b):\n',
 'The CSV header is name,score. Mira scored 17. Her CSV row is',
 'The original sentence is "The cat sleeps." Replacing cat with dog gives',
 'A record contains key=LV320072 and enabled=true. Its JSON representation is',
 'To reverse the list [1, 2, 3] without changing its values, write',
]


@torch.inference_mode()
def continue_ids(model,tokenizer,prompt_ids,budget):
    validate_generation_budget(model.config,len(prompt_ids),budget)
    ids=torch.tensor([prompt_ids],device='cuda');out=[];entropy=[];started=time.monotonic()
    for _ in range(budget):
        result=model(ids,attention_mask=torch.ones_like(ids))
        logits=result['logits'][:,-1].float()
        if not bool(torch.isfinite(logits).all()):raise RuntimeError('nonfinite generation logits')
        p=logits.softmax(-1);entropy.append(float(-(p*p.clamp_min(1e-30).log()).sum()))
        token=int(logits.argmax(-1));out.append(token)
        if token==model.config.eos_token_id:break
        ids=torch.cat((ids,torch.tensor([[token]],device='cuda')),1)
    eos=bool(out and out[-1]==model.config.eos_token_id)
    return {'ids':out,'text':tokenizer.decode(out,skip_special_tokens=True),'entropy':entropy,
            'eos':eos,'early_eos_operational_lt8':eos and len(out)<8,
            'max_token_exhaustion':not eos and len(out)==budget,'seconds':time.monotonic()-started}


@torch.inference_mode()
def preflight(models,tokenizer,original_root):
    import fdt_rlm.models.fdt_v3 as current
    spec=importlib.util.spec_from_file_location('fdt_rlm.models.astra_original_v3',original_root/'src/fdt_rlm/models/fdt_v3.py')
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    rows=[]
    for name,model in models.items():
        ids=torch.tensor([tokenizer.encode(PROMPTS[16])],device='cuda')
        current_fn=current.sparse_chunked_prefix_summaries
        actual=model(ids,attention_mask=torch.ones_like(ids))['logits'][:,-1]
        try:
            current.sparse_chunked_prefix_summaries=legacy.sparse_chunked_prefix_summaries
            baseline=model(ids,attention_mask=torch.ones_like(ids))['logits'][:,-1]
        finally:current.sparse_chunked_prefix_summaries=current_fn
        error=float((actual-baseline).abs().max())
        rows.append({'model':name,'runtime':runtime_capabilities(model.config),
            'parameters':sum(p.numel() for p in model.parameters()),
            'safe_short_old_new_max_logit_error':error,'tolerance':3e-4,
            'pass':error<=3e-4 and bool(torch.isfinite(actual).all())})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['preflight','generation','exact','cache'],required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20260916)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths={'V20':args.source_root/'runs/fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2/latest.pt',
           'Diana':args.source_root/'runs/fdt_v4_1_424m_alpha020_retention_smoke_500k_20260907_r1/latest.pt'}
    tokpath=args.source_root/'tokenizers/fdt_v3_bpe_24k'
    selected={name:path for name,path in paths.items() if args.mode in ('preflight','generation') or name=='Diana'}
    identity={'schema':1,'mode':args.mode,'seed':20260916,'dtype':'float32','tf32':False,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'code_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('src','scripts') for p in sorted((ROOT/d).rglob('*.py'))},
        'tokenizer_sha256':{p.name:sha(p) for p in sorted(tokpath.iterdir()) if p.is_file()},
        'checkpoints':{name:{'path':str(path.resolve()),'sha256':sha(path)} for name,path in selected.items()}}
    ensure_run_identity(args.out,identity)
    tokenizer=load_tokenizer(str(tokpath))
    models={name:load(path) for name,path in paths.items() if args.mode in ('preflight','generation') or name=='Diana'}
    meta={'mode':args.mode,'gpu':torch.cuda.get_device_name(0),'dtype':'float32','tf32':False,
          'algorithm':'greedy','repetition_penalty':'off','ngram_penalty':'off','backend':'full_recompute',
          'optimizer_updates':0,'tokenizer_sha256':sha(tokpath/'tokenizer.json'),
          'checkpoints':{k:{'path':str(paths[k]),'sha256':sha(paths[k]),'runtime':runtime_capabilities(v.config)} for k,v in models.items()}}
    if not completed(args.out/'metadata.json'):save(args.out/'metadata.json',meta)
    if args.mode=='preflight':
        if not completed(args.out/'preflight.json'):
            result=preflight(models,tokenizer,args.source_root);save(args.out/'preflight.json',result);print(json.dumps(result),flush=True)
    elif args.mode=='exact':
        for case in cases():
            path=args.out/f"case_{case['id']:03d}.json"
            if completed(path):continue
            run=exact_generate(models['Diana'],tokenizer,case['records'],case['query'])
            save(path,{'case':case,'run':run,'metrics':score(case,run,tokenizer,models['Diana'].config.eos_token_id)})
            print('exact',case['id'],flush=True)
    elif args.mode=='cache':
        from audit_astra_cache import audit_context
        model=models['Diana']
        unsupported=[audit_context(model,n) for n in [63,64,65,511,512,513,2048,4096,8192,16383]]
        if not completed(args.out/'current_candidate.json'):save(args.out/'current_candidate.json',unsupported)
        model.set_transition_alpha(0.)
        rows=[]
        for n in [63,64,65,511,512,513,2048]:
            if completed(args.out/f'alpha0_{n}.json'):continue
            r=audit_context(model,n);save(args.out/f'alpha0_{n}.json',r);rows.append(r);print('cache',n,r['status'],flush=True)
    else:
        for index,prompt in enumerate(PROMPTS):
            path=args.out/f'cross_{index:03d}.json'
            if completed(path):continue
            prompt_ids=list(tokenizer.encode(prompt));prefix={};continuations={}
            for name,model in models.items():prefix[name]=continue_ids(model,tokenizer,prompt_ids,32)
            for producer,p in prefix.items():
                for consumer,model in models.items():
                    if p['eos']:
                        continuations[producer+'->'+consumer]={'status':'PREFIX_EOS','ids':[],'text':''}
                    else:continuations[producer+'->'+consumer]=continue_ids(model,tokenizer,prompt_ids+p['ids'],64)
            save(path,{'id':index,'prompt':prompt,'prompt_ids':prompt_ids,'prefixes':prefix,'continuations':continuations})
            print('cross',index,flush=True)
        for index in range(16,24):
            path=args.out/f'gold_{index:03d}.json'
            if completed(path):continue
            gold=['Mira is','The access identifier is','The parcel belongs to','The blue box contains','The meeting starts at','The station is called','The invoice currency is','The server uses port'][index-16]
            ids=tokenizer.encode(PROMPTS[index])+tokenizer.encode(' '+gold)
            save(path,{'id':index,'prompt':PROMPTS[index],'gold_prefix':gold,'runs':{n:continue_ids(m,tokenizer,ids,64) for n,m in models.items()}})
        for i,(old,new) in enumerate([('74','31'),('LV320072','QZ851904'),('Nadia','Lina'),('17','42')]):
            base=PROMPTS[16+i]
            for variant,text in [('original',base),('changed',base.replace(old,new)),('omitted',base.replace(old,'[not provided]'))]:
                path=args.out/f'ground_{i}_{variant}.json'
                if completed(path):continue
                save(path,{'id':i,'variant':variant,'prompt':text,'expected':new if variant=='changed' else old if variant=='original' else None,
                    'runs':{n:continue_ids(m,tokenizer,tokenizer.encode(text),64) for n,m in models.items()}})
                print('grounding',i,variant,flush=True)
    print('COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
