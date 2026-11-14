from pathlib import Path
import sys,torch
from run_astra_phase2_eval import load,save,continue_ids,PROMPTS,sha
from fdt_rlm.tokenization import load_tokenizer

source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
checkpoint=source/'runs/fdt_v4_1_424m_alpha020_retention_smoke_500k_20260907_r1/latest.pt'
model=load(checkpoint);tokenizer=load_tokenizer(str(source/'tokenizers/fdt_v3_bpe_24k'))
gold={16:'Mira is',17:'The access identifier is',21:'The station is called',22:'The invoice currency is'}
rows=[]
for alpha in (0.,.2,.25):
    model.set_transition_alpha(alpha)
    for index,prefix in gold.items():
        ids=tokenizer.encode(PROMPTS[index])+tokenizer.encode(' '+prefix)
        run=continue_ids(model,tokenizer,ids,64)
        row={'id':index,'alpha':alpha,'legacy_scale':model.legacy_position_scale,'prompt':PROMPTS[index],'gold_prefix':prefix,'run':run}
        rows.append(row);print(alpha,index,run['text'],flush=True)
save(out/'results.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'optimizer_updates':0,'rows':rows})
