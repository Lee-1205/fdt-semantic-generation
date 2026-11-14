"""128 planned real rows, frozen checkpoint, plus LM-head-only gradient samples."""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_astra_phase2_eval import load,save,sha
from train_fdt_v4_1_retention import TokenBudgetSourcePlanner,chunked_weighted_lm_loss


def main(source_root,out):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    run=source_root/'runs/fdt_v4_1_424m_alpha020_retention_smoke_500k_20260907_r1'
    manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8-sig'))
    model=load(run/'latest.pt');planner=TokenBudgetSourcePlanner(manifest['source_token_fractions'])
    shards={};orders={};cursors={};paths={}
    for name,folder in manifest['dataset_dirs'].items():
        if name not in planner.fractions:continue
        p=next((Path(folder)/'shards/train').glob('*.pt'));paths[name]=p
        shards[name]=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        orders[name]=torch.randperm(len(shards[name]['input_ids']),generator=torch.Generator().manual_seed(manifest['source_states'][name]['seed']))
        cursors[name]=0
    rows=[];head_gradients={}
    for index in range(128):
        name=planner.choose_source();row_id=int(orders[name][cursors[name]]);cursors[name]+=1
        data={k:v[row_id:row_id+1].to(device='cuda',dtype=torch.long) for k,v in shards[name].items() if isinstance(v,torch.Tensor) and k in ('input_ids','labels','attention_mask')}
        planner.record(name,int(data['attention_mask'].sum()))
        with torch.no_grad():hidden=model(data['input_ids'],attention_mask=data['attention_mask'],return_logits=False)['hidden']
        stats=[]
        with torch.set_grad_enabled(name not in head_gradients):
            loss=chunked_weighted_lm_loss(model,hidden,data['labels'],data['attention_mask'],model.config.pad_token_id,model.config.eos_token_id,manifest['eos_loss_weight'],128,False,accounting=stats)
            if name not in head_gradients:
                gradient=torch.autograd.grad(loss,model.lm_head.weight)[0]
                head_gradients[name]={'scope':'LM head only; hidden detached; not full-model gradient share','row_id':row_id,'l2':float(gradient.norm()),'per_weighted_token_loss_l2':float(gradient.norm())}
                del gradient
        record=stats[0];record.update(source=name,row_id=row_id,optimizer_step=index//8,
            configured_coefficient=manifest['base_lm_weight']/8,
            effective_scalar_contribution=record['normalized_source_loss']*manifest['base_lm_weight']/8)
        rows.append(record)
        del hidden,loss,data
    sources={}
    for name in planner.fractions:
        subset=[r for r in rows if r['source']==name]
        sources[name]={key:sum(r[key] for r in subset) for key in ['active_tokens','supervised_tokens','raw_loss_sum','weighted_loss_sum','loss_denominator','effective_scalar_contribution']}
        sources[name]['rows']=len(subset);sources[name]['planned_active_share']=planner.fractions[name]
    totals={key:sum(v[key] for v in sources.values()) for key in sources[next(iter(sources))] if key not in ('planned_active_share',)}
    for value in sources.values():
        for key in ['active_tokens','supervised_tokens','effective_scalar_contribution']:
            value[key+'_share']=value[key]/totals[key]
    save(out,{'scope':'first128plannedrows-frozen-latest-checkpoint','checkpoint':str(run/'latest.pt'),
        'checkpoint_sha256':sha(run/'latest.pt'),'shards':{n:{'path':str(p),'sha256':sha(p)} for n,p in paths.items()},
        'rows':rows,'sources':sources,'head_gradient_samples':head_gradients,
        'current_row_mean':sum(r['normalized_source_loss'] for r in rows)/len(rows),
        'counterfactual_global_weighted_token_mean':sum(r['weighted_loss_sum'] for r in rows)/sum(r['loss_denominator'] for r in rows),
        'optimizer_updates':0,'full_model_gradient_influence':'UNKNOWN'})
    print(json.dumps(sources),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
