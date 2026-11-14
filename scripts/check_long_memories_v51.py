"""Read-only isolated memory ablations and source integrity on the saved long codec."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from research.trainable_long_codec_v50 import TrainableLongCodec
from research.text_diffusion_v1 import pack_prompts,pack_answers
import text_generation_support_v1 as support
import learning_io_v2 as io

def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_long_memory_readonly_v51' or p['output']!=str(output.resolve())
        or p['max_calls']!=64 or p['indices']!=list(range(0,16,2)) or p['wall_seconds']!=180
        or p['cutoff']!=p['deadline']-30 or p['deadline']-p['start']!=180):raise ValueError('readonly plan')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('changed pin '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('core changed')
    return p

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();p=validate(a.plan,a.output)
    if (a.output/'worker_started.json').exists() or time.time()>=p['cutoff']:raise ValueError('one fresh readonly check')
    torch.set_num_threads(4);began=time.time()
    io.atomic_json(a.output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=began))
    state=torch.load(p['source'],map_location='cpu',weights_only=True)
    model=TrainableLongCodec(Config(**state['core']),input_capacity=state['input_capacity'])
    model.enable_position_adapter();model.enable_long_capacity();model.load_state_dict(state['model'],strict=True)
    model=model.to('cuda').eval();del state
    rows=io.load_json(Path(p['data'])/'dev.json');calls=0;results=[]
    def charge():
        nonlocal calls
        if time.time()>=p['cutoff'] or (a.output/'STOP').exists():raise InterruptedError('fixed clock/STOP')
        if calls>=64:raise ValueError('readonly call cap')
        calls+=1
    for i in p['indices']:
        row=rows[i];text=row['prompt'];raw=text.encode();length=len(raw)
        x=pack_prompts([text],length,'cuda');y=pack_answers([text],length+1,'cuda')
        scores={};commits={};logit_delta={};normal_logits=None;checks={}
        for disabled in (None,'persistent','contextual','exact'):
            charge()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ep=model.encode_input(x,source_ids=(row['id'],),source_versions=(p['data_sha'],),model_epoch='memorycheck',
                    disabled_memories=() if disabled is None else (disabled,))
                charge();logits=model.teacher_logits(ep.commit,y)
            key=disabled or 'normal';commits[key]=ep.commit.reader_tensor().cpu()
            nll=torch.nn.functional.cross_entropy(logits.transpose(1,2),y)
            scores[key]=dict(NLL=float(nll),accuracy=float((logits.argmax(-1)==y).float().mean()))
            if disabled is None:
                normal_logits=logits
                exact=ep.memories.exact;source=exact.sources[0]
                spans=exact.spans[0]
                offset=0
                for start,end in spans:
                    assert start==offset;source.span(start,end).decode('utf-8');offset=end
                assert offset==length and source.raw==raw and source.sha256==hashlib.sha256(raw).hexdigest()
                weights=ep.diagnostics['candidate_weights'];null=ep.diagnostics['null_weight']
                torch.testing.assert_close(weights.sum(-1)+null,torch.ones_like(null),atol=2e-6,rtol=2e-6)
                ids=ep.diagnostics['candidate_ids'];assert int(ids.max())<len(spans)
                try:exact.assert_epoch('wrong-epoch')
                except ValueError:stale_rejected=True
                else:raise ValueError('stale Exact index accepted')
                # Independent sequential recurrence on CPU, not the bank's vectorized formula.
                mu=ep.diagnostics['token_memberships'][0].double().cpu()
                value=ep.diagnostics['token_values'][0].double().cpu()
                bank=ep.memories.contextual;decay=bank.decay.double().cpu()
                N=torch.zeros_like(bank.long_N[0].cpu());D=torch.zeros_like(bank.long_D[0].cpu())
                RN=N.clone();RD=D.clone()
                for membership,v in zip(mu,value):
                    mass=membership[:,None];N+=mass*v[None];D+=mass
                    RN=decay[:,None]*RN+mass*v[None];RD=decay[:,None]*RD+mass
                for actual,expected in [(bank.long_N[0].cpu(),N),(bank.long_D[0].cpu(),D),
                    (bank.recent_N[0].cpu(),RN),(bank.recent_D[0].cpu(),RD)]:
                    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-7)
                assert bool(((mu>=0)&(mu<=1)).all())
                checks=dict(exact_raw_hash_spans=True,exact_top16_plus_NULL_normalized=True,stale_exact_rejected=stale_rejected,
                    fuzzy_N_D_full_sequential_recurrence=True,fuzzy_memberships_0_1=True,
                    candidate_codepoints=len(spans),Commit_slots=commits[key].shape[1],
                    exact_null_mean=float(null.mean()),persistent_scope='learned prior read; certified write/consolidation not implemented')
            else:
                logit_delta[key]=float((logits-normal_logits).abs().max())
            del ep
        del normal_logits,logits
        commit_delta={k:float((v-commits['normal']).abs().max()) for k,v in commits.items() if k!='normal'}
        assert bool((commits['contextual'][:,2:]==0).all())
        results.append(dict(index=i,bytes=length,scores=scores,checks=checks,Commit_max_delta=commit_delta,logit_max_delta=logit_delta))
        io.atomic_json(a.output/'progress.json',dict(phase='readonly_memory_checks',completed=len(results),calls=calls,
            native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    io.atomic_json(a.output/'result.json',dict(status='MEMORY_READONLY_CHECK_COMPLETE',rows=results,actual_model_calls=calls,
        optimizer_steps=0,checkpoint_writes=0,denoise_calls=0,copy_bytes=0,seconds=time.time()-began,
        source_unchanged=support.sha(p['source'])==p['source_sha']))
    return 0
if __name__=='__main__':raise SystemExit(main())
