"""Independent saved-cache review; narrow frozen-D termination gradient diagnostic.

No full codec instantiation, new E/GPU/SGD, target-conditioned inference, or
source memory writes. Three answer partial patches only use a copied frozen D.
"""
import collections,hashlib,json,math,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from torch.nn import functional as F
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.thought_cache_v95 import assemble_input
from research.readout_decision_trace_v88 import decode,masks
from research.commit_patch_readout_v54 import CommitPatchReadout
from research import linked_lastlinear_runtime_v79 as linked


def main():
    began=time.monotonic();torch.set_num_threads(2);rng=torch.get_rng_state().clone()
    run=ROOT/'runs/thought_cache_20261113_v95';p=io.load_json(run/'plan.json');res=io.load_json(run/'result.json')
    terminal=io.load_json(run/'ROOT_terminal_native.json');owner=io.load_json(run/'final_status.json')
    assert terminal['before_analysis'] and terminal['all_owned_native_proxy_gone'] and terminal['python_count']==0
    assert owner['status']=='COMPLETE' and owner['exit_code']==0 and owner['exit']['all_owned_processes_gone']
    refs=[]
    def read(ref):
        path=Path(ref['path']);assert path.resolve().is_relative_to(run.resolve()) and path.stat().st_size==ref['bytes'] and support.sha(path)==ref['SHA']
        refs.append(ref);return torch.load(path,map_location='cpu',weights_only=True)
    static=read(res['static_memories']);data=Path(p['data']);index=io.load_json(run/'input_index.json');count=0
    for split,rows in index.items():
        ff=io.load_json(data/(split+'_features.json'));assert [r['id'] for r in rows]==[f['id'] for f in ff]
        for f,r in zip(ff,rows):
            raw=f['prompt'].encode();sha=hashlib.sha256(raw).hexdigest();d=read(r);m=assemble_input(static,d)
            assert d['source_id']==d['source_SHA']==r['id']==sha
            assert m.memory_revisions==((sha,support.sha(data/'manifest.json'),sha),)
            assert m.source.shape==(1,2+math.ceil(len(raw)/16),512)
            assert not m.exact_valid.any() and not m.exact_keys.count_nonzero() and not m.exact_values.count_nonzero()
            assert all(not t.requires_grad and torch.isfinite(t).all() for t in (m.source,m.prior,m.contextual_long,m.contextual_recent))
            count+=1
    assert count==480
    records=io.load_json(run/'target_index_loss_only.json');answers=io.load_json(data/'answers_loss_only.json')
    metrics=collections.Counter();margins=[];cases=[];fixtures={}
    for r in records:
        key=r['answer_SHA'];text=answers[key];raw=text.encode();n=len(raw);target=read(r['target']);trace=read(r['ceiling_trace']);decoded=decode(trace)
        assert r['expected']==text and decoded['text']==r['output'] and decoded['EOS']==r['EOS'] and decoded['UTF8']==r['UTF8']
        assert target['source_codec_SHA']==p['source_codec_SHA'] and target['answer_SHA']==key and target['semantic_target'].shape==(1,35,512)
        assert target['valid_extent_mask'].shape==(1,35) and target['valid_extent_mask'][0].tolist()==[True]*(3+math.ceil(n/16))+[False]*(32-math.ceil(n/16))
        labels=target['labels'];assert labels.shape==(34,16) and labels.flatten()[:n].tolist()==list(raw) and int(labels.flatten()[n])==256 and (labels.flatten()[n+1:]==257).all()
        for sample,values in zip(trace['sample_patch_indices'].tolist(),trace['sample_full_logits']):
            start=sample*16;assert torch.equal(values.argmax(-1).to(torch.int16),trace['argmax'][start:start+16])
            choices=torch.stack([values.masked_fill(~m,-torch.inf).argmax(-1).to(torch.int16) for m in masks('cpu')])
            assert torch.equal(choices,trace['choices'][:,start:start+16])
            expected=values.gather(-1,labels[sample,:,None]).squeeze(-1)-values.logsumexp(-1)
            assert torch.allclose(expected,trace['gold_log_probability'][start:start+16],atol=3e-5,rtol=1e-5)
        patch=n//16;sample=trace['sample_patch_indices'].tolist().index(patch);z=trace['sample_full_logits'][sample,n%16]
        gap=float(z[256]-z[:256].max());margins.append(gap)
        prefix=decoded['text'].startswith(text);tail=decoded['text'][len(text):] if prefix else None
        metrics.update(dict(answers=1,exact=int(decoded['text']==text),UTF8_EOS=int(decoded['UTF8'] and decoded['EOS']),
            complete_expected_sentence_prefix=int(prefix),byte_prefix_correct=sum(a==b for a,b in zip(raw,decoded['text'].encode()[:n])),target_bytes=n,
            whitespace_only_tail=int(prefix and not tail.strip()),within_patch_EOS_wrong=int(trace['argmax'][n]!=256)))
        cases.append(dict(expected=text,output=decoded['text'],correct_relation_sentence_prefix=prefix,tail=tail,gold_EOS_margin_vs_byte=gap))
        fixtures.setdefault(n,(target,trace,r))
    assert len(records)==31 and set(fixtures)=={14,19,20} and metrics['complete_expected_sentence_prefix']==31
    assert metrics['byte_prefix_correct']==metrics['target_bytes'] and metrics['exact']==res['codec_target_ceiling_exact']==0
    assert res['calls']==dict(encode=511,readout=31,denoise=0) and res['optimizer_steps']==0
    # Narrow diagnostic: three actual partial answer patches. The output
    # parameters remain frozen, but d(loss)/d(meaning) must be available.
    selection=io.load_json(p['selection']);full=linked.load_verified(selection['selected_runtime'],selection['selected_SHA'],ROOT,support.state_sha)
    torch.manual_seed(95115);head=CommitPatchReadout();prefix='readout.readout.'
    head.load_state_dict({k[len(prefix):]:v for k,v in full['model'].items() if k.startswith(prefix)},strict=True);head.requires_grad_(False);head.eval()
    before=support.state_sha(head.state_dict());locals=[];glob=[];partial_labels=[];positions=[];saved=[]
    for n,(target,trace,r) in sorted(fixtures.items()):
        patch=n//16;locals.append(target['semantic_target'][0,2+patch]);glob.append(target['semantic_target'][0,:2]);partial_labels.append(target['labels'][patch]);positions.append(n%16)
        saved.append(trace['sample_full_logits'][trace['sample_patch_indices'].tolist().index(patch)])
    local=torch.stack(locals).detach().requires_grad_(True);global_views=torch.stack(glob).detach();y=torch.stack(partial_labels);ii=torch.arange(3);pos=torch.tensor(positions)
    logits=head(local,global_views,torch.zeros(3,dtype=torch.bool));max_saved_delta=float((logits.detach()-torch.stack(saved)).abs().max())
    byte_mask=(y<256);byte_loss=F.cross_entropy(logits[byte_mask],y[byte_mask]);eos_loss=F.cross_entropy(logits[ii,pos],torch.full((3,),256))
    gb=torch.autograd.grad(byte_loss,local,retain_graph=True)[0];ge=torch.autograd.grad(eos_loss,local)[0]
    # Directional derivative only, NOT latent fitting or a learned research update.
    direction=ge/ge.norm(dim=-1,keepdim=True).clamp_min(1e-12);step=.001
    with torch.no_grad():
        moved=head(local.detach()-step*direction,global_views,torch.zeros(3,dtype=torch.bool))
        new_eos=float(F.cross_entropy(moved[ii,pos],torch.full((3,),256)));new_byte=float(F.cross_entropy(moved[byte_mask],y[byte_mask]))
    assert new_eos<float(eos_loss) and ge.norm()>0 and all(x.grad is None and not x.requires_grad for x in head.parameters())
    assert support.state_sha(head.state_dict())==before and support.sha(selection['selected_runtime'])==selection['selected_SHA']
    torch.set_rng_state(rng);assert torch.equal(torch.get_rng_state(),rng) and not torch.cuda.is_initialized()
    result=dict(status='ROOT_V95_FROZEN_CACHE_AND_TERMINATION_CAUSE_REVIEWED',source_rows=count,loss_only_answers=31,metrics=dict(metrics),
        EOS_margin=dict(min=min(margins),max=max(margins),mean=sum(margins)/31),manual_all31_expected_outputs_read=True,cases=cases,
        interpretation='All31 expected relation/unknown sentences are exact prefixes; extra suffix bytes follow before later neural EOS. Not missing target relation content, NOT whole semantic fidelity or perfect general codec.',
        cause_scope='FrozenD weak/unlearned EOS within partial16-byte patch. Fixed35-slot target carries correct byte prefix but does not repair within-patch stop. Pure latent MSE to these targets would inherit this known termination error.',
        narrow_CPU=dict(full_codec_instantiated=False,encoder_forwards=0,frozen_readout_patch_calls=2,rows_per_call=3,optimizer_steps=0,
            head_SHA_before_after=before,max_CPU_vs_saved_GPU_logit_delta=max_saved_delta,EOS_CE_before=float(eos_loss.detach()),EOS_CE_direction_test=new_eos,
            byte_CE_before=float(byte_loss.detach()),byte_CE_direction_test=new_byte,EOS_latent_gradient_norms=ge.norm(dim=-1).tolist(),byte_latent_gradient_norms=gb.norm(dim=-1).tolist(),
            frozen_head_gradient_None=True,global_Torch_RNG_restored=True,CUDA_initialized=False),
        source_condition_cache_no_answer_fields=True,all_source_cache_SHA_and_revision_checked=True,all_targets_traces_SHA_checked=True,
        all_sampled_full258_argmax_UTF8_choices_goldLP_checked=True,NOT_full258_all_positions_saved=True,model_file_preserved=True,
        optimizer_and_denoise_actual0=True,old_and_this_cache_budgets_closed=True,unused_time_calls_not_reused=True,
        next='Train ONLY new thought weights with latent VP objective PLUS differentiable frozenD byte and separately normalized EOS supervision on TRAIN. No raw/gold length/target in sample; no output crop/string repair; allcodecweightsbuffers frozen. Validate full driver/scaler/RNG/ownedOS in NEWbounded contract before any thought learning.',
        Cfree=shutil.disk_usage(ROOT).free,seconds=time.monotonic()-began)
    assert sum(r['bytes'] for r in refs)==res['tensor_payload_bytes']
    io.atomic_json(run/'root_cache_and_termination_review.json',result)
    io.atomic_json(run/'root_result.json',dict(status='READONLY_CACHE_COMPLETE_TARGET_TERMINATION_LIMIT_IDENTIFIED',plan_SHA=support.sha(run/'plan.json'),
        review_SHA=support.sha(run/'root_cache_and_termination_review.json'),metrics=dict(metrics),actual_training=False,all_owned_gone_before_analysis=True))
    print(json.dumps(dict(status=result['status'],metrics=dict(metrics),EOSmargin=result['EOS_margin'],CPU=result['narrow_CPU'],seconds=result['seconds'])))


if __name__=='__main__':main()
