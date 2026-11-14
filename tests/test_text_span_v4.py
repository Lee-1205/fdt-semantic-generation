import inspect,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from fdt_local.model_masked import Config
from research.text_generation_data_v1 import build as source_data,render
from research.text_diffusion_v1 import pack_prompts,pack_answers
from research.text_semantic_reader_v3 import SemanticGenerator
from research.text_span_data_v4 import build,query_spans,write_new,finite_pair_orders,local_pairs
from research.text_span_reader_v4 import SpanSemanticGenerator
torch.set_num_threads(2)

def inputs():
 rows=[]
 for lang in ('en','ko'):
  prompt,answer,label=render('A','H','less',True,lang,0)
  rows.append({'prompt':prompt,'answer':answer,'label':label,'language':lang})
 tokens=pack_prompts([r['prompt'] for r in rows])
 mask=torch.cat((tokens!=256,torch.ones(2,1,dtype=torch.bool)),1)
 context=torch.randn(2,161,32)
 spans=torch.tensor([query_spans(r)['span_starts'] for r in rows])
 labels=torch.tensor([query_spans(r)['head_labels'] for r in rows])
 return rows,tokens,mask,context,torch.randn(2,40,32),spans,labels

def test_korean_utf8_span_offsets_and_query_only_reverse_pair():
 rows,*_=inputs()
 ko=rows[1];s=query_spans(ko);raw=ko['prompt'].encode()
 assert raw[s['span_starts'][0]:s['span_ends'][0]]==b'H'
 assert s['span_starts'][0]>ko['prompt'].index('H')
 data=source_data();artifact,stats=build(data,list(range(1024)))
 again,stats2=build(data,list(range(1024)));assert artifact==again and stats==stats2
 assert len(artifact['features']['train'])==1024 and stats['train']['complete_reverse_pairs']>0
 assert all(set(row)=={'id','prompt'} for panel in artifact['features'].values() for row in panel)
 for name,pairs in artifact['pairs'].items():
  for pair in pairs:
   a,b=[artifact['targets'][name][pair[k]] for k in ('first','second')]
   assert a['head_labels'][:2]==b['head_labels'][:2][::-1]

def test_goldfree_pointer_features_distractors_and_padding():
 model=SpanSemanticGenerator(Config.smoke());rows,tokens,mask,prompt,latent,spans,labels=inputs()
 assert list(inspect.signature(model.context).parameters)==['prompt','mask','tokens','latent','zero_latent']
 context,read=model.context(prompt,mask,tokens,latent)
 assert torch.isfinite(context[0]).all()
 assert torch.equal(read['candidate_mask'],(tokens>=65)&(tokens<=72)&mask[:,:160])
 assert not read['candidate_mask'][tokens==256].any()
 assert torch.allclose(read['entity_probabilities'].sum(-1),torch.ones(2,2),atol=1e-6)
 assert (read['pointer_logits'][~read['candidate_mask'][:,None].expand(-1,2,-1)]==-torch.inf).all()
 # How's uppercase H is a raw-byte distractor, not automatically the query role.
 h_positions=(tokens[0]==ord('H')).nonzero().flatten().tolist()
 assert len(h_positions)>1 and int(spans[0,0]) in h_positions

def test_reader_phase_gradients_and_frozen_decoder():
 m=SpanSemanticGenerator(Config.smoke());m.decoder.requires_grad_(False)
 rows,tokens,mask,prompt,latent,spans,labels=inputs()
 _,read=m.context(prompt,mask,tokens,latent)
 loss=m.reader_loss(read,spans,labels);loss.backward()
 assert torch.isfinite(loss) and m.reader.pointer_key.weight.grad.abs().sum()>0
 assert all(p.grad is None for p in m.decoder.parameters())

def test_teacher_realizer_phase_and_evaluation_prohibitions():
 m=SpanSemanticGenerator(Config.smoke());m.reader.requires_grad_(False)
 rows,tokens,mask,prompt,latent,spans,labels=inputs()
 context=m.teacher_context_for_training(labels,latent)
 loss=m.decoder.loss(pack_answers([r['answer'] for r in rows]),context);loss.backward()
 assert torch.isfinite(loss) and m.decoder.head.weight.grad.abs().sum()>0
 assert all(p.grad is None for p in m.reader.parameters())
 _,read=m.context(prompt,mask,tokens,latent)
 m.eval()
 with pytest.raises(RuntimeError):m.teacher_context_for_training(labels,latent)
 with pytest.raises(RuntimeError):m.matched_prediction_mask(read,labels)
 with pytest.raises(RuntimeError):m.consistent_context_for_training(read,labels,latent)

def test_exact_v3_parameter_mapping_preserves_decoder_factor_semantics():
 old=SemanticGenerator(Config.smoke());new=SpanSemanticGenerator(Config.smoke())
 info=new.reader.warmstart_v3(old.state_dict());new.decoder.load_state_dict(old.decoder.state_dict(),strict=True)
 assert set(info['new_keys'])=={'pointer_query.weight','pointer_key.weight','byte_positions.weight'}
 assert torch.equal(new.reader.values[0].weight,old.reader.values[0].weight)
 assert torch.equal(new.reader.semantic_heads[0].weight,old.reader.heads[2].weight)
 labels=inputs()[-1]
 onehots=[torch.nn.functional.one_hot(labels[:,i],n).float() for i,n in enumerate((8,8,3,2))]
 expected=torch.stack([p@old.reader.values[i].weight+old.reader.types[i] for i,p in enumerate(onehots)],1)
 assert torch.equal(new.reader.factor_values(onehots),expected)

def test_no_candidates_fail_closed_without_oracle_fallback():
 model=SpanSemanticGenerator(Config.smoke());tokens=pack_prompts(['abc xyz'])
 mask=torch.cat((tokens!=256,torch.ones(1,1,dtype=torch.bool)),1)
 with pytest.raises(ValueError):model.context(torch.randn(1,161,32),mask,tokens,torch.randn(1,40,32))

def test_artifact_overwrite_rejected_before_source_reads(tmp_path):
 existing=tmp_path/'artifact';existing.mkdir()
 with pytest.raises(FileExistsError):write_new(tmp_path/'missing_source',existing)

def test_transition_uses_prediction_only_when_all_semantics_match():
 m=SpanSemanticGenerator(Config.smoke());labels=torch.tensor([[0,1,0,0],[2,3,1,1]])
 probs=torch.nn.functional.one_hot(labels[:,:2],8).float()
 semantic=[torch.nn.functional.one_hot(labels[:,i],n).float()*10 for i,n in ((2,3),(3,2))]
 semantic[0][1]=torch.tensor([10.,0.,0.]) # Second row's predicted relation is wrong.
 fake={'entity_probabilities':probs,'semantic_logits':semantic,'factors':torch.randn(2,4,32)}
 latent=torch.randn(2,40,32)
 context,matched=m.consistent_context_for_training(fake,labels,latent)
 teacher=m.teacher_context_for_training(labels,latent)
 predicted=m.decoder.context(fake['factors'],torch.ones(2,4,dtype=torch.bool),latent)
 assert matched.tolist()==[True,False]
 assert torch.equal(context[0][0],predicted[0][0]) and torch.equal(context[0][1],teacher[0][1])

def test_finite_pair_batches_no_reset_and_json_cursor_nextbatch():
 import json
 data=source_data();artifact,stats=build(data,list(range(1024)));pairs=artifact['pairs']['train']
 order=finite_pair_orders(pairs,1024,4,31)
 assert len(order)==4096 and order==finite_pair_orders(pairs,1024,4,31)
 for epoch in range(4):assert sorted(order[epoch*1024:(epoch+1)*1024])==list(range(1024))
 for epoch in range(4):
  pair_count=sum(len(local_pairs(order[i:i+8],pairs)) for i in range(epoch*1024,(epoch+1)*1024,8))
  assert pair_count==len(pairs)
 state=json.loads(json.dumps({'order':order,'cursor':2040,'phase':'reader','attempts':255}))
 assert state['order'][state['cursor']:state['cursor']+8]==order[2040:2048]
 assert state['order'][4096:]==[]

def test_pair_semantic_reversal_consistency_not_gold_feature():
 m=SpanSemanticGenerator(Config.smoke())
 entity=torch.nn.functional.one_hot(torch.tensor([[0,1],[1,0]]),8).float().requires_grad_()
 rel=torch.tensor([[9.,-2.,-2.],[-2.,9.,-2.]],requires_grad=True)
 read={'entity_probabilities':entity,'semantic_logits':[rel,torch.zeros(2,2)],'factors':torch.zeros(2,4,32)}
 assert float(m.pair_consistency_loss(read,[(0,1)]))==0
 wrong={**read,'semantic_logits':[rel.flip(0),torch.zeros(2,2)]}
 # Invert only one row's entity prediction to break reversal equivalence.
 wrong['entity_probabilities']=torch.stack((entity[0],entity[0]))
 loss=m.pair_consistency_loss(wrong,[(0,1)]);assert loss>0;loss.backward();assert entity.grad is not None
 m.eval()
 with pytest.raises(RuntimeError):m.pair_consistency_loss(read,[(0,1)])
