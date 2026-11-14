import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_fdt_v21_knowledge_scale import V21Writer,add_varied_retrieval


class CharTokenizer:
    eos_token_id=2
    pad_token_id=0
    def encode(self,text,**kwargs):return [ord(c)+20 for c in text]
    def __call__(self,text,**kwargs):
        return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self,ids,**kwargs):return ''.join(chr(i-20) for i in ids if i>20)


def test_grounded_writer_rejects_context_loss_instead_of_cropping_source(tmp_path):
    writer=V21Writer(tmp_path,CharTokenizer(),80,set())
    prompt='Records:\nK00=AB1234\nK01='+('Z'*80)+'\nCopy only the complete value stored at K00.\nAnswer:'
    assert not writer.add_grounded(prompt,'AB1234','retrieval',20)
    assert not writer.ids


def test_writer_never_attaches_eos_to_a_silently_cut_answer(tmp_path):
    writer=V21Writer(tmp_path,CharTokenizer(),512,set())
    assert not writer.add_grounded('Question: name?\nAnswer:','long complete answer','factual_qa',5)
    assert not writer.add_grounded('Python:\n','def f():\n    """unfinished','python_code',96)
    assert not writer.ids


def test_random_natural_crop_has_no_terminal_eos_by_default(tmp_path):
    writer=V21Writer(tmp_path,CharTokenizer(),32,set())
    assert writer.add_ids([40,41],[42,43],'natural_lm')
    assert writer.ids[0][:5]==[40,41,42,43,0]
    assert writer.labels[0][:5]==[-100,-100,42,43,-100]
    assert writer.add_ids([44,45],[46,47],'natural_lm',terminal_eos=True)
    assert writer.ids[1][:5]==[44,45,46,47,2]


def test_retrieval_sampler_advances_after_rejected_layout(tmp_path,monkeypatch):
    writer=V21Writer(tmp_path,CharTokenizer(),512,set());seen=[]
    def accept(prompt,answer,category,*args,**kwargs):
        seen.append(prompt)
        if len(seen)==1:return False
        writer.categories[category]+=1
        return True
    monkeypatch.setattr(writer,'add_grounded',accept)
    add_varied_retrieval(writer,1,17)
    assert len(seen)==2
    # The first rejected length8 layout must advance to length16.
    first=seen[0].split('K00=')[1].splitlines()[0]
    second=seen[1].split('K00=')[1].splitlines()[0]
    assert len(first)==8 and len(second)==16


def test_removing_artificial_eos_does_not_bypass_historical_exact_dedup(tmp_path):
    from prepare_fdt_v21_knowledge_scale import token_hash
    writer=V21Writer(tmp_path,CharTokenizer(),32,{token_hash([40,41,42,43,2])})
    assert not writer.add_ids([40,41],[42,43],'natural_lm')
    assert not writer.ids
