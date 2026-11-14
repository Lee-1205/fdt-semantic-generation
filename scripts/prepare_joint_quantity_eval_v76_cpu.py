"""Fresh prospective64 evaluation strings; targets never enter TRAIN/features."""
import collections,hashlib,itertools,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
from prepare_fuzzy_extraction_v73_cpu import CONTEXT,suffixes
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
D=ROOT/'research/artifacts/joint_quantity_eval_20261113_v76_cpu_v1'
NAMES=['Alaric','Blythe','Cyrene','Denzel']
DESCRIPTIONS={'kg':('a mass change','stock weight'),'degrees C':('a temperature change','chamber temperature'),'km':('a displacement change','route length'),'items':('a delivery','remaining stock')}

def main():
    assert not D.exists();rng=random.Random(76114);suffix=suffixes();features=[];targets=[]
    for group,(unit,layout,negative) in enumerate(itertools.product(DESCRIPTIONS,('middle','end'),(False,True))):
        actor=NAMES[group%4];peer=NAMES[(group+1)%4];kind,other=DESCRIPTIONS[unit];sign='-' if negative else '+'
        role=(group*5+7)%16;clauses=[]
        for i,j in itertools.product(range(2),repeat=2):
            a=10+2*group+i;b=50+2*group+j
            A=f'{a} copper washers' if unit=='items' else f'{sign}{a}.75 {unit}'
            B=f'{b} copper washers' if unit=='items' else f'+{b}.25 {unit}'
            clause=(f"Review {7100+group} distinguished {actor}'s {kind} of {A} from the independent {other} that {peer} verified as {B}." if layout=='end' else
                f"The independent {other}, verified by {peer}, was {B} during review {7100+group}; {actor} separately documented {kind} of {A}. Neither figure replaces the other.")
            clauses.append((i,j,a,b,A,B,clause))
        prefix=rng.choice(suffix[(role-len(clauses[0][-1].encode()))%16])
        for i,j,a,b,A,B,clause in clauses:
            text=prefix+clause;identity='eval76:'+hashlib.sha256(text.encode()).hexdigest()[:24]
            names=[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'\b(?:'+'|'.join(NAMES)+r')\b',text)]
            spans=complete_expression_spans(text,names)
            def span(s):
                s=s.removesuffix(' copper washers');matches=list(re.finditer(r'(?<![A-Za-z0-9_])'+re.escape(s)+r'(?![A-Za-z0-9_])',text));assert len(matches)==1
                return dict(start=matches[0].start(),end=matches[0].end(),text=s)
            features.append(dict(id=identity,prompt=text))
            targets.append(dict(id=identity,quartet_id=f'eval76:{group}',member=[i,j],mode='unsigned' if unit=='items' else 'signed',
                unit=unit,layout=layout,sign='' if unit=='items' else sign,actor=actor,peer=peer,A=a,B=b,
                primary_text=A,secondary_text=B,A_span=span(A),B_span=span(B),EOS_position=role,
                clause_start=len(prefix.encode()),clause=clause,critical_spans=spans))
    train=io.load_json(ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1/train_features.json')
    assert len(features)==64 and len({r['prompt'] for r in features})==64
    assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in features for r in train)
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in train for n in NAMES)
    D.mkdir()
    for n,v in [('features',features),('targets',targets)]:io.atomic_json(D/(n+'.json'),v)
    manifest=dict(schema='joint_quantity_v76_prospective_eval',rows=64,correlated_quartets=16,new_prompt_strings=True,
        names_previously_observed_but_never_currentTRAIN=NAMES,EOS_roles=sorted({t['EOS_position'] for t in targets}),
        source_UTF8_bytes=sum(len(r['prompt'].encode()) for r in features),features=['id','prompt'],targets_score_only=True,
        new_natural_or_blind_corpus=False,NN_calls=0,E2_registry_access=0,downloads=0,
        limitations='New authored wrappers after prior observed failures;16 correlated quartets, previously observed held names;EOS role and cell covary, not an isolated positional/length experiment.')
    io.atomic_json(D/'manifest.json',manifest);io.atomic_json(D/'builder_SHA.json',dict(file=str(Path(__file__)),SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(manifest))
if __name__=='__main__':main()
