"""Read saved text only: graded lexical similarity and ROOT proposition review.

No model forward, optimizer, download, embedding judge or new evaluation prompts.
chrF implementation: character1..6, whitespace removed, case-sensitive, beta2,
mean ngram precision/recall then F2; sentence-macro aggregate, 0..100 scale.
These text similarities are not calibrated semantic-equivalence probabilities.
"""
import hashlib,json,re,statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/artifacts/owt_free_similarity_20261028_cpu'
STOP=set('a an the and or but if then than of to in on at by for from with as is are was were be been being it its this that these those he she they we i you his her their our my your have has had do does did not no any all can could would will should more most some into through about also up down out so who which what when where how'.split())
REVIEWS=[
 ('Denial of personal culpability for deceiving the American public','Political-sounding words only; culpability, deception and negation are not faithfully expressed.'),
 ('Fans wave light-up flowers and stand in anticipation','Opening Fanstile resembles fans; actions and accessories are absent.'),
 ('Disney donut sundae and surprised quoted question','SUS/OS byte-like fragments; the source proposition is absent.'),
 ('Few ransom payments because victims can restore backups','Ransom, victims, causal backup restoration are absent.'),
 ('A panel depicts an assault team facing Slade Wilson in Moscow','Attack is loosely related; named people, location and panel relation absent.'),
 ('Israel-Syria ceasefire during Assad family rule','No preserved Israel, Syria, ceasefire or Assad relation.'),
 ('She failed sobriety testing and was charged with DUI','She survives; failure, test and DUI charge absent.'),
 ('I walk away from a stall without paying for taken items','I then and pay survive; get pay does not preserve without-paying relation.'),
 ('Speech restrictions are effective; therefore no explosion','But more survives; negation, causal claim and speech restriction absent.'),
 ('Fabbri was barely on two named people\u2019s radar three months ago','Named entities, time and scouting relation absent.'),
 ('A second academic reviews an anonymous article copy','We survives; review, second academic and anonymous copy absent.'),
 ('Wubbels and lawyer name others\u2019 inaction as a main concern','Complaints loosely resembles concern; people and inaction relation absent.'),
 ('Gunmen hold vessels from six listed countries','List-like surface only; gunmen, vessels and country list absent.'),
 ('Jamaican two-man bobsled qualifies for Sochi Winter Olympics','Invented numeral2019999; qualification, team and location absent.'),
 ('Janissa prevents assassination using a wall','Japansians is not Janissa; prevention, assassination and wall absent.'),
 ('One function returns a temporary string; another builds pointers into it','starting/string resemblance only; named functions, temporary object and pointer relation absent.')
]

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def edit(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]
def chrf(ref,hyp):
    a=''.join(ref.split());b=''.join(hyp.split());ps=[];rs=[]
    for n in range(1,7):
        aa=Counter(a[i:i+n] for i in range(len(a)-n+1));bb=Counter(b[i:i+n] for i in range(len(b)-n+1))
        hit=sum((aa&bb).values());ps.append(hit/sum(bb.values()) if bb else 0);rs.append(hit/sum(aa.values()) if aa else 0)
    p=statistics.mean(ps);r=statistics.mean(rs)
    return 100*5*p*r/(4*p+r) if p*r else 0.
def words(s):return re.findall(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z]+)?",s.lower())
def f1(ref,hyp):
    r=Counter(ref);h=Counter(hyp);hit=sum((r&h).values());return 2*hit/(sum(r.values())+sum(h.values())) if r or h else 1.
def lcs(a,b):
    row=[0]*(len(b)+1)
    for x in a:
        nxt=[0]
        for j,y in enumerate(b):nxt.append(row[j]+1 if x==y else max(nxt[-1],row[j+1]))
        row=nxt
    return row[-1]
def prefix(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    return n
def metrics(ref,hyp):
    rw,hw=words(ref),words(hyp);rc=[x for x in rw if x not in STOP];hc=[x for x in hw if x not in STOP]
    return {'chrF2_1to6':chrf(ref,hyp),'byte_edit_similarity':1-edit(ref.encode(),hyp.encode())/max(len(ref.encode()),len(hyp.encode()),1),
        'word_F1':f1(rw,hw),'content_word_F1':f1(rc,hc),'ROUGE_L_word_F1':2*lcs(rw,hw)/max(len(rw)+len(hw),1),
        'correct_initial_UTF8_bytes':prefix(ref.encode(),hyp.encode()),'source_bytes':len(ref.encode()),'output_bytes':len(hyp.encode()),
        'shared_content_words':sorted(set(rc)&set(hc))}

def main():
    assert chrf('Exact same.','Exact same.')==100 and chrf('abc','XYZ')==0
    OUT.mkdir(exist_ok=False);allrows={};summaries={};files={}
    for version in (12,13,14):
        p=ROOT/f'runs/owt_fuzzy_reconstruction_20261028_v{version}/final_generation.json';files[str(p)]=digest(p);raw=load(p)
        allrows[str(version)]={};summaries[str(version)]={}
        for mode,rows in raw.items():
            scored=[]
            for i,row in enumerate(rows):
                item={'index':i,'id':row['id'],'source':row['source'],'output':row['generated']['text'],**metrics(row['source'],row['generated']['text'])}
                if version==14 and mode=='normal':item.update(source_proposition=REVIEWS[i][0],ROOT_reading=REVIEWS[i][1],faithful_paraphrase=False)
                scored.append(item)
            allrows[str(version)][mode]=scored;summary={k:statistics.mean(x[k] for x in scored) for k in ('chrF2_1to6','byte_edit_similarity','word_F1','content_word_F1','ROUGE_L_word_F1','correct_initial_UTF8_bytes')}
            summary['median_correct_initial_UTF8_bytes']=statistics.median(x['correct_initial_UTF8_bytes'] for x in scored)
            summary['any_exact_content_word_rows']=sum(bool(x['shared_content_words']) for x in scored)
            refs=[x['source'] for x in scored];own=0;off=[];margin=[]
            for i,row in enumerate(scored):
                scores=[chrf(ref,row['output']) for ref in refs];best=max(scores)
                own+=scores[i]==best and scores.count(best)==1;other=[v for j,v in enumerate(scores) if j!=i]
                off.extend(other);margin.append(scores[i]-statistics.mean(other))
            summary.update(own_source_unique_top1_of16=own,own_vs_other15_chrF_margin=statistics.mean(margin),other_source_mean_chrF=statistics.mean(off))
            summaries[str(version)][mode]=summary
    result={'status':'SAVED_TEXT_SIMILARITY_AND_ROOT_READING_COMPLETE','source_files_sha256':files,'summaries':summaries,'rows':allrows,
        'ROOT_V14_reading':{'reviewed':16,'faithful_semantic_paraphrase':0,'not_independent_or_blind_judgment':True},
        'metric_definition':'Case-sensitive whitespace-stripped char1..6 beta2 macro P/R; Unicode alphanumeric word F1 and word-LCS F1; fixed English stop list content F1; byte edit normalized by max length. Posthoc descriptive metrics, not calibrated semantic truth.',
        'limitations':['Same previously observed16devdocuments, no new evaluation or inference','chrF/word overlap can reward function words or spelling fragments; no pretrained semantic scorer was run','ROOT proposition judgment is qualitative and unblinded','State controls are content-use evidence, not proof denoised semantics are complete'],
        'additional_model_forwards':0,'optimizer_updates':0,'E2_final_accessed':False,
        'primary_sources':['https://aclanthology.org/W15-3049/','https://arxiv.org/abs/1904.09675','https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf']}
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Saved free reconstruction: graded text similarity and ROOT reading','', 'Posthoc, no new model forward. chrF is lexical overlap, not a semantic-equivalence probability.','']
    for version in summaries:
        lines.append(f'V{version}: '+json.dumps(summaries[version],ensure_ascii=False))
    for row in allrows['14']['normal']:
        lines.extend(['',f"## {row['index']}",f"Source: {row['source']}",f"Output: {row['output']}",f"ROOT reading: {row['ROOT_reading']}",f"chrF={row['chrF2_1to6']:.3f}; content F1={row['content_word_F1']:.4f}; shared={row['shared_content_words']}"])
    (OUT/'all16_reading.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=True))
if __name__=='__main__':main()
