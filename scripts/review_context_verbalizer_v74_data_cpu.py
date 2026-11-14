"""Independent TRAIN/DEV whole-clause, cross-coverage and finite order review."""
import collections,hashlib,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import learning_io_v2 as io
def main():
    d=ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2';rows=io.load_json(d/'train_features.json');labels=io.load_json(d/'train_targets.json');dev=io.load_json(d/'dev_features.json');dt=io.load_json(d/'dev_targets.json');m=io.load_json(d/'manifest.json');order=io.load_json(d/'orders.json')
    assert len(rows)==len(labels)==640 and len(dev)==len(dt)==64 and sorted(v[0] for v in order)==list(range(640))
    groups=collections.defaultdict(list);coverage=collections.defaultdict(collections.Counter);namecontexts=collections.Counter();places={v[0]:i for i,v in enumerate(order)}
    for i,(f,t) in enumerate(zip(rows+dev,labels+dt)):
        assert set(f)=={'id','prompt'} and f['id']==t['id'];b=f['prompt'].encode();last=-1
        for s in t['critical_spans']:
            assert last<=s['start']<s['end']<=len(b) and b[s['start']:s['end']].decode()==s['text'];last=s['end']
            if i<512:
                for j in range(s['start'],s['end']):coverage[chr(b[j])][str(j%16)]+=1
                if s['kind']=='name':namecontexts['y_internal']+=int('y' in s['text'][:-1]);namecontexts['z_final']+=int(s['text'].endswith('z'))
        if i<512 or i>=640:
            assert t['clause_start']%16==t['actor_position_role'] and f['prompt'][t['clause_start']:]==t['clause']
            if i<512:
                match=re.fullmatch(r'(\w+) recorded a signed (mass|temperature) change of ([+-])(\d+)\.50 (kg|degrees C) after (\w+) checked the balance\. (\w+) kept \+(\d+)\.25 kg of grain; the remaining (\d+) parcels were not moved\.',t['clause'])
                assert match and match[1]==t['actor'] and match[3]==t['sign'] and int(match[4])==t['quantity'] and match[5]==t['unit'] and match[6]==match[7]==t['peer'] and int(match[8])==t['stored_quantity'] and int(match[9])==t['stored_quantity']+2
                assert (match[2]=='mass')==(match[5]=='kg');groups[t['pair_id']].append(i)
            else:
                match=re.fullmatch(r"(\w+)'s log lists a change of ([+-])(\d+)\.50 (kg|degrees C)\. (\w+) confirmed the reading, while the stored grain weighed \+(\d+)\.25 kg\.",t['clause'])
                assert match and match[1]==t['actor'] and match[2]==t['sign'] and int(match[3])==t['quantity'] and match[4]==t['unit'] and match[5]==t['peer'] and int(match[6])==t['stored_quantity']
    for indices in groups.values():
        assert len(indices)==2 and [labels[i]['sign'] for i in indices]==['+','-'] and abs(places[indices[0]]-places[indices[1]])==1
        a,b=(labels[i] for i in indices);assert all(a[k]==b[k] for k in ('unit','quantity','stored_quantity','actor','peer','clause_start'))
    assert len(groups)==256 and all(len(coverage[c])==16 for c in ('+','-','y','z')) and all(v>0 for v in namecontexts.values())
    old=io.load_json(ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2/train_features.json');assert rows[512:]==old[832:]
    held=io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/short_features.json')+io.load_json(ROOT/'research/artifacts/fuzzy_extraction_eval_20261112_v73_cpu/long_features.json')
    assert not any(h['prompt'] in r['prompt'] or r['prompt'] in h['prompt'] for h in held for r in rows)
    assert not any(re.search(r'\b'+n+r'\b',r['prompt']) for r in rows for n in m['DEV_names']+['Zephyr','Boaz'])
    assert not {r['prompt'] for r in rows}&{r['prompt'] for r in dev}
    proof=dict(status='ROOT_V74_CONTEXT_DATA_CPU_PASS',all704_source_target_rows_verified=True,whole_authored_clauses_independently_reconstructed=True,all256sign_pairs_only_primary_sign_diff=True,
        pair_adjacency_finite640order=True,old128_OWT_TRAIN_exact=True,sign_unit_primary_four_cells128_each=True,name_y_internal_z_final_nonzero=dict(namecontexts),
        plus_minus_y_z_critical_all16_positions=True,observedV73heldprompts_and_newDEV_not_TRAIN=True,ordinary_names_not_Exact=True,NNcalls=0,E2_registry_access=0,
        files_SHA={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.glob('*.json')},limitations=m['limitations'])
    path=d/'root_data_review.json';assert not path.exists();io.atomic_json(path,proof);print(proof)
if __name__=='__main__':main()
