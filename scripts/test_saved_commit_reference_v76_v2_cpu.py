"""Small packet compatibility contracts; no research model or CUDA execution."""
import copy,hashlib,json,shutil,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import torch
from saved_commit_reference_v76_v2 import resolve_saved_commit,sha
D=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1/legacy_reference_fixture_v1'

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.disk_usage(ROOT).free<100*1024**3+1024**2:raise OSError('100GiB+1MiB CPU guard')
        D.mkdir(exist_ok=False);cls.row=dict(id='fixture',prompt='A tiny source.')
        cls.packet=dict(source_id='fixture',source_SHA=hashlib.sha256(cls.row['prompt'].encode()).hexdigest(),
            embedding_sha='fixture_upstream',Exact_entries=0,targets=torch.tensor([[65,256,257]]),fuzzy_commit=torch.arange(96.).reshape(1,3,32))
        cls.path=(D/'old_direct.pt').resolve();torch.save(cls.packet,cls.path)
        cls.ref=dict(path=str(cls.path),SHA=sha(cls.path),key='fuzzy_commit',embedding_sha='fixture_upstream')
        cls.linked={k:v for k,v in cls.packet.items() if k!='fuzzy_commit'};cls.linked['fuzzy_commit_reference']=cls.ref
        cls.pins={str(cls.path):cls.ref['SHA']}
    def resolve(self,p,pins=None):
        return resolve_saved_commit(p,self.row,'fixture_upstream',self.pins if pins is None else pins,ROOT,32)
    def test_direct_and_one_level_reference_same_tensor(self):
        a,_=self.resolve(self.packet);b,r=self.resolve(self.linked)
        self.assertTrue(torch.equal(a,b));self.assertEqual(r,self.ref)
    def test_unpinned_wrong_SHA_and_wrong_embedding_rejected(self):
        with self.assertRaises(ValueError):self.resolve(self.linked,{})
        for key,value in [('SHA','wrong'),('embedding_sha','wrong')]:
            p=copy.deepcopy(self.linked);p['fuzzy_commit_reference'][key]=value
            with self.assertRaises(ValueError):self.resolve(p)
    def test_wrong_source_Exact_and_shape_rejected(self):
        for k,v in [('source_id','wrong'),('source_SHA','wrong'),('Exact_entries',1),('fuzzy_commit',torch.ones(1,2,32))]:
            p=copy.deepcopy(self.packet);p[k]=v
            with self.assertRaises(ValueError):self.resolve(p)
    def test_targets_are_integrity_only_and_mismatch_rejected(self):
        p=copy.deepcopy(self.linked);p['targets'][0,0]=66
        with self.assertRaises(ValueError):self.resolve(p)
    def test_nested_or_ambiguous_reference_rejected(self):
        p=copy.deepcopy(self.linked);p['fuzzy_commit']=self.packet['fuzzy_commit']
        with self.assertRaises(ValueError):self.resolve(p)
        nested=(D/'nested.pt').resolve();torch.save(self.linked,nested);p=copy.deepcopy(self.linked)
        p['fuzzy_commit_reference']['path']=str(nested);p['fuzzy_commit_reference']['SHA']=sha(nested)
        with self.assertRaises(ValueError):self.resolve(p,{str(nested):sha(nested)})


    def legacy(self,change=None):
        policy='DORMANT_UNTIL_FUTURE_EXPLICIT_INTERNAL_COPY_REQUEST'
        old={k:copy.deepcopy(v) for k,v in self.packet.items() if k not in ('source_id','source_SHA','embedding_sha')}
        old.update(logits=torch.zeros(1,3,258),Exact_policy=policy,evaluation_mode='saved_V73_identical_upstream_readout')
        if change:old.update(change)
        file=(D/('legacy_'+str(len(list(D.glob('legacy_*.pt'))))+'.pt')).resolve();torch.save(old,file)
        packet=copy.deepcopy(self.linked);packet['Exact_policy']=policy
        packet['fuzzy_commit_reference'].update(path=str(file),SHA=sha(file))
        return packet,{str(file):sha(file)}
    def test_known_legacy_metadata_comes_from_strict_referencer(self):
        packet,pins=self.legacy();tensor,_=self.resolve(packet,pins)
        self.assertTrue(torch.equal(tensor,self.packet['fuzzy_commit']))
        with self.assertRaises(ValueError):self.resolve(packet,{})
    def test_legacy_partial_metadata_policy_and_unknown_schema_rejected(self):
        for change in ({'source_id':'fixture'},{'Exact_policy':'wrong'},{'unexpected_field':1},{'Exact_entries':1}):
            packet,pins=self.legacy(change)
            with self.assertRaises(ValueError):self.resolve(packet,pins)

if __name__=='__main__':
    torch.set_num_threads(2);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert not torch.cuda.is_initialized()
    proof=dict(status='V76_ONE_LEVEL_COMMIT_REFERENCE_CPU_PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,source_model_forwards=0,CUDA_initialized=False,new_driver_ready=False,
        cache_for_existing_fixed_upstream_retention_only=True,files_SHA={str(p):sha(p) for p in [ROOT/'scripts/saved_commit_reference_v76_v2.py',Path(__file__)]})
    (D.parent/'root_legacy_commit_reference_contract.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful():sys.exit(1)
