"""Focused storage/recovery checks. Tiny arithmetic only, no actual model."""
import copy,json,random,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import text_generation_support_v1 as support
from research import linked_runtime_v75 as linked

class StorageContracts(unittest.TestCase):
    def setUp(self):
        self.fixture_root=(ROOT/'research/artifacts/linked_runtime_v75_test_temp').resolve();self.fixture_root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='v75_linked_',dir=self.fixture_root);self.root=Path(self.temp.name).resolve()
        random.seed(75);np.random.seed(75);torch.manual_seed(75)
        self.names=[f'active.{i}' for i in range(7)]
        self.parent={**{k:torch.randn(2,3) for k in self.names},'frozen.anchor':torch.randn(4,5)}
        self.parentfile=self.root/'parent.pt';torch.save(dict(model=self.parent,old_optimizer=dict(preserved=torch.arange(4))),self.parentfile)
        self.ref=dict(path=str(self.parentfile),SHA=linked.file_sha(self.parentfile),model_key='model',model_SHA=support.state_sha(self.parent))
        self.params={k:torch.nn.Parameter(v.clone()) for k,v in self.parent.items() if k in self.names}
        self.opt=torch.optim.AdamW(list(self.params.values()),lr=3e-5);self.scaler=torch.amp.GradScaler('cpu',init_scale=1024.)
        for _ in range(3):self.update(self.params,self.opt,self.scaler)
        self.full=dict(schema='tiny_enabled_scaler_runtime',model={**self.parent,**{k:p.detach().clone() for k,p in self.params.items()}},
            optimizer=copy.deepcopy(self.opt.state_dict()),scaler=self.scaler.state_dict(),optimizer_names=[self.names],
            torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_keys=np.random.get_state()[1].tolist(),
            phase='head',cursor=3,attempts=3,orders=[[1],[0],[3],[2]],history=[dict(attempt=i) for i in (1,2,3)],
            exposure=dict(rows=3,bytes=81),calls=dict(readout=3),cache_refs=[],inflight=None)
        self.env=linked.pack(self.full,self.parent,self.ref,self.names,support.state_sha)
    def tearDown(self):
        if not self.root.is_relative_to(self.fixture_root) or self.root==self.fixture_root or not self.root.name.startswith('v75_linked_'):raise ValueError('owned temporary cleanup boundary')
        self.temp.cleanup()
    @staticmethod
    def update(params,opt,scaler):
        opt.zero_grad(set_to_none=True)
        loss=sum((p*torch.rand_like(p)).square().mean() for p in params.values())
        scaler.scale(loss).backward();scaler.unscale_(opt);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    def test_populated_Adam_enabled_scaler_rng_next_update(self):
        meta=linked.save_verified(self.root/'linked.pt',self.env,self.root,support.state_sha)
        restored=linked.load_verified(meta['path'],meta['SHA'],self.root,support.state_sha)
        self.assertEqual(support.state_sha(restored),support.state_sha(self.full))
        other={k:torch.nn.Parameter(restored['model'][k].clone()) for k in self.names}
        opt=torch.optim.AdamW(list(other.values()),lr=3e-5);opt.load_state_dict(restored['optimizer'])
        scaler=torch.amp.GradScaler('cpu');scaler.load_state_dict(restored['scaler'])
        torch.set_rng_state(self.full['torch_rng']);self.update(self.params,self.opt,self.scaler)
        torch.set_rng_state(restored['torch_rng']);self.update(other,opt,scaler)
        self.assertEqual(support.state_sha({k:p.detach() for k,p in self.params.items()}),support.state_sha({k:p.detach() for k,p in other.items()}))
        self.assertEqual(support.state_sha(self.opt.state_dict()),support.state_sha(opt.state_dict()))
        self.assertEqual(self.scaler.state_dict(),scaler.state_dict())
        self.assertEqual(self.ref['SHA'],linked.file_sha(self.parentfile))
        with self.assertRaises(FileExistsError):linked.save_verified(self.root/'linked.pt',self.env,self.root,support.state_sha)
    def test_frozen_change_and_active_dtype_rejected(self):
        wrong=copy.deepcopy(self.full);wrong['model']['frozen.anchor'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'nonactive'):linked.pack(wrong,self.parent,self.ref,self.names,support.state_sha)
        wrong=copy.deepcopy(self.env);wrong['active_weights'][self.names[0]]=wrong['active_weights'][self.names[0]].double()
        with self.assertRaisesRegex(ValueError,'dtype'):linked.unpack(wrong,self.root,support.state_sha)
    def test_parent_corruption_missing_and_boundary_rejected(self):
        wrong=copy.deepcopy(self.env);wrong['parent_reference']['SHA']='0'*64
        with self.assertRaisesRegex(ValueError,'parent file SHA'):linked.unpack(wrong,self.root,support.state_sha)
        wrong=copy.deepcopy(self.env);wrong['parent_reference']['path']=str(self.root/'absent.pt')
        with self.assertRaises(FileNotFoundError):linked.unpack(wrong,self.root,support.state_sha)
        wrong=copy.deepcopy(self.env);wrong['parent_reference']['path']=str(self.root.parent/'outside.pt')
        with self.assertRaisesRegex(ValueError,'outside'):linked.unpack(wrong,self.root,support.state_sha)
    def test_runtime_corruption_and_unknown_active_rejected(self):
        wrong=copy.deepcopy(self.env);wrong['runtime']['cursor']+=1
        with self.assertRaisesRegex(ValueError,'fullstate digest'):linked.unpack(wrong,self.root,support.state_sha)
        wrong=copy.deepcopy(self.env);wrong['active_names'].append('unknown');wrong['active_weights']['unknown']=torch.zeros(1)
        with self.assertRaisesRegex(ValueError,'active model identity'):linked.unpack(wrong,self.root,support.state_sha)
    def test_interrupted_attempt_preserved_not_replayed(self):
        state=copy.deepcopy(self.full);state['attempts']=4;state['cursor']=4;state['inflight']=dict(stage='after_consumption',rows=[2]);state['calls']['readout']=4
        env=linked.pack(state,self.parent,self.ref,self.names,support.state_sha);actual=linked.unpack(env,self.root,support.state_sha)
        self.assertEqual(support.state_sha(actual),support.state_sha(state));self.assertEqual(len(actual['history']),3);self.assertIsNotNone(actual['inflight'])

if __name__=='__main__':
    torch.set_num_threads(2);suite=unittest.defaultTestLoader.loadTestsFromTestCase(StorageContracts);result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert not torch.cuda.is_initialized()
    if not result.wasSuccessful():sys.exit(1)
