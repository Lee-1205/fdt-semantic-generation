"""Exercise actual wrapper methods with fake evaluation/storage, never PyTorch."""
import ast
import gc
import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from checkpoint_store import CheckpointStore, PersistenceError, json_file
from run_control import Budget
from test_checkpoint_store import MemoryDrive
from test_run_control import Clock


class FakeParent:
    def export(self):
        # Test fixture bytes, not model weights.
        (self.out / 'runtime.pt').write_bytes(b'fixture-runtime-weights')


def load_wrapper_methods():
    path = pathlib.Path(__file__).with_name('durable_runner.py')
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DurableTrainer']
    namespace = {'Trainer': FakeParent, 'json': json, 'json_file': json_file,
                 'gc': gc, 'os': os, 'pathlib': pathlib, 'PersistenceError': PersistenceError,
                 'wrapper_digest': lambda: 'fixture-wrapper'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['DurableTrainer'], namespace


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.local = pathlib.Path(self.temporary.name)
        self.clock = Clock()
        cls, self.namespace = load_wrapper_methods()
        self.trainer = cls.__new__(cls)
        self.trainer.out = self.local
        self.trainer.store = CheckpointStore(MemoryDrive(), self.local, 'fixture-wrapper')
        self.trainer.contract = {'source': 'fixture-source'}
        payload = self.local / 'checkpoint-fixture.pt'
        payload.write_bytes(b'fixture-full-checkpoint')
        self.trainer.store.commit(payload, self.trainer.contract,
                                  {'optimizer_updates': 3, 'optimizer_attempts': 3, 'phase': 'complete'})
        self.trainer.phase = 3
        self.trainer.updates = 3
        self.trainer.device = 'fixture-no-device'
        self.trainer.dev = ['dev']
        self.trainer.plan = SimpleNamespace(micro_batch=4, test_graphs=128)
        self.trainer.model = SimpleNamespace(counts=lambda: {'parameters': 102237197})
        self.evaluations = []

        def evaluate(model, items, device, micro_batch, seeds):
            self.evaluations.append(items)
            self.clock.now += 2
            return {'fixture_evaluation': len(self.evaluations), 'no_training': True}

        self.namespace['evaluate'] = evaluate
        self.namespace['make_split'] = lambda split, count, **kw: [split, count, kw]

    def tearDown(self):
        self.temporary.cleanup()

    def test_no_eval_or_export_after_budget_expiry(self):
        self.assertFalse(self.trainer.finish(Budget(0, self.clock)))
        self.assertEqual(self.evaluations, [])
        self.assertNotIn('runtime.pt', self.trainer.store.backend.items)

    def test_partial_final_evaluation_resumes_missing_stages(self):
        self.assertFalse(self.trainer.finish(Budget(3, self.clock)))
        self.assertEqual(len(self.evaluations), 2)
        checkpoint_sha = self.trainer.store.committed['sha256']
        self.assertTrue(self.trainer.finish(Budget(100, self.clock)))
        self.assertEqual(len(self.evaluations), 4)
        self.assertEqual(self.trainer.store.committed['sha256'], checkpoint_sha)
        self.assertTrue(self.trainer.finish(Budget(100, self.clock)))
        self.assertEqual(len(self.evaluations), 4)
        self.assertIn('evaluation_complete.json.complete.json', self.trainer.store.backend.items)

    def test_runtime_metadata_uses_published_binary_hash(self):
        self.assertTrue(self.trainer.finish(Budget(100, self.clock)))
        scope = {'checkpoint_sha256': self.trainer.store.committed['sha256'],
                 'source': 'fixture-source', 'wrapper': 'fixture-wrapper', 'updates': 3}
        marker = self.trainer.store.artifact_marker('runtime.pt', scope)
        metadata = self.trainer.store.load_artifact('runtime.json', scope)
        self.assertEqual(metadata['sha256'], marker['sha256'])

    def test_uncommitted_training_cannot_publish_completion(self):
        self.trainer.phase = 2
        with self.assertRaises(PersistenceError):
            self.trainer.finish(Budget(100, self.clock))


if __name__ == '__main__':
    unittest.main()
