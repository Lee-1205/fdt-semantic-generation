"""Failure-injection tests for persistence; no CPU/GPU training is performed."""
import hashlib
import json
import pathlib
import tempfile
import unittest
from checkpoint_store import CheckpointStore, ContractError, PersistenceError


class MemoryDrive:
    def __init__(self):
        self.items = {}
        self.fail_name = None
        self.corrupt_upload = False

    def files(self):
        return [dict(v['meta']) for v in self.items.values()]

    def put(self, path, name):
        if name == self.fail_name:
            raise PersistenceError('injected upload interruption')
        data = pathlib.Path(path).read_bytes()
        if self.corrupt_upload and name.endswith('.pt'):
            data += b'corrupt'
        previous = self.items.get(name)
        meta = {'id': name, 'name': name, 'size': str(len(data)),
                'version': str(int(previous['meta']['version']) + 1 if previous else 1),
                'md5Checksum': hashlib.md5(data).hexdigest(), 'modifiedTime': '2026-10-06'}
        self.items[name] = {'meta': meta, 'bytes': data}
        return dict(meta)

    def stat(self, file_id):
        return dict(self.items[file_id]['meta'])

    def get(self, file_id, path):
        pathlib.Path(path).write_bytes(self.items[file_id]['bytes'])
        return pathlib.Path(path)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.local = pathlib.Path(self.temp.name)
        self.remote = MemoryDrive()
        self.store = CheckpointStore(self.remote, self.local, 'wrapper')
        self.contract = {'source': 'source', 'config': {'fixed': 1}, 'plan': {'steps': 8}}

    def tearDown(self):
        self.temp.cleanup()

    def commit(self, update):
        path = self.local / 'payload.pt'
        path.write_bytes(f'weights-and-optimizer-step-{update}'.encode())
        return self.store.commit(path, self.contract,
            {'optimizer_updates': update, 'optimizer_attempts': update, 'phase': 'semantic'})

    def restored(self, wrapper='wrapper', source='source'):
        return CheckpointStore(self.remote, self.local / 'new_runtime', wrapper).restore(source)

    def test_clean_runtime_restores_latest(self):
        self.commit(1)
        self.commit(2)
        self.assertEqual(self.restored()['optimizer_updates'], 2)

    def test_interrupted_large_upload_preserves_previous(self):
        self.commit(1)
        self.remote.fail_name = 'checkpoint_slot_1.pt'
        with self.assertRaises(PersistenceError):
            self.commit(2)
        self.assertEqual(self.restored()['optimizer_updates'], 1)

    def test_interrupted_marker_preserves_previous(self):
        self.commit(1)
        self.remote.fail_name = 'checkpoint_slot_1.json'
        with self.assertRaises(PersistenceError):
            self.commit(2)
        self.assertEqual(self.restored()['optimizer_updates'], 1)

    def test_pointer_failure_recovers_committed_marker(self):
        self.commit(1)
        self.remote.fail_name = 'latest_verified.json'
        with self.assertRaises(PersistenceError):
            self.commit(2)
        self.assertEqual(self.restored()['optimizer_updates'], 2)

    def test_corrupt_upload_does_not_commit(self):
        self.commit(1)
        self.remote.corrupt_upload = True
        with self.assertRaises(PersistenceError):
            self.commit(2)
        self.assertEqual(self.restored()['optimizer_updates'], 1)

    def test_latest_corruption_falls_back(self):
        self.commit(1)
        self.commit(2)
        self.remote.items['checkpoint_slot_1.pt']['bytes'] += b'after-save-corruption'
        self.assertEqual(self.restored()['optimizer_updates'], 1)

    def test_changed_source_or_wrapper_blocks_resume(self):
        self.commit(1)
        with self.assertRaises(ContractError):
            self.restored(source='changed-source')
        with self.assertRaises(ContractError):
            self.restored(wrapper='changed-wrapper')

    def test_unverified_existing_weights_do_not_start_over(self):
        path = self.local / 'uncommitted.pt'
        path.write_bytes(b'uncommitted')
        self.remote.put(path, 'checkpoint_slot_0.pt')
        with self.assertRaises(PersistenceError):
            self.restored()

    def test_modified_active_slot_blocks_next_commit(self):
        self.commit(1)
        self.remote.items['checkpoint_slot_0.pt']['meta']['version'] = '99'
        with self.assertRaises(PersistenceError):
            self.commit(2)
        self.assertNotIn('checkpoint_slot_1.pt', self.remote.items)

    def test_malformed_latest_marker_falls_back(self):
        self.commit(1)
        self.commit(2)
        self.remote.items['checkpoint_slot_1.json']['bytes'] = b'{"complete": true}'
        self.assertEqual(self.restored()['optimizer_updates'], 1)

    def test_invalid_slot_file_pair_is_rejected(self):
        self.commit(1)
        second = self.commit(2)
        second['file'] = 'checkpoint_slot_0.pt'
        self.remote.items['checkpoint_slot_1.json']['bytes'] = json.dumps(second).encode()
        self.assertEqual(self.restored()['optimizer_updates'], 1)

    def test_duplicate_markers_are_not_selected(self):
        self.commit(1)
        item = self.remote.items['checkpoint_slot_0.json']
        self.remote.items['duplicate-marker-id'] = {'meta': dict(item['meta']), 'bytes': item['bytes']}
        with self.assertRaises(PersistenceError):
            self.restored()

    def test_artifact_requires_matching_checkpoint_scope(self):
        path = self.local / 'evaluation.json'
        path.write_text('{"accuracy": 0.5}')
        scope = {'checkpoint_sha256': 'fixed-checkpoint'}
        self.store.publish_artifact(path, path.name, scope)
        self.assertIsNone(self.store.artifact_marker(path.name, {'checkpoint_sha256': 'different'}))
        self.assertEqual(self.store.load_artifact(path.name, scope), {'accuracy': 0.5})

    def test_artifact_interrupted_marker_is_incomplete(self):
        path = self.local / 'evaluation.json'
        path.write_text('{}')
        self.remote.fail_name = 'evaluation.json.complete.json'
        with self.assertRaises(PersistenceError):
            self.store.publish_artifact(path, path.name, {'run': 1})
        self.assertIsNone(self.store.artifact_marker(path.name, {'run': 1}))

    def test_artifact_download_sha256_detects_corruption(self):
        path = self.local / 'evaluation.json'
        path.write_text('{"accuracy": 0.5}')
        scope = {'run': 1}
        self.store.publish_artifact(path, path.name, scope)
        self.remote.items[path.name]['bytes'] += b' '
        with self.assertRaises(PersistenceError):
            self.store.load_artifact(path.name, scope)

    def test_download_transport_error_blocks_restore(self):
        self.commit(1)
        def disconnected(file_id, path):
            raise RuntimeError('Drive transport unavailable')
        self.remote.get = disconnected
        with self.assertRaises(RuntimeError):
            self.restored()


if __name__ == '__main__':
    unittest.main()
