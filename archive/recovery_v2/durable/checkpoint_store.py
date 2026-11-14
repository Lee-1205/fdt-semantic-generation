"""Two-slot remote checkpoint commit, independent of the model source contract."""
import hashlib
import json
import os
import pathlib
import re
import time


class PersistenceError(RuntimeError):
    pass


class ContractError(PersistenceError):
    pass


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            md5.update(block)
    return {'sha256': sha.hexdigest(), 'md5Checksum': md5.hexdigest(),
            'size': pathlib.Path(path).stat().st_size}


def json_file(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)
    return path


class GoogleDriveBackend:
    """Drive API acknowledgment and checksum, rather than a mounted-disk guess."""
    def __init__(self, service, folder_id):
        self.service, self.folder_id = service, folder_id

    def files(self):
        out, token = [], None
        while True:
            reply = self.service.files().list(
                q=f"'{self.folder_id}' in parents and trashed=false",
                fields='nextPageToken,files(id,name,size,md5Checksum,version,modifiedTime)',
                pageSize=100, pageToken=token).execute(num_retries=3)
            out.extend(reply.get('files', []))
            token = reply.get('nextPageToken')
            if not token:
                return out

    def named(self, name):
        found = [item for item in self.files() if item['name'] == name]
        if len(found) > 1:
            raise PersistenceError('복수 파일을 자동 덮어쓰지 않습니다: ' + name)
        return found[0] if found else None

    def put(self, path, name):
        from googleapiclient.http import MediaFileUpload
        path = pathlib.Path(path)
        existing = self.named(name)
        quota = self.service.about().get(fields='storageQuota').execute(num_retries=3)['storageQuota']
        if quota.get('limit'):
            spare = int(quota['limit']) - int(quota.get('usage', 0))
            if spare < path.stat().st_size + 128 * 1024 * 1024:
                raise PersistenceError('Drive 여유 공간 부족. 기존 파일 삭제/유료 업그레이드 없이 중단합니다.')
        media = MediaFileUpload(str(path), mimetype='application/octet-stream',
                                resumable=True, chunksize=8 * 1024 * 1024)
        fields = 'id,name,size,md5Checksum,version,modifiedTime'
        if existing:
            request = self.service.files().update(fileId=existing['id'], media_body=media, fields=fields)
        else:
            request = self.service.files().create(
                body={'name': name, 'parents': [self.folder_id]}, media_body=media, fields=fields)
        result = None
        while result is None:
            _, result = request.next_chunk(num_retries=3)
        return self.stat(result['id'])

    def stat(self, file_id):
        return self.service.files().get(fileId=file_id,
            fields='id,name,size,md5Checksum,version,modifiedTime').execute(num_retries=3)

    def get(self, file_id, path):
        from googleapiclient.http import MediaIoBaseDownload
        path = pathlib.Path(path)
        temporary = path.with_name(path.name + '.download')
        with open(temporary, 'wb') as stream:
            download = MediaIoBaseDownload(stream, self.service.files().get_media(fileId=file_id),
                                            chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = download.next_chunk(num_retries=3)
        os.replace(temporary, path)
        return path


class CheckpointStore:
    def __init__(self, backend, local, wrapper_sha256):
        self.backend = backend
        self.local = pathlib.Path(local)
        self.local.mkdir(parents=True, exist_ok=True)
        self.wrapper_sha256 = wrapper_sha256
        self.committed = None

    def put_verified(self, path, name):
        expected = hashes(path)
        actual = self.backend.put(path, name)
        if int(actual.get('size', -1)) != expected['size'] or actual.get('md5Checksum') != expected['md5Checksum']:
            raise PersistenceError('Drive 크기/체크섬 검사 실패: ' + name)
        return actual, expected

    def named(self, name):
        found = [item for item in self.backend.files() if item['name'] == name]
        if len(found) > 1:
            raise PersistenceError('복수 파일을 자동 선택하지 않습니다: ' + name)
        return found[0] if found else None

    @staticmethod
    def valid_marker(marker):
        if not isinstance(marker, dict):
            return False
        for field in ('slot', 'commit_sequence', 'size', 'optimizer_updates', 'optimizer_attempts'):
            if type(marker.get(field)) is not int:
                return False
        return (marker.get('schema') == 1 and marker.get('complete') is True and
                marker['slot'] in (0, 1) and marker['commit_sequence'] > 0 and
                marker['size'] > 0 and 0 <= marker['optimizer_updates'] <= marker['optimizer_attempts'] and
                marker.get('file') == f"checkpoint_slot_{marker['slot']}.pt" and
                isinstance(marker.get('contract'), dict) and
                isinstance(marker.get('drive_file_id'), str) and
                isinstance(marker.get('drive_version'), str) and
                bool(re.fullmatch(r'[0-9a-f]{64}', str(marker.get('sha256', '')))) and
                bool(re.fullmatch(r'[0-9a-f]{32}', str(marker.get('md5Checksum', '')))))

    @staticmethod
    def metadata_matches(remote, marker):
        return (int(remote.get('size', -1)) == marker['size'] and
                remote.get('md5Checksum') == marker['md5Checksum'] and
                str(remote.get('version')) == marker['drive_version'] and
                remote.get('name') == marker['file'])

    def artifact_marker(self, name, scope):
        item = self.named(name + '.complete.json')
        if not item:
            return None
        path = self.local / ('remote_' + name + '.complete.json')
        self.backend.get(item['id'], path)
        try:
            marker = json.loads(path.read_text())
            if (not isinstance(marker, dict) or marker.get('complete') is not True or
                    marker.get('scope') != scope or marker.get('file') != name):
                return None
            remote = self.backend.stat(marker['drive_file_id'])
            return marker if self.metadata_matches(remote, marker) else None
        except (ValueError, KeyError, TypeError):
            return None

    def publish_artifact(self, path, name, scope):
        remote, digest = self.put_verified(path, name)
        marker = {'schema': 1, 'complete': True, 'file': name, 'scope': scope,
                  'drive_file_id': remote['id'], 'drive_version': str(remote['version']),
                  **digest, 'modifiedTime': remote['modifiedTime']}
        self.put_verified(json_file(self.local / (name + '.complete.json'), marker),
                          name + '.complete.json')
        return marker

    def load_artifact(self, name, scope):
        marker = self.artifact_marker(name, scope)
        if not marker:
            raise PersistenceError('검증된 결과 파일이 없습니다: ' + name)
        path = self.local / name
        self.backend.get(marker['drive_file_id'], path)
        if hashes(path)['sha256'] != marker['sha256']:
            raise PersistenceError('결과 파일 SHA256 불일치: ' + name)
        return json.loads(path.read_text())

    def commit(self, payload_path, contract, state):
        previous = self.committed
        if previous:
            current = self.backend.stat(previous['drive_file_id'])
            if (str(current.get('version')) != previous['drive_version'] or
                current.get('md5Checksum') != previous['md5Checksum']):
                raise PersistenceError('현재 검증본이 외부에서 바뀌었습니다. 동시 실행을 중단합니다.')
        slot = 0 if previous is None else 1 - previous['slot']
        sequence = 1 if previous is None else previous['commit_sequence'] + 1
        name = f'checkpoint_slot_{slot}.pt'
        remote, digest = self.put_verified(payload_path, name)
        marker = {
            'schema': 1, 'complete': True, 'slot': slot, 'commit_sequence': sequence,
            'file': name, 'drive_file_id': remote['id'], 'drive_version': str(remote['version']),
            **digest, 'modifiedTime': remote['modifiedTime'],
            'contract': json.loads(json.dumps(contract)),
            'durability_wrapper_sha256': self.wrapper_sha256,
            'optimizer_updates': state['optimizer_updates'],
            'optimizer_attempts': state['optimizer_attempts'],
            'phase': state['phase'], 'saved_unix_time': time.time(),
            'verified_by': 'Drive files.get size + md5Checksum; SHA256 verified on reload',
        }
        marker_path = json_file(self.local / f'checkpoint_slot_{slot}.json', marker)
        self.put_verified(marker_path, marker_path.name)
        # The marker is written only after the large object has been acknowledged.
        # A failed pointer/status update can still be recovered from this marker.
        self.committed = marker
        pointer = json_file(self.local / 'latest_verified.json', marker)
        self.put_verified(pointer, pointer.name)
        status = {**state, 'weights_saved_to_drive': True, 'drive_backup': marker,
                  'previous_2240_update_weights_recovered': False,
                  'run_kind': 'fresh_reconstruction_after_missing_weights',
                  'not_previous_experiment_results': True}
        self.put_verified(json_file(self.local / 'training_status.json', status), 'training_status.json')
        print('DRIVE_COMMIT_VERIFIED=' + json.dumps({
            'updates': marker['optimizer_updates'], 'attempts': marker['optimizer_attempts'],
            'phase': marker['phase'], 'size': marker['size'], 'sha256': marker['sha256'],
            'drive_file_id': marker['drive_file_id'], 'commit_sequence': sequence}), flush=True)
        return marker

    def restore(self, expected_source):
        files = self.backend.files()
        markers = [item for item in files if item['name'] in
                   ('checkpoint_slot_0.json', 'checkpoint_slot_1.json')]
        if len({item['name'] for item in markers}) != len(markers):
            raise PersistenceError('복수 완료 marker를 자동 선택하지 않습니다.')
        candidates = []
        for item in markers:
            try:
                path = self.local / ('remote_' + item['name'])
                self.backend.get(item['id'], path)
                candidate = json.loads(path.read_text())
                if self.valid_marker(candidate):
                    candidates.append(candidate)
            except (OSError, ValueError, PersistenceError):
                continue
        candidates.sort(key=lambda x: x['commit_sequence'], reverse=True)
        for marker in candidates:
            if marker.get('contract', {}).get('source') != expected_source:
                raise ContractError('Source fingerprint 변경. 기존 실행을 재개하지 않습니다.')
            if marker.get('durability_wrapper_sha256') != self.wrapper_sha256:
                raise ContractError('저장 계층 fingerprint 변경. 기존 실행을 재개하지 않습니다.')
            name = marker['file']
            if name not in ('checkpoint_slot_0.pt', 'checkpoint_slot_1.pt'):
                raise ContractError('안전하지 않은 체크포인트 파일명')
            try:
                remote = self.backend.stat(marker['drive_file_id'])
                if not self.metadata_matches(remote, marker):
                    continue
                path = self.local / name
                self.backend.get(remote['id'], path)
                if hashes(path)['sha256'] != marker['sha256']:
                    continue
                self.committed = marker
                json_file(path.with_suffix('.json'), {
                    'file': name, 'sha256': marker['sha256'],
                    'updates': marker['optimizer_updates'], 'contract': marker['contract']})
                json_file(self.local / 'latest.json', json.loads(path.with_suffix('.json').read_text()))
                print('RESTORE_SHA256_VERIFIED=' + json.dumps({
                    'updates': marker['optimizer_updates'], 'sha256': marker['sha256']}), flush=True)
                return marker
            except (OSError, PersistenceError):
                continue
        if markers or any(item['name'].startswith('checkpoint_') for item in files):
            raise PersistenceError('검증 가능한 복구본 없음. 기존 실행을 새 가중치로 덮어쓰지 않습니다.')
        return None
