"""Build the recovery notebook, embedding the unchanged research source."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile


class Notebook:
    @staticmethod
    def new_markdown_cell(source):
        return {'cell_type': 'markdown', 'metadata': {}, 'source': source,
                'id': hashlib.sha256(source.encode()).hexdigest()[:12]}

    @staticmethod
    def new_code_cell(source):
        return {'cell_type': 'code', 'metadata': {}, 'source': source, 'outputs': [],
                'execution_count': None, 'id': hashlib.sha256(source.encode()).hexdigest()[:12]}

HERE = Path(__file__).parent
ROOT = HERE.parent
ORIGINAL = ROOT / 'fdt100_v032'
names = ('model.py', 'data.py', 'train.py', 'runtime.py', 'app.py', 'role_experiment.py')
sources = {name: (ORIGINAL / name).read_text() for name in names}
sources.update({name: (HERE / name).read_text()
                for name in ('checkpoint_store.py', 'durable_runner.py', 'run_control.py')})
source_sha = hashlib.sha256()
for name in ('model.py', 'data.py', 'train.py', 'runtime.py'):
    source_sha.update(name.encode())
    source_sha.update((ORIGINAL / name).read_bytes())
assert source_sha.hexdigest() == 'de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79'

setup = '''import os,pathlib,sys,torch
assert torch.cuda.is_available(), 'Colab 런타임을 T4 GPU로 설정하세요. CPU 학습은 실행하지 않습니다.'
assert 'T4' in torch.cuda.get_device_name(), 'T4 GPU만 승인되었습니다: '+torch.cuda.get_device_name()
PROJECT=pathlib.Path('/content/fdt100_durable_v2');PROJECT.mkdir(exist_ok=True)
SOURCES=''' + repr(sources) + '''
for name,body in SOURCES.items():
    target=PROJECT/name
    if target.exists() and target.read_text()!=body:
        raise RuntimeError('기존 소스와 다릅니다. 저장된 계약을 바꿔 재개하지 않습니다: '+name)
    target.write_text(body)
for module_name in ('model','data','train','runtime','checkpoint_store','durable_runner','run_control'):
    module=sys.modules.get(module_name)
    if module is not None and pathlib.Path(getattr(module,'__file__','') or '').parent!=PROJECT:
        raise RuntimeError('다른 버전의 모듈이 메모리에 있습니다. 런타임을 다시 시작한 뒤 v2만 실행하세요: '+module_name)
sys.path.insert(0,str(PROJECT))
os.environ['OMP_NUM_THREADS']='2'
torch.set_num_threads(2)
print({'GPU':torch.cuda.get_device_name(),'torch':str(torch.__version__),'source':'de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79'})
'''
auth = '''from google.colab import auth
auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
credentials,_=google.auth.default()
drive_service=build('drive','v3',credentials=credentials,cache_discovery=False)
root=drive_service.files().get(fileId='1cH9Ha58Uc3HTHhp9BV9yJpvgE84-ABil',fields='id,name,mimeType').execute()
assert root['mimeType']=='application/vnd.google-apps.folder'
print('연구 폴더 확인:',root['name'])
'''
launch = '''from durable_runner import start
# 이번 실행의 시간 예산입니다. 학습 계획/fingerprint를 변경하지 않습니다.
MAX_RUN_HOURS=6.0
assert 0<MAX_RUN_HOURS<=10, '권장 6시간, 설정 범위는 0 초과 10시간 이하입니다.'
# 저장된 검증본이 있으면 동일 계약으로 복구합니다.
# 없으면 이전 가중치 소실을 명시한 새 1440-update 실행으로 시작합니다.
# 매 update 저장은 느릴 수 있습니다. 저장 실패 시 다음 update를 시작하지 않습니다.
status=start(drive_service,wall_seconds=MAX_RUN_HOURS*3600)
print('현재 학습 단계:',status['phase'],'/ 실제 성공 updates:',status['optimizer_updates'])
print('중단/완료 사유:',status['stop_reason'])
'''
inspect = '''from durable_runner import run_folder,wrapper_digest,EXPECTED_SOURCE
from checkpoint_store import CheckpointStore,GoogleDriveBackend
store=CheckpointStore(GoogleDriveBackend(drive_service,run_folder(drive_service)),
                      '/content/fdt100_v2_reload_audit',wrapper_digest())
marker=store.restore(EXPECTED_SOURCE)
assert marker is not None, '완료 marker가 있는 검증본이 없습니다.'
import json
from model import FDT100,Config
payload=torch.load(store.local/marker['file'],map_location='cpu',weights_only=True)
assert json.loads(json.dumps(payload['contract']))==marker['contract']
assert payload['updates']==marker['optimizer_updates']
loaded=FDT100(Config(**marker['contract']['config']))
loaded.load_state_dict(payload['model'],strict=True)
assert loaded.counts()['parameters']==102237197
assert all(bool(torch.isfinite(t).all()) for t in loaded.parameters())
print({'full_download_sha256_verified':True,'strict_model_reload':True,
       'parameters':loaded.counts()['parameters'],'optimizer_updates':payload['updates'],
       'optimizer_state_present':bool(payload['optimizer']),
       'rng_and_amp_state_present':all(k in payload for k in ['scaler','torch_rng','cuda_rng','sampler_rng','noise_rng'])})
del payload,loaded
'''
cells = [Notebook.new_markdown_cell('''# FDT 102M · T4 · 6시간 예산 · 매 업데이트 Drive 저장/복구 v2

이전 2,240회 학습 가중치는 Drive에 저장되지 않았고 이전 런타임에서 사라졌습니다.
이 노트북은 이전 가중치 복구가 아닙니다. 새 복구 실행 `FDT100_T4_recovery_20261006_v2`을 생성합니다.
원본 모델 source `de05e0…`와 모델/데이터 코드를 바꾸지 않고 저장 계층을 외부에서 추가했습니다.
복구 실행의 검증본이 생긴 이후에는 source/config/data/plan/PyTorch/GPU가 일치할 때만 이어 갑니다.

**실행:** Colab에서 T4를 선택한 뒤 1→2→3 셀을 실행합니다. 다음 접속에도 같은 순서로 실행하면 검증본을 불러옵니다.
집에서 실행할 때 전원을 연결하고 노트북의 자동 절전/최대 절전 진입을 끄며 인터넷을 유지하세요.
3번 셀의 `MAX_RUN_HOURS=6.0`은 코드의 시간 예산입니다. 중간 저장에도 시간 계산이 초기화되지 않습니다.
시간이 되면 다음 update를 시작하지 않고 검증본을 보존한 뒤 반환합니다. 진행 중인 저장/평가는 예산을 넘길 수 있습니다.
1,440회 계획이 먼저 끝나거나 검증 gate 실패/Drive 오류가 생기면 6시간 이전에도 중단합니다.
전체 평가가 미완료이면 다음 접속의 3번 셀에서 학습 추가 없이 남은 평가만 진행합니다.
4번 셀은 선택 검사이며 3번 셀이 정상 반환한 뒤에만 실행하세요. 같은 연구를 다른 런타임에서 동시에 실행하지 마세요.

**저장:** 최초 초기값과 매 optimizer 시도 뒤 전체 가중치, optimizer, AMP, 진행 단계, 실제 노출 횟수, 난수 상태를 저장합니다.
Drive API가 반환한 크기·MD5를 확인한 뒤 완료 marker를 게시합니다. 다시 불러올 때 파일 전체 SHA-256을 확인합니다.
큰 체크포인트 두 슬롯은 이번 복구 실행 전용이며, 직전 검증본을 남겨 둔 채 다른 슬롯을 교체합니다. 기존 실험 파일은 수정/삭제하지 않습니다.

**한계:** Colab의 런타임 종료나 GPU 할당을 보장할 수 없습니다. 저장 중 종료되면 현재 미확정 update 하나는 다시 계산해야 할 수 있습니다.
화면과 탭을 열어 두어도 Colab의 세션 제한이 사라지지는 않습니다. Google 공식 FAQ는 무료 실행의 최대를
사용량/가용성에 따라 최대 12시간으로 설명하며 더 이른 종료도 가능합니다. 자동 클릭·keepalive는 사용하지 않습니다.
매 update마다 수백 MB~GB를 전송하므로 학습이 크게 느려질 수 있습니다. Drive 용량 부족/저장 실패 시 학습을 중단합니다.
학습 계획은 새 실행의 grounding 480, semantic 320, diffusion 640회입니다. 이전 학습 실적과 합산하지 않습니다.
미실행 60/15/25% 제안과 표현 분리·부정 일반화 실험은 이 복구 학습에 적용하지 않았습니다.'''),
Notebook.new_markdown_cell('## 1. T4와 원본 소스 확인'),Notebook.new_code_cell(setup),
Notebook.new_markdown_cell('## 2. Google Drive 인증 및 폴더 확인'),Notebook.new_code_cell(auth),
Notebook.new_markdown_cell('## 3. 검증본 복구 또는 새 실행 시작 (기본 6시간 예산)'),Notebook.new_code_cell(launch),
Notebook.new_markdown_cell('## 4. 저장본 전체 다운로드·SHA-256·모델 재로드 검사'),Notebook.new_code_cell(inspect)]
for cell in cells:
    if cell['cell_type'] == 'code':
        ast.parse(cell['source'])
nb = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
    'accelerator':'GPU', 'colab':{'name':'FDT_T4_100M_Durable_Recovery_v2.ipynb','gpuType':'T4'},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}}
notebook = ROOT / 'FDT_T4_100M_Durable_Recovery_v2.ipynb'
notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + '\n')
package = ROOT / 'FDT_T4_100M_Durable_Recovery_v2.zip'
with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(notebook, notebook.name)
    for name in names:
        archive.write(ORIGINAL / name, 'original_source/' + name)
    for path in HERE.glob('*'):
        if path.is_file():
            archive.write(path, 'durable/' + path.name)
    report = ROOT / 'FDT_Code_Audit_2026_10_06_v2.md'
    if report.exists():
        archive.write(report, report.name)
print(json.dumps({'source_sha256':source_sha.hexdigest(),
                  'notebook':str(notebook),'package':str(package)}))
