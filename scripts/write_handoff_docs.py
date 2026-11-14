"""Write factual handoff from committed local smoke evidence; no model execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "runs/smoke_report.json").read_text(encoding="utf-8"))
parent = report["actual_102m"]
proof = report["resume_proof"]
latest = json.loads((ROOT / "runs/smoke_102m/complete_000002.json").read_text())
preflight = json.loads((ROOT / "docs/local_preflight.json").read_text())
baseline = yaml.safe_load((ROOT / "configs/local_baseline.yaml").read_text())
candidate = {**baseline, "variant": "masked", "research_stage": "E1; CPU padding check only; optimizer not executed"}
(ROOT / "configs/local_masked_candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
(ROOT / "configs/research_candidates.yaml").write_text(yaml.safe_dump({
    "status": "PROPOSED_NOT_EXECUTED", "E2": {"query_target_fraction": [0.25]*4,
    "grounding_target_fraction": [0.25]*4, "expression_family_split": True,
    "balance_via": "typed-graph buckets + query selection + bounded reject sampling; never relabel oracle"},
    "E4": {"phase_update_hypothesis": [0.60, 0.15, 0.25], "updates": 1000,
    "execution_authorized": False, "not_the_existing_recovery_plan": True},
}, sort_keys=False), encoding="utf-8")


def write(name, text):
    (ROOT / name).write_text(text.strip() + "\n", encoding="utf-8")


write("README_LOCAL_KR.md", f"""
# FDT 102M 로컬 실행

이 프로젝트는 UTF-8 관계 문구를 처리하는 **102,237,197 parameter adapter**다. 범용 언어모델이나 vNext 전체 구현이 아니다. 원본은 `archive/`에 보존했고, 이번 로컬 검증은 새 seed에서 시작했다. 과거 2,240회 가중치를 복구한 결과가 아니다.

검증 환경은 Windows11 build26200, Python3.10.11, PyTorch2.10.0+cu128, CUDA12.8, RTX5060Ti16GB/compute12.0이다. 실행은 전용 `.venv`를 쓰며 Colab/Drive/Google 로그인 없이 동작한다. 연결된 Google 도구로 Colab GPU를 시작·지속 저장할 수 없어 사용자가 허용한 로컬 fallback으로 검증했다. 마우스 제어는 중단했다.

## 설치와 점검

PowerShell에서 이 프로젝트 폴더를 작업 위치로 연다. 기존 `.venv`는 설치·검증됐다. 새 설치에는 `setup_local.ps1`을 사용한다. 관리자 권한이나 execution policy 변경은 필요하지 않으며 개별 명령을 터미널에서 실행할 수도 있다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local inspect
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local preflight --config configs/local_baseline.yaml
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q --basetemp .pytest_new_check
```

공식 설치 근거: [PyTorch2.10 CUDA12.8 Windows wheel](https://pytorch.org/get-started/previous-versions/), [NVIDIA RTX5060Ti capability12.0](https://developer.nvidia.com/cuda/gpus). 최신 버전을 임의로 설치하기보다 실제 검증한 조합을 고정했다. `requirements-lock.txt`는 실제 환경이다.

## 이번에 실행한 제한 검증

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local smoke --config configs/smoke.yaml --max-attempts 5 --wall-seconds 600
```

위 명령은 **이미 실행됐으며 재실행하면 차단된다**. 전역 smoke 예산5/5회가 소진됐다. GPU window {report['total_gpu_window_seconds']:.3f}초, 모든 시도 성공/AMPskip0. 실제102M grounding2고유update + fresh-process 반복1, tiny semantic1/diffusion1이다. 작은 모델 두 phase를 실제102M의 semantic/diffusion 훈련 완료로 해석하지 않는다.

102M 첫 loss={parent['first']['loss']:.8f}, clip전 norm={parent['first']['gradient_norm_before_clip']:.8f}. 다음 loss={parent['second']['loss']:.8f}. 이는 다른 입력의 smoke 수치이며 성능 개선 추세가 아니다. peak allocated={parent['second']['peak_allocated_bytes']/2**30:.3f}GiB/reserved={parent['second']['peak_reserved_bytes']/2**30:.3f}GiB.

## 지금 실행해도 되는 검증된 다음 명령

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local verify-checkpoint --run-dir runs/smoke_102m --fresh-process
```

전체 payload를 별도 프로세스에서 다시 읽고 모델 구조·완료 marker·hash를 확인한다. 실제 다음 optimizer update 복구는 `runs/smoke_102m/resume_proof.json`에서 이미 입력·loss·모델·Adam·AMP·RNG까지 exact-match로 확인했다.

## 장기 학습 경계

장기 학습은 아직 시작하지 않았다. 사용자의 **“학습 시작”** 지시 뒤 새 run ID와 명시적 예산으로 실행해야 한다. 아래는 구현된 CLI 인터페이스 예시이며 전체1440회 실행 결과가 아니다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local train --config configs/local_baseline.yaml --run-dir runs/baseline_authorized_001 --max-attempts 1440 --wall-seconds 21600
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local resume --run-dir runs/baseline_authorized_001 --max-attempts 100 --wall-seconds 1800
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local status --run-dir runs/baseline_authorized_001
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local stop --run-dir runs/baseline_authorized_001
```

`stop`은 현재 kernel/update와 durable 저장이 끝난 경계에서 중단한다. SIGINT/SIGTERM도 cooperative stop을 요청한다. `STOP` 파일이 있으면 재개도 멈추므로 재개 지시 후 해당 run의 STOP 파일만 명시적으로 해제해야 한다. 전원 강제 종료·OS/장치 실패 자체를 막지는 못한다. 로컬 계산은 네트워크 연결 없이 가능하지만 PC 전원이 켜져 있어야 한다.

평가는 검증된 complete marker와 고정 suite를 받는다. 이 명령은 인터페이스/fixture만 검증했고 아직 capability 평가를 실행하지 않았다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m fdt_local evaluate --checkpoint runs/baseline_authorized_001/complete_000001.json --suite configs/development_eval.yaml
```

매 optimizer 시도 뒤 전체 model/Adam/scaler/RNG/sampler/phase/LR 진행/노출/history를 저장한다. SHA/크기/내부계약을 확인한 뒤 complete marker를 게시한다. 다음 시도는 이 검증 이후다. 저장 실패는 다음 시도를 막고 이전 정상본을 보존한다. 최근3개 untagged와 명시적 initial/parent/final tag만 유지하며 다른 run이나 사용자 자료는 회전 대상이 아니다. C: 여유20GiB와 새 임시 payload 예상 크기를 매 시도/저장 전에 검사한다. 다른 앱의 디스크 소비까지 통제한다는 보장은 아니다.

`local_checkpoint_verified=true`, `backup_to_drive_verified=false`, `evaluation_complete=false`는 서로 독립이다. Windows directory fsync 보장은 하지 않는다. 상세는 `docs/FDT_HANDOVER.md`, `docs/FDT_CODE_AUDIT.md`, `docs/FDT_NEXT_EXPERIMENTS.md`.
""")

write("docs/FDT_HANDOVER.md", f"""
# FDT 로컬 인계 — 2026-10-26

## 완료한 범위

Downloads의 이중 중첩 폴더를 확인하고 manifest12개 파일의 size/SHA를 모두 대조했다. ZIP SHA `57e854be4e946e966ab2165df350c81538a9f322f7d4c14329422f83883d09ba`, 원본 four-file filename+bytes digest `de05e00dee06c3ad8ff026a2e890fc0adb2ac8e51424bf10bee24f294d055e79` 일치. 경로 탈출·symlink·실행 파일을 검사한 뒤 archive에 풀었고 notebook 전체를 실행하지 않았다.

원본 model.py는 baseline `src/fdt_local/model.py`에 byte-identical하게 보존했다. 데이터·고정 closure·basis·loss는 유지하고 runtime/CLI/환경/저장을 새 로컬 분기로 만들었다. `model_masked.py`는 별도E1 후보다. 전용venv, CUDA정책, FileLock, atomicpayload/marker, source/config/data/plan/env strict검사,20GiB floor, boundedattempt/deadline, perattempt diagnostics를 구현했다. Google/Colab/fcntl/T4-only runtime 의존성이 기본 경로에 없다.

## 실제 검증

- 원본29 fixture 검사 PASS (UTF8 모드). 첫 실행의 Windows cp949 오류는 `-X utf8`로 해결했고 원본을 바꾸지 않았다.
- 새 로컬27 CPU계약 PASS. 초기 testcollection의 문법 오류와 기존pytest임시경로 권한 문제는 수정/전용basetemp로 해결했다. 이들은 학습 성능 결과가 아니다.
- 실제102M 파라미터수 {parent['parameters']:,}, GPU RTX5060Ti, Python3.10.11/torch2.10.0+cu128. CUDA tensor backward와 phase별 smoke를 수행했다.
- 전체5 optimizerattempts/성공5/skip0, GPUwindow{report['total_gpu_window_seconds']:.3f}s. 실제102M은2고유groundingupdates와같은nextupdate의freshprocess반복1회다. Tinysemantic/diffusion각1회는 별도모델이다.
- Freshprocess에서 model+populatedAdam+AMP+모든사용RNG/입력/loss의 다음trajectory SHA가 동일했다: `{proof['actual_sha256']}`.
- 최신full checkpoint921,622,281B, SHA `{latest['sha256']}`, `runs/smoke_102m/complete_000002.json`. 별도CPU프로세스 verify-checkpoint도 통과했다. 실제size/save시간/peakVRAM은 runs/smoke_report.json에 있다.
- 과거2240회 checkpoint를 찾거나 복구하지 않았다. 동봉 ZIP에는 그런 가중치가 없다. 새seed를 과거누적값에더하지 않는다. capability평가/장기학습/vNext전체 구현은 미실행이다.

## 읽은 자료와 접근 한계

인계프롬프트·SHA목록·EVIDENCE_NOTES·감사v2·이전인계·상태JSON·실험요약의 schema/핵심결과·별도corpus계획을 읽었다. vNext v1.1에서 routing/인과기억/관계/VP/Commit/Realizer/Writer/OutcomeCodec/비용·검사 구간을 읽고설계와코드를대조했다. PDF30페이지의텍스트를추출해초록·계보·102M결과/ablation구간을확인했다. 전체PDF시각렌더링이나모든65수식의새생산구현은하지않았다.

대화도구로 '모델 구조 설명', 'FDT 상세분석', 'FDT 학습·구조 개선 점검' 각최근최대3turn을읽었다. 첫대화의assistant본문은일부content-reference만반환되어전문을읽었다고하지않는다. 손실함수관련대화전체/모든계정Library/Drive하위폴더전체를조회하지않았다. 동봉원본만으로가능한구현은계속했다. 문서의과거assistant제안은현재사용자승인으로승격하지않았다.

## 사용자 최신 지시와 실행 경계

자동화재개명시승인으로 fdt ACTIVE. 처음엔Colab없이로컬을요청했지만이후100MColab선호→마우스제어중단/Google런타임지속제어불가시로컬fallback을명시했다. Browserhelper가URL을검증하지못해중단했고이후마우스조작없음. Googlefiletools는GPU런타임제어API를제공하지않아로컬fallback으로검증했다. 계정로그인·ColabGPU·클라우드공유·유료구매없음. keepalive로제한을우회하지않는다.

사용자는C:항상20GB여유를요청했고더보수적인20GiB를하한으로사용한다. 최종관측free={shutil.disk_usage(ROOT).free:,}B. 충분해원본/다른프로젝트삭제없이진행했다. 이번smoke예산5회는소진됐고자동화도추가optimizer를실행하면안된다. 장기학습은사용자'학습 시작'지시후별도예산/새run으로만한다.

## 다음 검증된 명령

`.\\.venv\\Scripts\\python.exe -B -X utf8 -m fdt_local verify-checkpoint --run-dir runs/smoke_102m --fresh-process`
""")

write("docs/FDT_CODE_AUDIT.md", """
# 수식·코드·운영 감사

## 설계와 실제 adapter

| 항목 | 실제102M코드/shape·정밀도·gradient | 한계·실패 조건 |
|---|---|---|
| μ/ρ/γ, E02~04유사 | model.Router: q[B,F,V,128],anchors[64,128],FP32,stableTop8,invalidzero;query/view/sigma/beta학습 | μ는진실확률/identity아님;float64기억의vNext전체구현아님 |
| Local attention E05~08유사 | Language9층/768/12heads/FF2048/W48,byte257embedding;FP16autocast/RMSFP32 | tokenizer16K문서경로아님,문구48UTF8bytes |
| 문맥memory | encode: mu[B,F,64],values256,FP32num/mass;한episode안factsaggregate | vNext prefix-read-before-write/float64bank/CASWriter미구현 |
| 관계E26~31유사 | rel_left/right[8,64,8],softpluspositiveG/T/F구분,NULL질량재정규화하지않음 | G/T저랭크를F에일반화불가;명시적typedproof가아닌학습메시지 |
| initializer | fixedmax-productclosure와주어진role16을latent512에매핑,Core2블록 | 스스로학습한개방자연어추론전체로해석불가 |
| VP E36~40 | objectives.diffusion_loss 및 model.rollout: t[B],Z[B,16,512],αε−σZ0,역변환αZ−σv,정확t=1;refiner6블록 | 원래시간의최소1개/대략절반을t=1;oraclevelocity대수PASS≠denoiser성능 |
| groundingloss | maskedCE,분모Σvalid*classweight,grounding4종 | query4종과label의미다름;classweight≠노출25% |
| semanticloss | CErelation+.25maskedMSE+.2groundCE,분모slot수*latent폭 | rowmean/accumulation1/K는기존목적유지,미니배치분모에따라가변slot가중이생길수있음 |
| diffusionloss | MSEv+.25MSEz0+.5CEreadout,encode는no_grad | diffusion을늘려frozenphrasegrounding학습했다고할수없음 |
| gradient | phase별원본trainableprefix유지,매시도missing/frozen/nonfinite,clip전norm/scale/skip기록 | smoke에서102Mgrounding만실제,다른두phase는tiny로검사 |
| Commit/Realizer | runtime.Commit는int/stringtuple복제·scalarfinite검사,표현기는Commit만참조 | vNexttensor-backedCommit/Claim/Permit/Writer전체미구현,범용생성기아님 |

65개 vNext수식 전체는 archive 명세에 보존되어 있으며 위 표는 **현재adapter에실제대응하는부분**이다. ExactStore/인증epoch/Codec/ClaimProof/RealizerKV·copypermit등은해당명세의미구현인터페이스다. 이번준비에서이를생산구현완료로보고하지않는다.

## E0/E1 분리

baseline model.py는원본과byte-identical하며Corepaddingmask미전달경로도그대로다. E0가중치의변경없는CPUforward비교가bit-exact였다. 별도model_masked.py는queryvalid/factvalid/contextNULLmask를self/crossattention에전달하고paddedresidual을차단했다. 같은validquery에paddingfact/query수를바꾼CPUtinyencode불변성을검사했다. E1CUDA성능·실제학습은하지않았으며baseline가중치에몰래적용하지않는다.

## 저장·실행 계약

- FileLock은Windows/LinuxOS잠금이다. 다른PC의분산잠금보장은아니다. import는학습을시작하지않고CUDA없음은오류다.
- model/optimizer/scaler/모든사용RNG/phase/성공update/시도/노출/history/trainable목록/manualcosineLR상태를완결시도경계에저장한다. 성공과AMPskip을구분한다.
- 유일한임시payload→flush/fsync→replace→전체safe-load/size/SHA/내부계약확인→complete marker→latest순서다. Windowsdirectoryfsync는지원보장안함. 완전전원상실시OS/하드웨어durability를절대보장하지않는다.
- 최신payload손상/marker필드오류/계약불일치/Windowsreplace실패/중복writer/디스크하한/미완료평가재개를fixture로검사했다. 완료marker가없으면진행저장성공으로세지않는다.
- 고정deadline은초기저장/중간저장시간을포함한다. kernel/flush가반환하기전즉시중단보장은없다. 매시도저장이50분간격보다촘촘하다.
- C:20GiB하한과임시payload예상을매시도/저장전에검사한다. 다른프로세스가갑자기디스크를채우는상황까지예방하는시스템관리서비스는아니다.
- 평가결과는checkpointSHA+suiteSHA로묶고부분완료시누락항목만이어간다. 학습phase완료와평가완료는별개다. full plan1440회와마지막phase전환은fixture만검사했고실제장기실행아님.

## 알려진 개선 지점

1. E0의Corepadding민감도와고정closure의기여를학습성능과구분해야한다.
2. 기존graphhash는entityrename동형그래프를완전히묶지않는다. 표현family분리는별도E2개발이다.
3. class균형sampler는아직실제trainer에적용하지않았다. 25%목표와60/15/25비율을현재학습노출로기록하면안된다.
4. 새로운evalsuite는아직GPU실행안했다. 이번5attemptsmoke는성능수준을측정하는학습이아니다.
""")

write("docs/FDT_NEXT_EXPERIMENTS.md", """
# 다음 실험: 준비와 실행 구분

| 단계 | 준비된항목 | 사전통제/통과기준 | 실제상태 |
|---|---|---|---|
| E0로컬동치·복구 | baselineconfig/CLI/원본archive/fullcheckpoint/복구proof | originaltinyforward동치,CUDA102Mfinitegradient,5attempt/600s이내,독립nextupdate동일 | 통과. 성능향상실험아님 |
| E1mask | local_masked_candidate.yaml/별도model파일/CPUpadding검사 | 동일seed/data/order/loss/예산,valid출력padding불변,baseline과분리 | CPUtiny통과,CUDA학습미실행 |
| E2표현·부정/균형 | research_candidates.yaml 목표계약 | train/dev/finalfamily+typedgraph교차분리; query4종/ground4종각25%별도계수;boundedrejectsampler공급량검증;oraclelabel변경금지 | 계획,실제sampler미적용 |
| E3독립이득 | role/no-role,fuzzy대체,initializer/diffusion/recurrent통제표 | sameparent또는sameinitialization/optimizer/data/실현노출/compute;graphclusterCI+seed분산 | 미실행 |
| E4손실 | 현재loss보존·MSE/Huber/negative-margin후보 | 한요소씩변경,clip전norm/loss분모/scale기록,softmaxCE와margin분리 | 60/15/25총1000회도미실행가설 |
| E5vNext | source명세와현재adapter대응표 | oracleOutcomeCodec정보보존→Commit/Exact権限→refiner→전체integration순서 | 전체구현미완료 |

장기실험예산은사용자'학습 시작'지시후새run/새고정time-attemptcap으로정한다. 기존smoke5회는모두소진됐으며추가미니실험을그예산에몰래붙이지않는다. 완성LM·범용추론·현재미학습문구성능향상은아직주장하지않는다. 원본480/320/640계획과과거2240업데이트보고를새run과혼합하지않는다.
""")

write("docs/DECISIONS.md", f"""
# 결정·사실·가설 장부

| 분류 | 내용 |
|---|---|
| 사용자확정 | 원본보존,별도localvenv,기본CUDA실행,새장기학습은'학습 시작'이후,기존5attempt/600s smoke한도 |
| 사용자최신수정 | 자동화ACTIVE.100M검증Colab선호였으나mouseoff/Googleruntime제어불가면localfallback허용. C:여유20GB최소,구현은20GiB로보수화 |
| 측정 | 원본12해시PASS,originalsourcebundlehash일치,actualparams102237197,CPU27+legacy29PASS,smoke5성공0skip/{report['total_gpu_window_seconds']:.3f}s,독립resumeexact |
| 측정 | fullpayload{latest['bytes']}B/SHA{latest['sha256']},localverifiedtrue/Drivebackupfalse |
| 해석 | 로컬이식·저장복구가능성은입증. 능력개선·오래된가중치회복·vNext전체구현은입증아님 |
| 미실행 | phase480/320/640,query25%/ground25%balancedtraining,60/15/25·1000회,role/fuzzy/diffusionmatchedablation,광범위능력평가 |
| 미확인 | 다른원격폴더의과거가중치존재,열람되지않은손실함수대화전문,Colab계정runtime실제상태 |

자동화는현재경계를지키며가능한CPU개발/연구준비를이어가고조용한상태대기에는알림을반복하지않는다. 사용자가중단을요청하면중단한다. Colab세션제한회피·유료자원·외부공유는허용하지않는다.
""")

write("setup_local.ps1", """
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$freeBytes = (Get-PSDrive C).Free
if ($freeBytes -lt 32GB) { throw '설치 임시 용량과 최소20GiB 여유를 위해32GiB 이상이 필요합니다.' }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { py -3.10 -m venv .venv }
& ./.venv/Scripts/python.exe -m pip install --upgrade pip
& ./.venv/Scripts/python.exe -m pip install numpy==2.2.6 PyYAML==6.0.3 filelock==4.1.0 typing_extensions sympy networkx jinja2 fsspec pytest==8.4.2
& ./.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
& ./.venv/Scripts/python.exe -m pip install -e . --no-deps
& ./.venv/Scripts/python.exe -B -X utf8 -m fdt_local preflight --config configs/local_baseline.yaml
""")
print("Documentation and non-executed research configs written")
