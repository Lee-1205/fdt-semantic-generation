# V38 네 번째 넓은 원문 복원 학습 계약

V37의 저장된 전체 모델/91개 populated Adam/scaler/모든 저장 RNG를 그대로 이관한다. 동일111732306parameter/38556357trainable/새parameter0, 세 K_P/K_C/K_E 읽기와 Commit-only decoder, p50prefix corruption clean.5+corrupt.5, 추론/refiner/time/velocity 동결/denoise0/copy0를 유지한다. K_P certified Writer·K_C 전체 층 prefix·K_E literal selector/transport가 미완성이란 범위도 바뀌지 않는다.

새 목적은 관측 개발 byte75.27→76.13% 개선 추세가 있지만 전체 문장 복원이 부족한 상태에서, 같은 언어 codec의 네 번째 넓은 유한 순회를 학습하는 것이다. 새 구조·새 corpus·추가 baseline·denoising을 혼합하지 않는다. 기존TRAIN8192문서/32768구간/2926478target UTF8bytes, seed37001/batch8/4096attempt, AMPskip도 노출/커서 소비, 재생/추가epoch0. 기존 baseLR 새 finite cosine만 적용하고 Adam step·moments는 초기화하지 않는다. 참고64 반복 적응/관측 dev 다중사용/문서당4구간 상관 한계를 유지한다.

새1200초/reserve120/cutoff1080/17000call/calculated15060/output6GiB/C시작35GiB/GPU12GiB다. 보수적5update/sec 계산820train+60startup+60checkpoint+60assessment+120shutdown=1120<1200. V37은 전체worker646.737초에4095update로 이 범위보다 짧았다. 과거 시계·미사용 예산을 열거나 새 상한을 확대하지 않는다. Runtime C22GiB/추가checkpoint1GiB 검사와 항상C20GiB 보호, import0/mid2048/final4096+emergency fullstate, 숫자/동결/gradient/AMP/진행/디스크/시계 STOP 및 독립 owned OS 감독은 동일하다.

ROOT7CPU검사 PASS32.23초, actual full-size worker CPUpreflight와 독립 owner --validate PASS, 실제 Windows CPUfixture native24292→31624 identity/자연종료/allownedgone PASS다. tiny fixture를 과거 모든 epoch 재실행 대신 canonical populated91 상태로 한정했다. 이것은 실제 CUDA 복구 증거가 아니다. 최초 pytest 기본 TEMP 접근 거부6setup오류/1pass XML을 보존하고 새로운 workspace basetemp에서 검사를 완료했다. 데이터·모델 코드를 바꾼 실패가 아니다.

CPU 준비 완료는 GPU 실행이나 개선 증거가 아니다. 실제 새 seal/resource/native 검사 뒤 ONE hidden launch만 한다. 실제 학습 중 ROOT는 OS 감독만 하고 모든 owned/proxy 종료 뒤 저장 출력80쌍을 직접 읽어 의미 보존·문자열 유사도·teacher 지표를 구분한다. E2final/registry 접근·에이전트·원본 삭제·자동 재시작·실행 중 sealed 코드/상한 변경은 없다.
