# FDT vNext: 통합 의미 처리 아키텍처와 수학 명세

**문서 버전:** v1.1 최종 설계 점검·개발 계획, 2026-10-03  
**범위:** 아키텍처, 전체 계산 과정, 수식, 텐서 차원, 기억 접근 권한, 검증과 실패 반환. 학습 실행·데이터셋 비율·손실 계수·학습 일정은 이번 확정 범위에 포함하지 않는다.  
**기준:** 보존된 v0.1 명세와 현재 대화의 설계 원칙. 수식과 인터페이스를 실행 가능한 기준으로 선택한 문서이며, 학습된 모델의 의미 복원·추론 성능을 입증한 문서는 아니다.

## 1. FDT의 중심과 확정 범위

FDT는 자연어를 여러 퍼지 앵커에 대한 수치적 소속도와 문맥 값으로 표현하고, 그 표현에 관계 연산을 적용하며, 디퓨전으로 분산된 의미의 잠재 상태를 반복 정제한다. 앵커는 사람이 개념 이름을 하나씩 지정한 사전이 아니다. 소속도는 개념적 유사성·활성의 모델 값이며 사실의 참 확률과 동일하지 않다.

외부 입력이 있으면 언어 인코더가 초기 의미 조건과 문맥 기억을 구성한다. 외부 입력이 없으면 학습된 의미 prior, 영구 기억, 기존 목표 또는 NULL 목표에서 초기 조건을 만든다. 두 경로는 같은 내부 의미 연산기로 합류한다. 내부 연산 동안 자연어 토큰을 생성하지 않는다.

Semantic Commit은 확정된 내용, 출처에 진술된 내용, 검증되지 않은 지식·가설, 모순과 미해결 정보를 구분해 보존한다. 자연어 표현은 Commit 이후 실행되고 Commit을 수정하지 않는다. 검증되지 않은 내부 가설을 영구 사실 기억에 기록하지 않는다.

이번에 고정하는 것은 계산 정의와 상태 접근 규칙이다. 아래 크기는 하나의 구현 기준이며 최적 규모라는 주장이 아니다. 향후 크기 변경은 새로운 설계 버전으로 기록한다. 공개 자연어 전체에 대한 완전한 의미 압축이나 보편적인 진실 검증은 이 문서의 보장이 아니다.

## 2. 전체 경로와 상태 수명

\[
\begin{aligned}
\mathcal C_x&=\mathcal I_\theta(x,\mathcal K_P,\mathcal K_E), &&x\ne\varnothing,\\
\mathcal C_\varnothing&=\mathcal P_\eta(g,\mathcal K_P,\omega), &&x=\varnothing,\\
C&=\operatorname{Commit}\!\left(\mathcal T_\phi(\mathcal C,\mathcal K_P,\mathcal K_C,\mathcal K_E)\right),\\
y&=\mathcal V_\theta(C,s), &&s\text{는 표현 조건}.
\end{aligned}\tag{E01}
\]

`ω`는 재현 가능한 난수 seed다. `g`는 잠재 목표 또는 NULL이다. 새 자연어 목표를 받은 경우에는 외부 입력 경로를 사용한다. 표현 조건 `s`에는 언어·문체·길이 등의 표현 조건만 허용하며 새로운 사실이나 목표 변경은 새 내부 episode의 입력이다.

~~~mermaid
flowchart TD
    I["외부 입력"] --> E["언어 인코더"]
    N["외부 입력 없음"] --> P["의미 prior 초기화"]
    E --> M["기억 snapshot과 초기 조건"]
    P --> M
    M --> R["관계 연산과 의미 디노이징"]
    R --> V["근거·모순·불확실성 검증"]
    V -->|"추가 계산, 예산 내"| R
    V -->|"완료 또는 미해결 반환"| C["불변 Semantic Commit"]
    C --> L["자연어 표현과 원문 복사"]
    L --> F["출력 일치 검사"]
    F -->|"표현 재시도, 같은 Commit"| L
    F -->|"통과 또는 제한 반환"| O["최종 출력"]
~~~

| 상태 | 내용 | 내부 정제 중 | 표현 중 | 수명 |
|---|---|---|---|---|
| 모델 파라미터 | 앵커, prior 값, 관계 인수, 신경망 가중치 | 읽기 | 허용된 언어 가중치 읽기 | 모델 버전 |
| Persistent Fuzzy Memory `K_P` | 정적 prior와 인증된 장기 기록 | 읽기 전용 snapshot | 직접 접근 없음 | episode 간 |
| Contextual Fuzzy Memory `K_C` | 입력 문서별 long/recent N·D | 읽기 전용 snapshot | 직접 접근 없음 | 현재 episode |
| Exact Memory `K_E` | 원문, 식별자, 출처 버전, 근거 | 읽기 전용 snapshot | 직접 접근 없음 | 출처 버전 |
| Working State | Z, 관계 후보, binding, 임시 근거 캐시 | 갱신 가능 | 접근 없음 | 내부 episode |
| Commit `C` | 확정 상태, 상태 표지, 원문 복사 허가 | Commit 후 불변 | 읽기 전용 | 응답·감사 |
| 언어 KV·copy cursor | 출력 prefix와 복사 위치 | 존재하지 않음 | 갱신 가능 | 출력 시도 |

영구 기억 Writer는 Commit 뒤 실행하며, 갱신 결과는 다음 episode에만 반영한다. 현재 episode가 읽은 snapshot은 바뀌지 않는다.

## 3. 기호와 차원

벡터는 열벡터이고 상태 행렬은 각 행에 하나의 토큰 또는 슬롯을 저장한다. 각 `W`는 이름이 다른 독립적인 선형 투영이며 기본적으로 bias 없이 사용한다. MLP와 명시한 분류 head는 bias를 허용한다. 메모리의 numerator/mass는 float64, 모델 계산과 인터페이스는 float32를 기준으로 한다. 미래의 혼합 정밀도 구현은 이 기준의 수치 동치 검사를 거친다.

| 항목 | 기호·기준값 |
|---|---|
| 언어 backbone | L=20, d=1216, h=19, head 폭 64 |
| 로컬 attention | causal window W=64, RoPE |
| 언어 FFN | SwiGLU 중간 폭 3264 |
| Fuzzy 입력 층 | 0부터 세어 1,3,5,…,19의 10개 층 |
| 정규화 | RMSNorm, epsilon=1e-6 |
| 앵커 | K=256, 선택 k=8, key 폭 d_k=256 |
| 기억 value | d_v=256 |
| 작업 상태 | m=16, d_z=512 |
| 잠재 관계 | R=8개와 별도 NULL 코드, 인수 폭 q=32 |
| Refiner | 공유 4층, 8 heads×64, FFN 폭 1408 |
| 입력 section 요약 | 최대 64개, 각 폭 512 |
| 문서별 기억 bank | 최대 8개, 출처 경계 분리 |
| Exact 활성 후보 | 매 조회 최대 16개 |
| 임시 근거 누적 캐시 | 최대 160개, source ID·version으로 식별 |
| binding 개체 | NULL 포함 최대 128개, key 폭 128 |
| 결과 참조 후보 | 위 128개 + episode-local 개념 슬롯 16개 = 최대 144개 |
| 결과 주장 | 최대 64개, 주장별 최대 8개 증명 근거 |
| 증명 후보 key | 폭 256, 개체 key 128과 구분 |
| 의미 predicate | 모델 feature 폭 128; 검증 ontology ID는 별도 |
| 진단 | 16개 정해진 flag |
| 반복 | DDIM 기본 4단계, 보정 필요시 추가 4단계 1회 |
| 출력 재시도 | 첫 시도와 추가 2회, 항상 같은 Commit |
| 기본 입력·출력 예산 | 각 16K 토큰; 원문 복사 포함 출력 64KiB 제한 |

한계를 초과하면 `capacity_exceeded`를 기록한다. 넘치는 문서·개체·주장을 조용히 버리고 전체를 처리했다고 보고하지 않는다. 읽기 bank 분할·source selection의 누락은 미해결 정보로 남긴다. 모델의 총 파라미터 수와 실제 지연은 구현 후 계산할 값이다.

## 4. 앵커 좌표계와 퍼지 소속도

canonical 앵커 인덱스는 층 간 공유한다. 단일 학습 행렬 `A∈R^(256×256)`를 두고 입력 층·refiner의 각 조회기 `ℓ`는 다른 key view를 사용한다. view가 달라도 j번 앵커라는 주소 의미는 공유한다. 이는 구조적 인덱스 공유이며 학습 없이 의미 정렬이 자동 성립한다는 주장이 아니다.

\[
a_{\ell j}=\operatorname{unit}(W^A_\ell A_j),\quad
r_\ell(z)=\operatorname{unit}(W^q_\ell z),\quad
d_{\ell j}^2=\|r_\ell(z)-a_{\ell j}\|_2^2.
\tag{E02}
\]

Core의 관계 연산과 Persistent 읽기는 전용 core key view를 공유한다. Contextual 읽기는 10개 입력 층의 key view를 각각 사용한다. 모든 view는 같은 canonical 주소 인덱스를 참조한다.

`unit(u)=u/||u||`는 `||u||>ε`일 때만 정의한다. invalid query 또는 invalid anchor에는 소속도 0을 적용한다. ε=1e-8. 인코더 `W^q_ℓ`은 256×1216, refiner용은 256×512, `W^A_ℓ`은 256×256이다. 유효 unit vector에 대해서 `d²=2(1−cos)`이다.

\[
\sigma_{\ell j}=\operatorname{softplus}(q_{\ell j})+10^{-3},\quad
b_{\ell j}=\beta_{\ell j}-\frac{d_{\ell j}^2}{2\sigma_{\ell j}^2},\quad
\mu_{\ell j}=\mathbf1[j\in\operatorname{TopK}_8(b)]\operatorname{sigmoid}(b_{\ell j}).
\tag{E03}
\]

invalid 앵커는 Top-K 후보에서 제외하고 score를 −∞로 둔다. 동률은 앵커 ID 순서로 결정한다. 소속도 합을 1로 강제하지 않는다. 서로 다른 상태가 같은 일부 앵커에 소속될 수 있으며 소속도만으로 개체 identity를 판단하지 않는다.

\[
\operatorname{Norm}_+(u)_j=
\begin{cases}u_j/\sum_i u_i,&\sum_i u_i>\varepsilon,\\0,&\text{그 외},\end{cases}
\qquad \rho=\operatorname{Norm}_+(\mu),\quad\gamma=\max_j\mu_j.
\tag{E04}
\]

양의 질량을 가진 경우 분모에 ε를 추가하지 않고 정확히 정규화한다. ε는 영벡터 판정에만 사용한다. γ는 활성도이며 신뢰도·진실 확률이 아니다. 근거 없음, 거짓, 드문 앵커 사용을 γ 하나로 구분하지 않는다.

## 5. 입력 인코더: Local Attention과 Fuzzy 읽기·쓰기

입력 embedding에는 token embedding과 `input_mode` embedding을 더한다. 위치는 RoPE로 제공한다. 토큰 i의 현재 표현만 사용하는 causal prefix 경로다. 각 head 폭은 64다.

\[
Q_{ih}=\operatorname{RoPE}_i(W^Q_h\bar h_i),\quad
K_{jh}=\operatorname{RoPE}_j(W^K_h\bar h_j),\quad
V_{jh}=W^V_h\bar h_j,\quad\bar h_i=\operatorname{RMSNorm}(h_i).
\tag{E05}
\]

\[
\operatorname{LA}_i=W^O\operatorname{Concat}_h\left[
\sum_{j=\max(1,i-63)}^i
\operatorname{softmax}_{j}\!\left(Q_{ih}^{\top}K_{jh}/8\right)V_{jh}\right].
\tag{E06}
\]

\[
\operatorname{FFN}(u)=W_2\left[\operatorname{SiLU}(W_1u)\odot W_3u\right].
\tag{E07}
\]

언어 층 W1,W3는 3264×1216, W2는 1216×3264다. Refiner에서는 각각 1408×512와 512×1408이다.

\[
u_i=h_i+\operatorname{LA}_i,\quad
\tilde u_i=u_i+\mathbf1[\ell\in\mathcal L_F]W^F_\ell c_i,\quad
h'_i=\tilde u_i+\operatorname{FFN}(\operatorname{RMSNorm}(\tilde u_i)).
\tag{E08}
\]

Fuzzy query는 RMSNorm(u_i)에서 만들고 value는 `v_i=W^v_ℓ RMSNorm(u_i)∈R256`이다. `W^F_ℓ`은 1216×256. E08의 c_i는 E12의 문맥 읽기 c_i^C와 E14의 입력용 prior 읽기 c_i^P의 합이다. 각 경로는 자체 활성도 gate를 유지한다. 인코더는 i−1까지의 기억을 읽은 뒤 현재 토큰을 기록한다. 문서 경계에서 local KV와 Contextual 상태를 초기화하며 다른 문서의 상태를 직접 이어붙이지 않는다.

\[
N^L_j(i)=N^L_j(i-1)+\mu_{ij}v_i,\qquad
D^L_j(i)=D^L_j(i-1)+\mu_{ij}.
\tag{E09}
\]

\[
\lambda_j=10^{-6}+(1-2\cdot10^{-6})\operatorname{sigmoid}(\delta_j),\quad
N^R_j(i)=\lambda_jN^R_j(i-1)+\mu_{ij}v_i,\quad
D^R_j(i)=\lambda_jD^R_j(i-1)+\mu_{ij}.
\tag{E10}
\]

\[
S^b_j(i)=
\begin{cases}N^b_j(i-1)/D^b_j(i-1),&D^b_j(i-1)>\varepsilon,\\0,&\text{그 외},\end{cases}
\quad b\in\{L,R\}.
\tag{E11}
\]

`D=0`는 관찰 기록이 없는 상태다. 이 자리에 prior 값을 넣지 않는다. prior는 별도 경로로 읽는다. N,D를 같은 수로 나누면 현재 평균은 같아도 다음 쓰기의 상대 가중치가 달라지므로 임의 rescale을 금지한다. 유한 float64의 범위를 벗어난 갱신은 실패로 반환하며, 의미를 바꾸는 재조정을 몰래 수행하지 않는다.

\[
g_{ij}=\operatorname{sigmoid}\!\left(\operatorname{MLP}_g[
u_i;S^L_j;S^R_j;\log(1+D^L_j);\log(1+D^R_j)]\right),\quad
c_i^C=\gamma_i\sum_j\rho_{ij}\left[g_{ij}S^L_j+(1-g_{ij})S^R_j\right].
\tag{E12}
\]

gate MLP의 입력 폭은 1730, hidden 폭은 128, 출력은 scalar다. 기억은 층별·문서별로 유지한다. 값들이 서로 다른 출처의 사실을 평균 내더라도 Exact 기록의 identity와 출처는 합치지 않는다. Fuzzy 평균은 검색과 조건화에 쓰이며 원문 사실의 대체 저장소가 아니다.

## 6. Persistent Memory: 학습 prior와 인증된 기록

\[
S_j^{P0}\in\mathbb R^{256},\quad
S_j^{PV}=
\begin{cases}N_j^P/D_j^P,&D_j^P>\varepsilon,\\0,&\text{그 외},\end{cases}
\quad
\mathcal K_P=\{A,S^{P0},N^P,D^P,\text{인증 기록 manifest}\}.
\tag{E13}
\]

S^P0는 파라미터에 학습될 의미 prior다. N^P,D^P는 근거 인증을 거친 기록의 누적값이다. 학습 prior의 활성만으로 근거가 확보됐다고 판단하지 않는다. 관계 인수도 모델 파라미터이며 임시 관계 후보와 구분한다.

\[
\zeta_j(z)=\operatorname{sigmoid}\left(\operatorname{MLP}_{P}[z;S_j^{P0};S_j^{PV};\log(1+D_j^P)]\right),\quad
c^P(z)=\gamma\sum_j\rho_j\left[\zeta_j S_j^{P0}+(1-\zeta_j)S_j^{PV}\right].
\tag{E14}
\]

Core용 gate 입력은 1025, hidden 128이다. 입력 층에는 [u_i1216;S^P0_256;S^PV_256;mass1]의 별도 gate(1729→128→1)를 사용한다. canonical value는 공유하고 query와 residual projection은 층별로 둔다. 인증 기록이 없다면 ζ=1로 강제해 prior만 읽는다. c^P는 값 평균이며 사실을 직접 확정하지 않는다. Claim의 출처 ID는 manifest와 Exact 경로를 통해 따로 확인한다.

\[
\eta_{\ell d}(z)=\operatorname{softmax}_{(\ell,d)\ \mathrm{valid}} f_C(z,\mathrm{bank}_{\ell d}),\quad
c^C(z)=\sum_{\ell,d}\eta_{\ell d}(z)W^C_\ell\left[
\gamma_{\ell}(z)\sum_j\rho_{\ell j}(z)\{g_jS^L_{\ell dj}+(1-g_j)S^R_{\ell dj}\}\right].
\tag{E15}
\]

bank summary는 유효 슬롯의 질량 가중 평균이다. bank score MLP는 `[z512;summary256]→128→1`, W^C는 256×256이다. Core read gate는 `[z512;SL256;SR256;2 masses]→128→1`이다. 유효 bank가 없으면 c^C=0이다. 문서 bank들의 값 혼합은 허용하지만 원문 출처와 검증 원장은 항상 분리한다.

## 7. Exact Memory와 근거 후보 조회

\[
e=\left(\mathrm{id},\mathrm{sourceID},\mathrm{version},\mathrm{span}_{byte},
\mathrm{rawBytes},\mathrm{SHA256},\mathrm{provenance},\mathrm{typedRecord?},\mathrm{embeddingEpoch},k_e,v_e\right).
\tag{E16}
\]

원문은 UTF-8 바이트로 보존하고 Unicode 정규화나 숫자 재표기를 하지 않는다. 해시는 원문과 source version의 변경 검출에 사용하며 외부 출처의 진실성 인증을 대신하지 않는다. provenance는 출처·수집 경로의 기록이다. 문서 chunk는 문장·문단 경계와 512토큰 상한으로 나누고 원문 span을 유지한다. 256 key는 `unit(W_key AttnPool(H_chunk))`, 256 value는 `W_value AttnPool(H_chunk)`로 구성한다. 별도 원문 store를 사용하는 경우에도 같은 고정 모델 버전의 인코더를 쓰며 새 학습을 실행하지 않는다.

\[
q_a=W_E\operatorname{RMSNorm}(Z_a),\qquad
\mathcal E_t=\operatorname{Top16}_{e\ \mathrm{authorized,source\ version\ valid,embedding\ epoch\ valid}}
\left\{\max_a q_a^\top k_e/16\right\}.
\tag{E17}
\]

W_E는 256×512다. 초기 조회에는 초기 슬롯 또는 입력 goal을 사용한다. retrieval은 근거 후보 검색일 뿐 사실 승인이나 copy 승인이 아니다. 인가된 입력·출처 snapshot만 검색하며 저장소 갱신을 허용하지 않는다.

\[
p(e\mid Z_a,\mathcal E_t)=\operatorname{softmax}_{e\in\mathcal E_t\cup\{\mathrm{NULL}\}}
f_E(Z_a,v_e),\qquad
c^E_a=\sum_{e\in\mathcal E_t}p(e\mid Z_a,\mathcal E_t)v_e.
\tag{E18}
\]

score MLP는 768→256→1이다. NULL에는 별도 학습 scalar score를 사용하고 value=0으로 둔다. NULL 질량을 제거한 뒤 재정규화하지 않는다. 후보 없음·동률·개체 중복·역할 불일치를 따로 기록한다. 높은 p 하나만으로 source를 확정하지 않는다.

\[
V_{\mathrm{span}}(e)=\mathbf1[
\mathrm{authorized}(e)\land\mathrm{version\ matches}\land
\mathrm{byte\ bounds\ valid}\land\mathrm{UTF8\ boundaries\ valid}\land
\mathrm{snapshotSpan}(e)=e.\mathrm{rawBytes}\land
\mathrm{SHA256}(\mathrm{rawBytes})=e.\mathrm{SHA256}].
\tag{E19}
\]

\[
\mathcal E_{\mathrm{seen}}\leftarrow
\operatorname{Dedup}_{\mathrm{sourceID,version,span}}(\mathcal E_{\mathrm{seen}}\cup\mathcal E_t),\qquad
|\mathcal E_{\mathrm{seen}}|\le160.
\tag{E20}
\]

초기 후보 16개, 최대 8단계의 재조회에서 각각 16개, 최종 재조회 16개를 합쳐 최대 160개다. 이는 작업 상태의 후보 캐시이며 Exact Store에 기록하는 과정이 아니다. 각 단계는 활성 후보 16개를 읽고, 최종 Claim/Proof Head는 seen 전체를 참조한다. source ID는 단계별 후보 번호와 구분한다.

## 8. 개체 identity와 역할 binding

\[
b_a=\sum_{e\in\mathcal B}p_B(e\mid Z_a)\varphi_e,\quad
\varphi_e\in\mathbb R^{128},\quad\varphi_{\mathrm{NULL}}=0,\quad
p_B=\operatorname{softmax}_{e\in\mathcal B}( (W_BZ_a)^\top\varphi_e/\sqrt{128}).
\tag{E21}
\]

\[
\mathcal B=\{(\mathrm{entityID},\mathrm{sourceID},\mathrm{span},\mathrm{role},\mathrm{time},\mathrm{scope},\varphi_e)\}\cup\{\mathrm{NULL}\}.
\tag{E22}
\]

W_B는 128×512다. entity key는 해당 원문 embedding의 128차원 투영이다. 이름 문자열이 같아도 source·context가 다르면 자동으로 같은 개체로 합치지 않는다. role/time/scope 특징은 코드 embedding 또는 보존된 원문 참조로 제공한다. NULL 포함 128개를 넘으면 한계 flag를 설정하고 확인되지 않은 개체를 억지로 기존 ID에 합치지 않는다. 내부 확률적 binding은 후보이며 Claim에서 사용할 discrete ID는 검증 경로에서 다시 확인한다.

## 9. 입력과 무입력의 초기 의미 조건

\[
g=\begin{cases}
W_g\operatorname{AttnPool}(H_{\mathrm{request}}),&\text{관찰된 요청이 있음},\\
g_{\mathrm{existing}},&\text{기존 잠재 목표가 있음},\\
g_{\mathrm{NULL}},&\text{그 외},
\end{cases}
\quad W_g\in\mathbb R^{512\times1216}.
\tag{E23}
\]

요청 토큰 집합이 명시되지 않은 입력은 전체 관찰 입력을 request span으로 사용한다. AttnPool은 학습 query 하나로 masked attention을 계산한다. 입력 goal은 H 인코딩 후에 만든다. 인코더 이전에 H가 필요한 목표 함수를 호출하지 않는다. 무입력에서 기존 목표가 없으면 NULL로 시작하며 스스로 발견한 가설을 사실로 간주하지 않는다.

입력에서는 H를 원문 section 단위로 pool한 뒤 512차원으로 투영한다. section이 64개를 넘으면 누락된 원문 범위를 capacity flag로 남기고, Exact 경로에서 다시 찾을 수 있도록 span manifest를 보존한다.

\[
\mathcal U_x=[g;H_{\mathrm{sec}};W_Pc^P;W_Cc^C;W'_Ec^E],\qquad
Z_{\mathrm{init}}=\operatorname{Init2Block}(Q_{\mathrm{slot}},\mathcal U_x)
\in\mathbb R^{16\times512}.
\tag{E24}
\]

여기서 대괄호는 토큰 행 연결이고 각 memory projection은 512×256이다. 입력 context는 최대 68개 행이다. Init2Block은 비인과 self-attention, 입력 context cross-attention, SwiGLU의 2개 블록이며 폭 512, 8 heads, FFN 폭 1408이다. 모든 context는 이미 관찰된 입력에서 온다. 초기 Exact 후보는 goal query로 한 번 조회하고 E20 캐시에 넣는다.

\[
q_a^P=S_a^{\mathrm{slot}}+g,\qquad
Z_{\mathrm{init},a}=S_a^{\mathrm{slot}}+
W_{\mathrm{initP}}[c^P(q_a^P);g]+0.1\epsilon_a^0,
\quad W_{\mathrm{initP}}\in\mathbb R^{512\times768}.
\tag{E25}
\]

무입력 초기화에는 학습 슬롯 prior, 기존 또는 NULL 목표, Persistent 읽기를 사용한다. ε⁰는 별도의 seed로 뽑은 표준정규 탐색 변수다. 이후 Z_init query로 기존 인가된 Exact snapshot을 한 번 조회한다. 이 초기화는 자율적인 후보 탐색 경로이며 가설의 사실 승인을 포함하지 않는다. 무입력 Contextual bank는 비어 있다. Z_init은 diffusion 시작 잡음 Z₁과 다른 조건이다.

## 10. 관계 행렬: 주소 이동과 퍼지 강도

기존 v0.1의 RowNormalize(softplus(UVᵀ))에는 rank≤q 보장이 없다. 원소별 비선형 변환은 rank를 증가시킬 수 있다. 이번 기준은 양수 인수를 먼저 구성한다.

\[
B^{(r)}=\operatorname{softplus}(U^{(r)})+10^{-6},\quad
D^{(r)}=\operatorname{softplus}(V^{(r)})+10^{-6},\quad
G^{(r)}=B^{(r)}D^{(r)\top}/\sqrt q,
\quad r=1,\ldots,8.
\tag{E26}
\]

B,D는 256×32다. 이 절의 D는 관계 인수이며 기억 mass D와 superscript·문맥으로 구분한다. 구현 필드 이름은 relation_factor_right로 고정한다.

\[
T^{(r)}_{ij}=\frac{G^{(r)}_{ij}}{\sum_uG^{(r)}_{iu}},\qquad
F^{(r)}_{ij}=\frac{G^{(r)}_{ij}}{1+G^{(r)}_{ij}},\qquad
T^{(0)}=F^{(0)}=0.
\tag{E27}
\]

T는 행 확률 주소 이동, F는 비배타적 퍼지 관계 강도다. G와 T의 rank는 q 이하이고 F에는 같은 보장이 없다. NULL의 0번 코드는 관계 없음이다. 관계 코드 8개는 학습 잠재 코드이며 검증 ontology의 관계 이름과 동일하지 않다.

\[
\pi_{ab}=\operatorname{softmax}_{r=0,\ldots,8}
\operatorname{MLP}_{\mathrm{rel}}[
Z_a;Z_b;Z_{\mathrm{init},a};Z_{\mathrm{init},b};
b_a;b_b;b_{\mathrm{init},a};b_{\mathrm{init},b};\Delta_{ab};g].
\tag{E28}
\]

MLP 입력 폭 3088, hidden 512, 출력 9다. Δ_ab∈R16은 순서·역할·시간·범위의 pair 특징이다. b_init은 고정 Z_init query로 현재 안정된 ID table에서 계산한다. 입력 순서를 바꾸면 다른 ordered pair다. MLP가 실제로 방향을 학습했는지는 향후 의미 성능 검증 항목이다.

\[
p_{ab}^\top=\rho_a^\top\sum_{r=1}^8\pi_{ab,r}T^{(r)},\qquad
\nu_{ab,j}=\max_i\left[\mu_{a,i}\sum_{r=1}^8\pi_{ab,r}F^{(r)}_{ij}\right].
\tag{E29}
\]

유효 ρ에 대해 p의 합은 1−π_ab,0이다. 이 질량을 제거한 뒤 다시 정규화하지 않는다. ν는 각 원소가 [0,1]인 다중 소속도이며 합 1을 요구하지 않는다. 두 값을 하나의 확률로 혼동하지 않는다.

\[
(F\circ H)_{ik}=\max_jF_{ij}H_{jk},\qquad
\left(\mu\circ F\right)_j=\max_i\mu_iF_{ij}.
\tag{E30}
\]

max-product 합성은 절단 없는 유한 비음수 행렬에서 결합법칙을 만족한다. 중간 Top-K 절단은 이 성질을 보존하지 않는다. 숫자 관계 강도 자체를 논리적 함의나 참값으로 승격하지 않는다. 개체·부정·시간·범위·근거는 별도 검증한다.

\[
r_{ab}=W_R[p_{ab};\nu_{ab};b_a;b_b;b_{\mathrm{init},a};b_{\mathrm{init},b};\Delta_{ab}],\quad
\kappa_{ab}=\gamma_a\max_{r=1,\ldots,8}\pi_{ab,r},\quad
R_b=\left(\max_a\kappa_{ab}\right)
\sum_a\operatorname{Norm}_{+,a}(\kappa_{ab})\,r_{ab}.
\tag{E31}
\]

W_R는 256×1040이다. Norm_+,a는 수신 슬롯 b에 들어오는 송신 a 방향의 비중을 정규화한다. 앞의 max gate는 모두 약한 관계 후보를 정규화로 증폭시키지 않도록 한다. NULL만 활성인 경우와 모든 입력이 invalid인 경우 R_b=0이다.

## 11. 각 단계의 조건과 Refiner

\[
\mathcal C_{t,a}^{\mathrm{raw}}=
[c^P_a;c^C_a;c^E_a;R_a;b_a;b_{\mathrm{init},a};d_{\mathrm{diag}}],
\quad
\mathcal C_{t,a}=W_{\mathrm{ctx}}\mathcal C_{t,a}^{\mathrm{raw}},
\quad W_{\mathrm{ctx}}\in\mathbb R^{512\times1296}.
\tag{E32}
\]

첫 네 항은 각각 256, binding 두 항은 각각 128, 진단은 16이다. 매 단계 현재 Z_t에서 소속도·기억·관계·binding을 다시 계산한다. 모델·기억 snapshot, 원문, source ID, Z_init은 불변이다. 단 후보 캐시와 working diagnostics는 갱신할 수 있다.

\[
Q_{ah}=W^Q_h\bar Z_a,\quad K_{bh}=W^K_h\bar Z_b,\quad V_{bh}=W^V_h\bar Z_b,\quad
\operatorname{SA}(Z)_a=W^O\operatorname{Concat}_h
\sum_b\operatorname{softmax}_b(Q_{ah}^{\top}K_{bh}/8)V_{bh}.
\tag{E33}
\]

Refiner self-attention은 16개 슬롯의 비인과 attention이다. 슬롯 ID embedding으로 순서를 구분한다. 자연어 토큰 위치와 대응하는 슬롯을 강제하지 않는다.

\[
e(t)=W_{t2}\operatorname{SiLU}(W_{t1}
[\sin(t\omega_j);\cos(t\omega_j)]_{j=0}^{127}),
\quad\omega_j=10^{-4j/127}.
\tag{E34}
\]

sin·cos 연결은 256차원이고 W_t1은 512×256, W_t2는 512×512다. time embedding은 모든 슬롯에 더한다.

\[
\begin{aligned}
h^0_a&=W_ZZ_{t,a}+e(t)+s_a,\\
u^l&=h^l+\operatorname{SA}(\operatorname{RMSNorm}(h^l)),\\
w^l&=u^l+\operatorname{CA}(\operatorname{RMSNorm}(u^l),\mathcal U_t),\\
h^{l+1}&=w^l+\operatorname{FFN}(\operatorname{RMSNorm}(w^l)),\quad l=0,\ldots,3,\\
\widehat v_t&=\operatorname{RMSNorm}(h^4)W_v^\top,\quad
\mathcal U_t=[\mathcal C_t;Z_{\mathrm{init}};H_{\mathrm{sec}};g].
\end{aligned}\tag{E35}
\]

W_Z,W_v는 512×512다. CA는 Q/K/V/O 각각 512×512, 8 heads×64다. context는 최대 97개 행이다. 무입력은 section을 masked empty로 둔다. 4블록은 각기 다른 파라미터를 갖고, 이 4블록 전체를 모든 diffusion 단계에서 공유한다. dropout은 설계 검증 기준에서 0이다.

## 12. Fuzzy-Guided Diffusion의 정확한 정의

\[
\alpha_t=\cos(\pi t/2),\qquad \sigma_t=\sin(\pi t/2),\qquad
\alpha_t^2+\sigma_t^2=1,\quad t\in[0,1].
\tag{E36}
\]

구현에서는 t=0의 (α,σ)=(1,0), t=1의 (0,1)을 정확하게 대입한다. 부동소수 cosine의 근삿값으로 끝점을 정의하지 않는다.

\[
Z_t=\alpha_tZ_0+\sigma_t\epsilon,\quad\epsilon\sim\mathcal N(0,I),\qquad
v_t=\alpha_t\epsilon-\sigma_tZ_0.
\tag{E37}
\]

Z0는 출처·역할·관계가 구분되는 결과 의미의 연속 잠재 상태다. 목표 의미 상태의 형식은 23절에서 정의한다. 이 문서는 학습을 실행하지 않으며 E37은 diffusion 모델이 무엇을 복원하는지의 수학 정의다.

\[
\widehat Z_0=\alpha_tZ_t-\sigma_t\widehat v_t,\qquad
\widehat\epsilon=\sigma_tZ_t+\alpha_t\widehat v_t.
\tag{E38}
\]

이는 [Z0,ε]를 [Zt,vt]로 옮기는 직교 회전의 역변환이다. 두 부호가 뒤바뀌면 복원 정의가 달라진다.

\[
Z_{t'}=\alpha_{t'}\widehat Z_0+\sigma_{t'}\widehat\epsilon,\qquad t'<t.
\tag{E39}
\]

결정적 DDIM, η=0 기준이다. 여기서 결정적이라는 것은 초기 seed와 모든 snapshot이 같을 때의 단계 갱신을 뜻한다. 신경망의 예측이 정확하지 않으면 유한 단계에서 정답으로 수렴한다는 보장은 없다.

\[
Z_1=\epsilon^1,\quad\epsilon^1\sim\mathcal N(0,I),\qquad
(t_0,t_1,t_2,t_3,t_4)=(1,3/4,1/2,1/4,0).
\tag{E40}
\]

Z_init은 조건이지 Z1이 아니다. 이름이 비슷한 두 상태를 혼동하지 않는다. Gaussian noise는 Z에만 넣는다. μ, ID, 원문 bytes, 검증 상태와 source metadata에는 넣지 않는다. 내부 과정에서는 자연어 출력 토큰을 생성하지 않는다.

## 13. 구조적 주장과 증명 후보

\[
h_c=\operatorname{Claim2Block}(q_c,[Z_0;
V_{\mathcal E_{\mathrm{seen}}}W_{\mathrm{ev}}^\top;
\Phi_{\mathcal B_{\mathrm{claim}}}W_{\mathrm{entity}}^\top]),\qquad c=1,\ldots,64.
\tag{E41}
\]

Claim2Block은 2층의 512폭·8head query decoder다. W_ev는 512×256, W_entity는 512×128이다. source snapshot은 읽기 전용이다. head의 출력은 구조적 field이며 자연어 문장 토큰이 아니다.

\[
\begin{aligned}
\chi_a&=W_{\mathrm{concept}}Z_{0,a}\in\mathbb R^{128},\quad
\mathcal B_{\mathrm{claim}}=\mathcal B\cup\{
(\operatorname{ConceptRef}(\mathrm{episodeID},a),\chi_a,\mu_a):a=1,\ldots,16\},\\
p_c^S(e)&=\operatorname{softmax}_{e\in\mathcal B_{\mathrm{claim}}}
((W_Sh_c)^\top\varphi_e/\sqrt{128}),\\
p_c^O(e)&=\operatorname{softmax}_{e\in\mathcal B_{\mathrm{claim}}}
((W_Oh_c)^\top\varphi_e/\sqrt{128}),\quad
r_c=W_{\mathrm{pred}}h_c\in\mathbb R^{128}.
\end{aligned}
\tag{E42}
\]

W_S,W_O는 128×512다. predicate feature r_c와 검증 ontology ID를 분리한다. subject/object는 Exact entity ID, episode-local ConceptRef 또는 NULL이다. W_concept는 128×512이고 ConceptRef의 key φ_e는 χ_a다. ConceptRef는 단일 앵커의 개념 이름이 아니라 슬롯의 의미 feature와 다중 소속도 μ를 보존하는 참조다. 같은 앵커에 소속됐다는 이유로 개념 identity를 합치지 않는다. Claim context는 최대 16+160+144=320개 행이다. source가 없는 ConceptRef와 NULL을 verified subject/object로 승인하지 않는다. 무입력에서도 개념 참조를 선택할 수 있지만 해당 주장은 PARAMETRIC_UNVERIFIED 또는 HYPOTHESIS다. verified 경로는 명시적인 Exact entity ID를 사용한다. predicate·polarity·time/scope reference·존재 여부는 각각 별도 field head로 제안한다. ontology ID는 최대 128개 및 UNKNOWN이다. 임의 자연어 관계를 알려진 ontology에 강제로 끼워 맞추지 않는다.

\[
P_c=(s_c,\mathrm{predicateID}_c,o_c,\mathrm{polarity}_c,
\mathrm{timeRef}_c,\mathrm{scopeRef}_c,\mathrm{evidenceIDs}_c,\mathrm{status}_c).
\tag{E43}
\]

주장을 동일시하는 canonical key에는 subject, predicate, object, polarity, time, scope를 모두 포함한다. 방향을 반전한 관계나 부정, 다른 시간의 명제를 같은 key로 합치지 않는다. 근거가 여러 개인 동일 주장만 합친다. 이 원장은 제한된 검증 field를 갖는 출력 계약이며 모든 열린 자연어 의미의 완전한 기호화라고 주장하지 않는다.

\[
p^{\mathrm{proof}}_{c,j}(e)=\operatorname{softmax}_{e\in\mathcal E_{\mathrm{seen}}\cup\{\mathrm{NULL}\}}
\left[(W_{\mathrm{proof},j}h_c+\xi_j)^\top
\varphi_e^{\mathrm{proof}}/16\right],\quad j=1,\ldots,8.
\tag{E44}
\]

W_proof,j는 256×512, ξ_j와 φ_e^proof는 256차원이다. φ_e^proof는 W_proofKey256×256 k_e로 계산한다. φ_NULL^proof=0이고 NULL에 별도 score bias를 더한다. 이는 E21의 128차원 개체 key와 다른 기호다. 모든 증명 위치를 병렬 제안하며 선택한 source ID와 순서만 hard verifier에 전달한다. 중복·순환·역할 오류를 신경망 score만으로 승인하지 않는다.

## 14. 검증의 범위와 상태 taxonomy

\[
V(P_c,\mathcal E)=
V_{\mathrm{span}}\land V_{\mathrm{identity}}\land V_{\mathrm{scope}}\land
\left[V_{\mathrm{typed\ direct}}\lor V_{\mathrm{rule\ proof}}\right].
\tag{E45}
\]

V1은 source 인가·version·span·hash·UTF-8 경계를 확인한다. V2는 원문에 연결된 인가된 typed record와 subject/predicate/object/polarity/time/scope가 정확하게 일치하는지를 확인한다. 신경망이 원문을 파싱해 만든 record는 그것만으로 인가된 typed premise가 되지 않는다. V3은 지정된 규칙과 ordered proof chain을 검사한다. 이 검증은 출처와 규칙에 대한 상대적 검증이지 현실 세계의 참을 자동 인증하는 절차가 아니다.

\[
(a\succ b)\land(b\succ c)\Rightarrow(a\succ c)
\quad\text{동일 strict-order predicate, time, scope에서만}.
\tag{E46}
\]

첫 규칙 엔진의 허용 범위는 typed record의 직접 일치와 strict-order transitivity다. strict order를 일반 관계에 적용하지 않는다. 최대 8개 premise의 chain에서 연결 방향, 역할, 개체 ID, polarity, time/scope를 확인하고, source snapshot에 반대 관계·순환이 있으면 CONFLICT를 반환한다. proof에 없는 참값을 숨겨 채우지 않는다. 모든 자연어 entailment를 판정하는 oracle는 가정하지 않는다.

\[
\mathrm{status}\in\{
\mathrm{SOURCE\_RELATIVE\_VERIFIED},
\mathrm{SOURCE\_ASSERTED},
\mathrm{PARAMETRIC\_UNVERIFIED},
\mathrm{HYPOTHESIS},
\mathrm{UNRESOLVED},
\mathrm{CONFLICT},
\mathrm{INVALID}\}.
\tag{E47}
\]

검증된 typed record·허용된 proof만 첫 상태를 얻는다. 원문에 진술됐으나 독립 규칙 검증이 없는 내용은 SOURCE_ASSERTED다. 모델 prior의 지식은 PARAMETRIC_UNVERIFIED다. 후보 관계는 HYPOTHESIS, 정보 부족은 UNRESOLVED, 동시 상충은 CONFLICT, pointer·증명·역할 오류는 INVALID다. status head는 후보 분류를 제안할 수 있지만 최종 verified flag는 hard verifier가 결정한다. entropy·앵커 활성도·예측 점수를 검증 확률로 보고하지 않는다.

## 15. 보정 반복과 종료

\[
d_{\mathrm{diag}}=(f_1,\ldots,f_{16})\in\{0,1\}^{16}.
\tag{E48}
\]

고정 flag 순서는 source_missing, source_duplicate, span_invalid, version_stale, entity_ambiguous, role_invalid, polarity_conflict, time_mismatch, scope_mismatch, proof_invalid, source_conflict, unsupported_predicate, capacity_exceeded, numeric_invalid, output_plan_incomplete, budget_exhausted다.

기본 4단계 뒤 claim/proof를 추출해 검증한다. 근거가 존재하고 모델이 binding·역할·proof 경로를 잘못 제안한 경우에는 한 번의 보정 pass를 허용한다. 근거 자체의 부재, source conflict, 미지원 predicate, capacity 한계는 추가 noise만으로 해결됐다고 간주하지 않고 해당 상태를 반환한다.

\[
Z_{\tau}^{\mathrm{repair}}=\alpha_\tau Z_{\mathrm{clean}}+\sigma_\tau\epsilon^2,
\quad \tau=1/2,\qquad
\mathcal T_{\mathrm{repair}}=(1/2,3/8,1/4,1/8,0).
\tag{E49}
\]

ε²는 독립 seed다. 보정 pass도 E32–E39의 같은 denoiser를 사용하고 근거 cache를 누적한다. 마지막에 한 번 재조회하고 모든 seen 후보에서 Claim/Proof를 다시 추출·검증한다. 최대 내부 denoiser 호출은 8회다. 검증되지 않은 결과를 verified로 바꿔 종료시키지 않는다. 미해결 정보를 포함하는 Commit도 유효한 종료다. 새로운 외부 정보가 필요하면 별도의 episode로 처리한다.

## 16. Semantic Commit: 실행 불변성과 내용 구분

\[
C=\operatorname{DeepImmutableSnapshot}
\left(Z_0,P,h_{\mathrm{claim}},U,E,\mathcal B_{\mathrm{claim}},g,\mathrm{plan},
\mathrm{modelVersion},\mathrm{memoryVersion},\mathrm{sourceVersions},
\mathrm{seeds},\mathrm{budgetTrace}\right).
\tag{E50}
\]

snapshot은 immutable bytes를 backing storage로 두고, reader에 읽기 전용 해석이나 독립 복사본을 제공한다. writable owner가 남은 ndarray view나 노출된 mutable tensor를 불변 snapshot이라고 부르지 않는다.

E에는 선택 근거의 ID·metadata와 승인된 원문 bytes를 복사한다. 원래 store의 writable view를 Commit에 넣지 않는다. 배열과 record를 독립 복제하고 writable alias를 차단한다. source bytes는 immutable bytes다. 신경망 reader에는 항상 독립 tensor 복사본을 제공하며, 프레임워크에서 변경 가능한 tensor가 Commit backing bytes와 메모리를 공유하도록 변환하지 않는다. ConceptRef의 χ·μ도 Commit 안에 복제해 보존한다. 출력 계획에는 주장 순서와 status, 허용된 copy span, 미해결 사항이 포함된다. claim feature는 Writer와 Realizer의 입력으로 보존한다.

\[
\operatorname{SHA256}(\operatorname{CanonicalSerialize}(C))_{\mathrm{before}}
=\operatorname{SHA256}(\operatorname{CanonicalSerialize}(C))_{\mathrm{after}}.
\tag{E51}
\]

serialize는 field order, dtype, shape, little-endian 연속 배열 bytes, source version과 원문 bytes를 고정한다. hash는 변화 검출 수단이고 권한 차단을 대신하지 않는다. stop_gradient는 미분을 차단할 뿐 원본 변경을 막지 못하므로 Commit의 정의로 사용하지 않는다. 향후 gradient 정책은 학습 문서에서 정하며 실행 불변성과 구분한다.

## 17. 자연어 표현: Commit만 읽는 모드

\[
\widetilde C=\operatorname{Project}_{512}
([Z_0;h_{\mathrm{claim}};\mathrm{statusFeatures};g;
\mathrm{bindingFeatures};\mathrm{evidenceFeatures};\mathrm{planFeatures}]).
\tag{E52}
\]

512폭 항목을 행으로 연결하고 typed scalar·ID·status는 type embedding과 projection으로 변환한다. source bytes는 별도 copy permit에도 보존한다. Realizer에는 원래 입력 H, Persistent/Contextual/Exact store, 입력 local KV, 미확정 working Z를 전달하지 않는다.

\[
\begin{aligned}
Q_{ih}&=(W_Qu_i)_h,\quad K_{nh}=(W_K\widetilde C_n)_h,\quad
V_{nh}=(W_V\widetilde C_n)_h,\\
\operatorname{CA}_C(u_i)&=W_O\operatorname{Concat}_{h=1}^{19}
\sum_n\operatorname{softmax}_n(Q_{ih}^{\top}K_{nh}/8)V_{nh}.
\end{aligned}
\tag{E53}
\]

Q/O는 1216×1216, K/V는 1216×512다. 별도 adapter 5개를 0-based 언어 층 3,7,11,15,19의 LA 잔차 다음, FFN 이전에 적용한다. input_mode와 realization_mode는 1216차원 mode embedding으로 구분한다. 언어 LA·FFN 가중치는 공유하고 realization의 Fuzzy 읽기·쓰기는 비활성화한다. KV는 모드 간 분리하고 출력 시작 및 재시도 때 새로 만든다.

\[
\begin{aligned}
p(y\mid C,s)&=\prod_{t=1}^{T}p_\theta(y_t\mid y_{<t},\widetilde C,s),\\
h^{0,\mathrm{real}}_i&=e_i^{\mathrm{out}}+e_{\mathrm{mode}}^{\mathrm{real}}+
e_{\mathrm{style}}(s),\quad
e_0^{\mathrm{out}}=e_{\mathrm{start}}\in\mathbb R^{1216},\\
p_{\mathrm{token}}&=\operatorname{softmax}(W_{\mathrm{token}}u_t),\quad
p_{\mathrm{action}}=\operatorname{softmax}(W_{\mathrm{action}}u_t),\\
p_{\mathrm{permit}}(n)&=\operatorname{softmax}_n[
(W_{\mathrm{copy}}u_t)^\top k_n^{\mathrm{permit}}/\sqrt{512}],\\
C_{\mathrm{runtime},t+1}&=C_{\mathrm{runtime},t}.
\end{aligned}
\tag{E54}
\]

출력 position 0에는 별도 학습 start vector를 넣으며 원문 출력·토큰 예산에는 포함하지 않는다. 최초 action은 이 위치를 언어 backbone과 Commit adapter에 통과시킨 hidden state에서 선택한다. 실제 출력 토큰은 position 1부터 시작하고 copy 이후에도 start를 포함한 KV를 재구성한다. style은 검증된 language/tone/format enum embedding의 합이며 새 사실을 담는 free-text prompt를 허용하지 않는다. W_token은 입력 embedding과 묶인 V×1216 행렬이다. tokenizer·V·특수토큰 ID는 별도 버전으로 고정한다. W_action은 3×1216, W_copy는 512×1216, permit key는 Commit feature에서 만든 512차원 값이다. permit이 없으면 CopySpan을 masked out한다. GenerateToken에서 EOS 등 제어토큰을 제외하고 EOS action으로 종료한다. copy는 여러 bytes를 한 action으로 내보내므로 출력 분포는 이 action process를 포함한 조건부 분포다.

문법·표현을 위한 언어 prefix는 허용한다. 신경망 hidden state가 의미 관련 계산을 전혀 하지 않는다는 철학적 주장은 검증할 수 없다. 검사할 것은 Commit 이전 출력 없음, 입력 사실 bypass 없음, Commit 수정 없음, 주장 추적과 출력 일치다.

\[
\mathrm{action}_t\in\{\operatorname{GenerateToken},
\operatorname{CopySpan}(\mathrm{permitID}),\operatorname{EOS}\},\qquad
\mathrm{EmitCopy}(p)=C.\mathrm{permits}[p].\mathrm{rawBytes}.
\tag{E55}
\]

permit에는 source ID·version·span·원문 hash·용도를 포함한다. copy는 C에 보존된 원문 bytes만 출력하고 live Exact Store를 읽지 않는다. 원본 선택의 정답과 선택한 원본의 정확 복사는 별개다. 복사 후 언어 prefix를 고정 tokenizer로 다시 인코딩하고 KV를 재구성한다. tokenizer의 span 경계 합성성을 가정하지 않는다. copy가 출력 예산을 넘으면 부분 문자열을 전체 원문처럼 내보내지 않는다.

출력의 새 주장, 역할 반전, 숫자·단위·날짜 변경, 부정·시간 변경, 무허가 복사와 미해결 상태 누락을 검사한다. typed claim은 원장 trace와 hard check를 사용하고 열린 자연어는 UNVERIFIABLE일 수 있다. 완전한 보편 entailment checker를 가정하지 않는다.

## 18. 영구 기억 Writer와 인증 무효화

\[
w_c=\mathbf1[
\mathrm{status}_c=\mathrm{SOURCE\_RELATIVE\_VERIFIED}
\land\mathrm{certificateCurrent}_c
\land\mathrm{sourceAuthorized}_c
\land\mathrm{representationEpochCompatible}_c
\land\mathrm{newCanonicalClaim}_c].
\tag{E56}
\]

Writer는 hard verifier의 인증서를 확인한다. 인증 재확인, expected_memory_version 비교(CAS), canonical 중복 판정과 N,D 갱신을 하나의 원자적 transaction 안에서 수행한다. 충돌 시 새 revision으로 최대 한 번 재시도하며, 실패는 별도의 writer event로 기록하고 현재 Commit을 수정하지 않는다. status score는 인증서가 아니다. canonical duplicate는 polarity/time/scope를 포함한 key로 판단한다. 동일 반복으로 mass를 늘리지 않는다. 기존 사실의 새 근거는 인증 metadata에 추가하되 같은 의미 record를 반복 누적하지 않는다. Writer는 inference 중 가중치나 앵커 주소를 변경하지 않는다.

\[
\begin{aligned}
v_c^P&=W_{\mathrm{write}}h_c\in\mathbb R^{256},
\qquad (\mu_c,\rho_c,\gamma_c)=\operatorname{Route}_P(h_c),\\
N_j^{P,\mathrm{new}}&=N_j^P+\sum_c w_c\mu_{cj}v_c^P,\qquad
D_j^{P,\mathrm{new}}=D_j^P+\sum_c w_c\mu_{cj}.
\end{aligned}\tag{E57}
\]

철회가 있으면 active records의 정해진 ID 순서에서 aggregate를 재계산해 반복적인 float subtraction의 잔차를 피한다. Persistent snapshot을 만들기 전에도 source registry와 address epoch의 인증 유효성을 확인한다.

W_write는 256×512다. 각 record에 μ·value·claim key·source certificate·representation epoch를 저장하고 aggregate와 원자적으로 commit한다. 인증 철회 또는 출처 버전 변경 시 record를 inactive로 처리하고 저장된 기여를 N,D에서 제외한다. aggregate는 현재 active 인증 record의 합과 같아야 한다. 원본 기록과 철회 이력은 감사용으로 남긴다. representation epoch는 앵커 주소뿐 아니라 key/value projection, backbone, Claim feature, W_write, tokenizer를 포함한 전체 체크포인트의 좌표계 버전이다. epoch가 바뀌면 기존 N,D와 cached Exact embedding을 활성 조회에서 제외한다. 원문과 인증 이력은 유지하고, 새 모델에서 재인코딩·재검증한 기록만 새 aggregate로 편입한다. 옛 h_c·μ·value를 새 checkpoint projection에 바로 넣어 이식하지 않는다.

생성 문장 y는 Writer 입력이 아니다. SOURCE_ASSERTED·PARAMETRIC_UNVERIFIED·HYPOTHESIS·UNRESOLVED·CONFLICT·INVALID는 E57에 기여하지 않는다. Exact의 외부 원문 보존은 외부 진술 수집이며 모델 가설을 검증된 사실로 저장하는 행위와 다르다.

## 19. 출력 검사와 반환

\[
\mathrm{Fidelity}(y,C)\in\{\mathrm{PASS},\mathrm{FAIL},\mathrm{UNVERIFIABLE}\},\qquad
n_{\mathrm{attempts}}\le3,\quad C^{(1)}=C^{(2)}=C^{(3)}.
\tag{E58}
\]

FAIL이면 같은 C에서 표현만 최대 2회 재시도한다. 실패 또는 검증 불가가 지속되면 원장에 근거한 제한된 typed 표현이나 미해결 상태를 반환한다. fallback에도 동일한 토큰·byte·UTF-8·copy-permit 한계를 적용한다. 원장을 전부 표현할 예산이 없으면 사실을 바꿔 줄이지 않고 허용된 짧은 상태 응답을 반환한다. 최소 상태 응답조차 예산을 넘으면 빈 text와 구조적 상태를 반환한다. 출력에서 발견한 새 사실을 Core로 되먹임하지 않는다. 새 외부 근거가 필요한 요청은 새 episode다.

PASS는 정의한 검사 범위에서의 일치다. 열린 자연어 전체의 진실성을 보장하지 않는다. structural capacity 초과는 감지할 수 있지만 neural 압축에서 모든 정보 누락을 항상 감지할 수는 없다. 16개 슬롯과 Claim Head의 충분성은 이후 모델 검증 항목이다.

## 20. 전체 처리 알고리즘

~~~text
FDT_RUN(input_or_none, source_snapshot, existing_latent_goal, style, seeds):
    parameters = freeze_model_version()
    representation_epoch = parameters.representation_epoch
    episodeID = EpisodeIDFromRequestOrReplay()
    budgets = FrozenBudgetFromConfig(parameters.config)
    K_P, K_E, source_registry = CaptureConsistentEpisodeSnapshot(
        input_or_none, source_snapshot, parameters, representation_epoch)
    diagnostics = zero_flags(16)
    seen = empty_evidence_cache(max_size=160)

    if input_or_none is not None:
        # 각 토큰: local -> M_<i 읽기 -> residual/FFN -> 현재 토큰 쓰기
        H, sections, K_C = LanguageEncoder.encode(input_or_none)
        K_C = deep_readonly_snapshot(K_C)
        goal = EncodeGoalFromObservedRequest(H)
        active = ExactRetrieve(goal, K_E, limit=16)
        seen = Dedup(seen union active)
        Z_init = InputSemanticInitializer(H, sections, K_P, K_C, active, goal)
    else:
        K_C, sections = empty_context()
        goal = ExistingLatentGoalOrNULL(existing_latent_goal)
        Z_init = PriorSemanticInitializer(K_P, goal, seeds.prior)
        active = ExactRetrieve(Z_init, K_E, limit=16)
        seen = Dedup(seen union active)

    binding = StableEntityTable(seen, include_NULL=True, max_entities=128)
    Z = StandardNormal(shape=(16,512), seed=seeds.base_diffusion)

    for t, t_next in [(1,.75),(.75,.5),(.5,.25),(.25,0)]:
        # 아래 Refresh는 조회·소속도·관계 계산이며 기억 Store에 쓰지 않는다.
        context, seen, binding = RefreshWorkingContext(
            Z, Z_init, K_P, K_C, K_E, seen, binding, diagnostics, goal)
        v = Refiner.predict_v(Z, t, Z_init, context, sections, goal)
        Z_clean, eps = VPInverse(Z, v, t)
        Z = alpha(t_next)*Z_clean + sigma(t_next)*eps

    claim_refs = BuildClaimReferenceBank(Z, binding, episodeID)
    proposals, proofs = ClaimProofHeads(Z, seen, claim_refs)
    checked = HardVerifier(proposals, proofs, K_E)
    if RepairEligible(checked, evidence=seen, budget_remaining=4):
        diagnostics = EncodeVerificationDiagnostics(checked)
        Z = alpha(.5)*Z + sigma(.5)*StandardNormal(seed=seeds.repair)
        for t, t_next in [(.5,.375),(.375,.25),(.25,.125),(.125,0)]:
            context, seen, binding = RefreshWorkingContext(
                Z, Z_init, K_P, K_C, K_E, seen, binding, diagnostics, goal)
            v = Refiner.predict_v(Z, t, Z_init, context, sections, goal)
            Z_clean, eps = VPInverse(Z, v, t)
            Z = alpha(t_next)*Z_clean + sigma(t_next)*eps

    final_active = ExactRetrieve(Z, K_E, limit=16)
    seen = Dedup(seen union final_active)
    binding = ExtendStableIDs(binding, final_active)
    claim_refs = BuildClaimReferenceBank(Z, binding, episodeID)
    proposals, proofs = ClaimProofHeads(Z, seen, claim_refs)
    checked = HardVerifier(proposals, proofs, K_E)
    C = DeepImmutableCommit(Z, checked, seen, claim_refs, goal, seeds, budgets)

    # 현재 snapshot은 그대로 두고 인증된 기록을 다음 episode에 반영한다.
    PersistentWriter.update_from_verified_commit(C)
    C_hash = Hash(C)
    for attempt in [0,1,2]:
        output_KV = new_realization_cache()
        output, trace = Realizer.generate(C, style, output_KV, seeds.output[attempt])
        verdict = FidelityCheck(output, trace, C)
        assert Hash(C) == C_hash
        if verdict == PASS:
            return output, C, verdict
    fallback, fallback_trace = TypedFallbackOrUnresolved(C, budgets)
    return fallback, C, FidelityCheck(fallback, fallback_trace, C)
~~~

Refresh는 E17, E20, E21, E28–E32다. CaptureConsistentEpisodeSnapshot은 model·representation·source·memory revision을 일관되게 포착하고 BuildAndFreezeExactSnapshot으로 읽기용 index를 구성한다. revision이 달라지면 섞인 snapshot을 반환하지 않고 제한된 재시도 또는 상태 오류로 종료한다. BuildAndFreezeExactSnapshot은 관찰된 입력 원문과 인가된 기존 출처를 같은 고정 인코더로 index하고 읽기 전용으로 만든다. 이미 생성한 입력 H는 index에 재사용할 수 있다. 중복 인코딩을 하면 그 비용을 기록한다. 각 helper는 shape·인가·budget·numeric flag를 검사한다. nonfinite 상태가 나오면 마지막 유한 Z 또는 영벡터와 UNRESOLVED·numeric_invalid로 종료하고 verified claim을 만들지 않는다. 내부 단계에서는 Realizer, 자연어 출력, Persistent Writer를 호출하지 않는다. source 재조회는 고정 snapshot 읽기이며 새 출처의 외부 수집이 아니다.

## 21. 인터페이스와 접근 권한

| 모듈 | 입력 → 출력 | 쓰기 권한 |
|---|---|---|
| LanguageEncoder | 관찰 토큰 → H[N,1216], section≤64×512, K_C | 현재 문서 K_C·입력 KV |
| PriorInitializer | goal·K_P → Z_init[16,512] | 초기 working state |
| FuzzyRoute | query → μ,ρ,γ·selected IDs | 없음 |
| MemoryRead | frozen N,D·prior·query → value256 | 없음 |
| ExactRetrieve | query·K_E → 후보≤16 | 임시 seen 캐시만 |
| RelationOperator | Z·μ·binding·role/time → R[16,256] | 관계 후보만 |
| Refiner | Z_t·t·조건 → v_hat[16,512] | 다음 Z |
| ClaimProofHeads | Z·seen·binding → 구조적 후보≤64 | 후보 원장만 |
| HardVerifier | 후보·snapshot·rules → status·certificates | 검증 결과만 |
| SemanticCommit | 상태·미해결·근거 → C | 새 Commit만 |
| PersistentWriter | 인증된 C → records·N^P,D^P | 다음 episode의 K_P |
| Realizer | C·표현 조건 → 출력·trace | 출력 KV·cursor |
| FidelityCheck | 출력·C → verdict | 검사 결과만 |

모든 인터페이스는 model/source/address version을 전달한다. H는 Realizer 인자에 없다. 원문 복사는 Commit permit을 통과한다. 인증 metadata와 aggregate는 원자적으로 갱신한다.

## 22. 추가 수학 분석

\[
N^R_j(i)=\lambda_j^iN^R_j(0)
+\sum_{u=1}^i\lambda_j^{i-u}\mu_{uj}v_u,\qquad
D^R_j(i)=\lambda_j^iD^R_j(0)
+\sum_{u=1}^i\lambda_j^{i-u}\mu_{uj}.
\tag{E59}
\]

순차 갱신은 이 닫힌 형태와 일치한다. long 평균이 순서·role·부정을 자동 보존하지는 않는다. 그 정보를 Exact와 binding metadata에 보존하는 이유다.

\[
d^{\mathrm{row},(r)}=G^{(r)}\mathbf1,\qquad
\rho^\top T^{(r)}
=\left[(\rho\oslash d^{\mathrm{row},(r)})^\top B^{(r)}\right]D^{(r)\top}/\sqrt q.
\tag{E60}
\]

양수 인수와 행 정규화 덕분에 T 이동을 O(Kq)로 계산할 수 있다. F의 비선형 변환에는 같은 인수분해 최적화를 적용하지 않는다.

\[
\operatorname{rank}(G)\le q,\qquad
\operatorname{rank}(T)\le q,\qquad
0<F_{ij}<1,\qquad 0\le\nu_j\le1.
\tag{E61}
\]

rank(T)는 양수 diagonal 왼쪽 곱으로 보존된다. rank(F)≤q는 보장되지 않는다. ν의 범위는 μ≤1, F≤1, 비음수 π의 합1로 성립한다. rank와 strict bound는 이상적인 실수 연산에 대한 성질이며 수치 구현에서는 dtype에 맞는 오차 허용을 적용한다. overflow·nonfinite는 numeric_invalid로 처리한다. 수치 범위가 의미 정답이나 논리 건전성을 증명하지 않는다.

\[
\begin{pmatrix}Z_t\\v_t\end{pmatrix}
=
\begin{pmatrix}\alpha_t&\sigma_t\\-\sigma_t&\alpha_t\end{pmatrix}
\begin{pmatrix}Z_0\\\epsilon\end{pmatrix},
\qquad R_t^\top R_t=I.
\tag{E62}
\]

직교성으로 E38이 성립한다. 정답 v를 넣은 검사는 복원 부호·schedule을 확인하지만 실제 denoiser 오류를 측정하지 않는다. pure noise 시작은 t=1 marginal과 맞고, 실제 rollout은 예측 오차로 중간 분포가 달라질 수 있다.

\[
\mathcal K_{\mathrm{read},i}=\mathcal K_{\mathrm{written},<i},\qquad
C_{\mathrm{after\ output}}=C_{\mathrm{before\ output}},\qquad
\Delta\mathcal K_P^{\mathrm{unverified}}=0.
\tag{E63}
\]

각각 입력 인과성, Commit 실행 불변성, 미검증 가설의 영구 사실 승격 금지를 뜻한다.

## 23. 목표 의미 상태: 미래 학습 연결 인터페이스

깨끗한 상태 Z0를 답안 문장의 embedding 하나로만 정의하지 않는다. 구조적 Outcome Record를 우선 형식으로 삼는다. 이 인터페이스는 설계에 포함하며 현재 학습·정답 데이터 호출은 수행하지 않는다.

\[
u_c=\operatorname{Project}_{512}
[s_c;r_c;o_c;\mathrm{polarity}_c;\mathrm{time}_c;\mathrm{scope}_c;
\mathrm{evidenceFeature}_c;\mathrm{status}_c],\qquad
Z_0^{\mathrm{target}}=\operatorname{OutcomeCodec}_{2\mathrm{block}}
(Q_{\mathrm{slot}},[u_c]_{c=1}^{n\le64}).
\tag{E64}
\]

Codec의 raw feature는 subject128, predicate128, object128, polarity one-hot2, time16, scope16, evidence256, status one-hot7의 681차원이다. projection은 512×681이다. source 없는 entity는 canonical concept feature 또는 NULL을 사용한다. evidence 평균이 source identity를 대신하지 않도록 ID와 원문 manifest를 따로 보존한다.

OutcomeCodec은 512폭·8head·FFN1408의 2-block query encoder다. field는 역할을 구분하는 embedding과 projection으로 만들고 출력은 16×512다. source bytes·metadata는 latent 밖에도 유지한다. 목표 파라미터 관리·학습 목적함수·데이터는 이번 문서에서 정하지 않는다. inference는 정답 OutcomeCodec을 호출하지 않는다.

원장이 없는 설명을 어떤 목표 상태로 인코딩할지는 후속 학습 설계에서 정한다. 그 선택을 숨겨 범용 의미 추론이 이미 정의·검증됐다고 주장하지 않는다. Codec이 관계·개체를 구별하고 필요한 정보를 담는지 경험적으로 검증해야 한다.

## 24. 계산 비용과 예산

\[
\begin{aligned}
\mathrm{Cost}_{\mathrm{encoder}}&=
O(NLW d)+O(NL d f_{\mathrm{lang}})
+O(NL_FK d_k)+O(NL_Fk d_v),\\
\mathrm{Cost}_{\mathrm{core}}&=
O\!\left(S_{\mathrm{den}}\left[
4((m+p)d_z^2+m^2d_z+mpd_z+md_zf_z)
+mL_F(Kd_k+d_zd_k)+RmKq+m^2RkK
+m^2(d_{\mathrm{rel}}f_{\mathrm{rel}}+d_{\mathrm{msg}}d_v)
+md_{\mathrm{ctx}}d_z
+\mathrm{MemoryGates}+\mathrm{Binding}+\mathrm{ExactLookup}\right]\right).
\end{aligned}\tag{E65}
\]

f_lang=3264, f_z=1408, p≤97, S_den≤8다. 모든 K key를 평가하는 routing과 선택 k개의 읽기·쓰기는 다른 비용이다. F는 사전 계산할 수 있고 8×256×256 float32로 2MiB다. ν는 π를 혼합한 뒤 max를 취하므로 max의 전후를 바꾸지 않는다. T의 인수분해 최적화를 F에 적용하지 않는다.

dense Exact scan은 M개 후보에 O(mM×256)이다. ANN은 별도 구현 선택이며 검색 성능을 추가 확인해야 한다. Claim/Proof의 seen160 조회, source/rule 검증, Writer, 출력과 copy 이후 KV 재구성 비용을 별도로 집계한다. 희소 읽기만으로 전체 모델을 O(Nk)라고 보고하지 않는다. 실제 FLOPs·VRAM·속도 우수성은 주장하지 않는다.

## 25. v0.1 대비 수정 기록

| v0.1 | 이번 기준 | 이유·상태 |
|---|---|---|
| 입력 경로 중심 | 무입력 prior를 같은 Core에 합류 | 통합 구조 요구 반영 |
| 층별 앵커 | canonical 주소·층별 key view | 관계 좌표 명시; 새 선택 |
| μ/(Σμ+ε) 읽기 | μ·ρ·γ 분리·zero guard | 약한 소속도 증폭 방지 |
| FP32 누적 재정규화 제안 | float64 기준·임의 N/D rescale 금지 | 다음 write가 달라지는 반례 |
| exp형 λ | 양 끝점에서 떨어진 sigmoid λ | 유한 수치 범위 선택 |
| prior·입력 기억 중심 | prior·인증 기록·문맥 기억 분리 | 사실 status와 수명 구분 |
| softplus(UVᵀ)의 저랭크 주장 | 양수 G·주소 T·퍼지 F 분리 | rank 오류 교정 |
| 관계 전부 비NULL | NULL 코드·질량 보존 | 없는 관계의 강제 선택 방지 |
| normalized 관계 이동 | p와 비배타적 ν 구분 | 퍼지 의미 연산 명시 |
| identity 세부 미정 | 안정된 ID·role/time/scope | 유사성과 동일 개체 구분 |
| 처음 찾은 후보만 사용 | 재조회·seen160 누적 | 근거 누락 방지 |
| approximate noise endpoint | 정확한 Gaussian 끝점·VP 역변환 | 초기 분포와 부호 정합성 |
| 4 또는 8의 미정 분기 | 기본4+조건부 보정4 최대1회 | 종료와 budget 규칙 선택 |
| verifier 세부 미정 | typed direct·strict-order proof | 검증 범위 명시 |
| stop_gradient Commit | deep immutable snapshot·hash | 미분과 실행 불변성 구분 |
| ID만 copy 전달 | 원문 bytes의 Commit permit | live 변경·무허가 복사 방지 |
| every4 adapter shape 미정 | 층3,7,11,15,19·Q/O1216·K/V512 | shape·mode 권한 고정 |
| 영구 가설 저장 금지 선언 | cert gate·중복·철회·epoch | Writer 동작 구체화 |
| 출력 실패 정책 미정 | 같은 C에서3회·typed fallback | 출력에서 Core로 피드백 방지 |
| 학습 비율·손실 계수 포함 | 현재 명세에서 제거·이전 버전 보존 | 설계 우선 지시 반영 |

새 수식·기준값은 이번 분석에서 선택한 설계 제안이다. 과거 대화의 합의나 실험 결과로 소급하지 않는다. 사용자 원칙과 기존 크기를 유지하며 수학적 오류와 미정 인터페이스를 구체화했다.

## 26. 수학·프로토콜 검사 기록

검사일: 2026-10-02. Seed: 20261002. NumPy: 2.3.5. 대수 계산 float64. **22개 통과, 실패 0개.**

| 번호 | 검사 | 결과·수치 |
|---:|---|---|
| 1 | 소속도 범위·비배타성·Top-8 | PASS; mu_sum=7.69202688436863; rho_sum=1.0 |
| 2 | 약한 소속도의 활성도 gate | PASS; old_strength=0.9999875001562479; new_strength=0.0001 |
| 3 | NULL·invalid query의 zero guard | PASS; all_zero=true |
| 4 | 기존 softplus 저랭크 주장 반례 | PASS; before_rank=1; after_rank=8 |
| 5 | G·T의 rank와 F 범위 | PASS; ranks_G_T_F=[3, 3, 12] |
| 6 | T 인수분해 이동 동치 | PASS; max_error=2.7755575615628914e-17 |
| 7 | NULL 관계의 질량 보존 | PASS; null_probability=0.35; transport_mass=0.65 |
| 8 | 퍼지 관계 소속도의 비배타성 | PASS; nu_sum=5.499275114244317; max_nu=0.5226939955957047 |
| 9 | max-product 결합법칙 | PASS; max_error=1.1102230246251565e-16 |
| 10 | 중간 Top-K 절단 반례 | PASS; full=0.6; truncated=0.06999999999999999 |
| 11 | 관계 방향 구분 | PASS; forward=0.9; reverse=0.1 |
| 12 | NULL·약한 관계 메시지 gate | PASS; null_message_zero=true; weak_amplitude=1e-05 |
| 13 | prefix 인과성·읽기 후 쓰기 | PASS; unchanged_prefix_reads=6; first_read_zero=true |
| 14 | 최근 기억의 닫힌 형태 | PASS; max_error=3.3306690738754696e-16 |
| 15 | N·D 임의 rescale 반례 | PASS; next_base=10.0; next_rescaled=50.5 |
| 16 | VP 직교 회전·역변환·끝점 | PASS; max_error=8.881784197001252e-16 |
| 17 | 정답 velocity의 DDIM 기본·보정 경로 | PASS; base_error=4.440892098500626e-16; repair_error=4.440892098500626e-16; oracle_only=true |
| 18 | Commit alias 차단·불변 hash | PASS; readonly_view_counterexample=true; sealed_unchanged=true |
| 19 | 영구 쓰기 인증·중복·철회 | PASS; eligible_statuses=1; revoked_aggregate_matches_rebuild=true |
| 20 | 원문 bytes 복사·snapshot span binding | PASS; byte_length=6; normalization_applied=false; forged_span_rejected=true |
| 21 | 증명 방향·역할·시간·순환 | PASS; valid_chain_accepted=true; invalid_cases_rejected=4 |
| 22 | 인터페이스 차원·근거 캐시 예산 | PASS; relation_input=3088; message_input=1040; context_input=1296; evidence_pool_max=160; refiner_context_max=97 |

65개 번호 수식의 태그 순서와 수학 delimiter·brace·환경 짝을 확인했다. 실제 adapter의 행렬 투영 차원과 681차원 Outcome feature도 검사했다. 이 검사는 LaTeX 렌더러의 시각 검사를 포함하지 않는다.

17번은 정답 velocity를 대입한 수식 검사다. NN denoiser가 정답을 예측했다는 결과가 아니다. 이전 작업의 임시 검사 파일이 사라졌으므로 이번 기록은 복원한 명세에서 새로 실행한 검사 결과다.

검사는 작은 행렬과 상태 전이의 algebra·protocol 검사다. 학습·optimizer 실행·실제 의미 추론 성능 평가를 포함하지 않는다.

## 27. 확정 계약과 이후 검증

이번 기준안은 입력·무입력 합류, 공유 앵커 좌표, 소속도/주소/활성도 분리, 세 기억의 분리, 방향 있는 관계 연산, VP velocity·DDIM 갱신, 근거 누적·binding, 범위가 명확한 검증, 보정 예산, Commit 불변성, Realizer 권한·원문 복사, 인증된 영구 쓰기를 정의한다.

구현 후 확인할 것은 실제 텐서·캐시·snapshot·인가·Writer 경로와 tokenizer/copy 동치성이다. 학습을 통해 확인할 것은 앵커의 의미 안정성, 역할·부정·개체 구별, Codec의 정보 보존, denoiser의 복원과 추론·검색·출력 충실성이다. 최적 반복 수나 데이터 비율은 현재 결정하지 않는다. 선행연구와 일부 구조가 겹친다는 이유로 이 설계를 폐기하지 않는다. 중심 가설을 검사 가능한 수식과 인터페이스로 정의하는 것이 이 단계의 결과다.

## 28. 수학 배경의 원문

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

위 논문은 diffusion과 DDIM 배경이다. FDT의 앵커·관계·기억·Commit이 이 논문과 동일하게 동작한다고 주장하지 않는다. 이번 작업은 신규성 판정이나 학습 실행이 아니라 아키텍처와 수식 명세화다.


## 29. 검사 재현 코드

아래 코드를 math_audit.py로 저장하고 이 명세와 같은 폴더에서 python3 math_audit.py를 실행한다. NumPy 외의 모델·GPU·학습 데이터가 필요하지 않다. 작은 수치 예시와 반례를 검증하는 코드이며 FDT 모델 구현은 아니다.

~~~python
"""FDT design algebra/protocol checks. No model training or optimizer."""
from pathlib import Path
import hashlib
import json
import re
from types import MappingProxyType
import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 20261002
rng = np.random.default_rng(SEED)
results = []
EPS = 1e-8

def record(name, detail):
    results.append({"name": name, "status": "PASS", "detail": detail})

def norm_positive(x):
    s = np.sum(x)
    return x / s if s > EPS else np.zeros_like(x)

def membership(query, anchors, beta=4., width=1., k=8):
    qn = np.linalg.norm(query)
    an = np.linalg.norm(anchors, axis=1)
    valid = an > EPS
    if qn <= EPS:
        return np.zeros(len(anchors))
    units = np.divide(anchors, an[:, None], out=np.zeros_like(anchors),
                      where=valid[:, None])
    b = beta - np.sum((units-query/qn)**2, axis=1)/(2*width**2)
    b[~valid] = -np.inf
    chosen = np.argsort(-b, kind="stable")[:min(k, int(valid.sum()))]
    mu = np.zeros(len(anchors))
    mu[chosen] = 1/(1+np.exp(-b[chosen]))
    return mu

def max_product(a, b):
    return np.max(a[:, :, None] * b[None, :, :], axis=1)

def vp(t):
    if t == 0:
        return 1., 0.
    if t == 1:
        return 0., 1.
    return np.cos(np.pi*t/2), np.sin(np.pi*t/2)

# 1. Independent bounded memberships and separate normalized addresses.
mu = membership(rng.normal(size=8), rng.normal(size=(12,8)))
rho = norm_positive(mu)
assert np.all((mu >= 0) & (mu <= 1)) and np.count_nonzero(mu) == 8
assert mu.sum() > 1 and abs(rho.sum()-1) < 1e-14
record("소속도 범위·비배타성·Top-8", {"mu_sum": float(mu.sum()), "rho_sum": float(rho.sum())})

# 2. Weak mass must not become a unit-strength read.
weak = np.full(8, 1e-4)
old_strength = weak.sum()/(weak.sum()+EPS)
new_strength = weak.max()*norm_positive(weak).sum()
assert old_strength > .99 and abs(new_strength-1e-4) < 1e-15
record("약한 소속도의 활성도 gate", {"old_strength": old_strength, "new_strength": new_strength})

# 3. Empty/invalid addresses are zero and finite.
assert np.array_equal(norm_positive(np.zeros(12)), np.zeros(12))
assert np.array_equal(membership(np.zeros(8), np.ones((12,8))), np.zeros(12))
record("NULL·invalid query의 zero guard", {"all_zero": True})

# 4. Nonlinearity can destroy the alleged old low-rank bound.
for _ in range(500):
    u, v = rng.normal(size=(8,1))*2, rng.normal(size=(8,1))*2
    old_g = u @ v.T
    transformed = np.logaddexp(0, old_g)
    transformed /= transformed.sum(axis=1, keepdims=True)
    if np.linalg.matrix_rank(transformed) == 8:
        break
assert np.linalg.matrix_rank(old_g) == 1
assert np.linalg.matrix_rank(transformed) == 8
record("기존 softplus 저랭크 주장 반례", {"before_rank": 1, "after_rank": 8})

# 5. Positive factors and row scaling preserve G/T rank, not F rank.
B = np.logaddexp(0, rng.normal(size=(12,3))*2)+1e-6
C = np.logaddexp(0, rng.normal(size=(12,3))*2)+1e-6
G = B @ C.T / np.sqrt(3)
T = G/G.sum(axis=1, keepdims=True)
F = G/(1+G)
ranks = [int(np.linalg.matrix_rank(x)) for x in (G,T,F)]
assert ranks[:2] == [3,3] and ranks[2] > 3
assert np.all((F > 0) & (F < 1)) and np.allclose(T.sum(axis=1), 1)
record("G·T의 rank와 F 범위", {"ranks_G_T_F": ranks})

# 6. Factorized transport equals direct transport.
fast = ((rho/G.sum(axis=1)) @ B) @ C.T / np.sqrt(3)
transport_error = float(np.max(np.abs(fast-rho@T)))
assert transport_error < 1e-14
record("T 인수분해 이동 동치", {"max_error": transport_error})

# 7. NULL probability is preserved as missing transport mass.
pi = np.array([.35,.15,.5])
mixture = pi[1]*T + pi[2]*T[:, ::-1]
address_mass = float((rho@mixture).sum())
assert abs(address_mass-.65) < 1e-14
record("NULL 관계의 질량 보존", {"null_probability": .35, "transport_mass": address_mass})

# 8. Fuzzy output is bounded elementwise but need not sum to one.
source_mu = np.full(12,.9)
mixed_F = .15*F + .5*F[:, ::-1]
nu = (source_mu[:,None]*mixed_F).max(axis=0)
assert np.all((nu >= 0) & (nu <= 1)) and nu.sum() > 1
record("퍼지 관계 소속도의 비배타성", {"nu_sum": float(nu.sum()), "max_nu": float(nu.max())})

# 9. Untruncated max-product is associative.
A, H, J = (rng.uniform(size=(5,5)) for _ in range(3))
assoc_error = float(np.max(np.abs(max_product(max_product(A,H),J)-max_product(A,max_product(H,J)))))
assert assoc_error < 1e-14
record("max-product 결합법칙", {"max_error": assoc_error})

# 10. Intermediate top-k can change a subsequent composition.
A2, H2 = np.array([[.7,.6]]), np.array([[.1],[1.]])
full, truncated = float(max_product(A2,H2)[0,0]), float(max_product(np.array([[.7,0.]]),H2)[0,0])
assert abs(full-.6) < 1e-15 and abs(truncated-.07) < 1e-15
record("중간 Top-K 절단 반례", {"full": full, "truncated": truncated})

# 11. Directed relations preserve the possibility of asymmetric values.
directed = np.array([[0.,.9],[.1,0.]])
forward = max_product(np.array([[1.,0.]]), directed)[0,1]
reverse = max_product(np.array([[0.,1.]]), directed)[0,0]
assert forward == .9 and reverse == .1
record("관계 방향 구분", {"forward": float(forward), "reverse": float(reverse)})

# 12. Normalize only address weights; retain relation activation amplitude.
kappa = np.zeros(3)
message = kappa.max()*np.sum(norm_positive(kappa)[:,None]*np.ones((3,4)), axis=0)
assert np.array_equal(message, np.zeros(4))
kappa = np.full(3,1e-5)
assert abs(kappa.max()*norm_positive(kappa).sum()-1e-5) < 1e-15
record("NULL·약한 관계 메시지 gate", {"null_message_zero": True, "weak_amplitude": float(kappa.max())})

# 13. Causal memory read before write: changing the suffix keeps prior reads.
mus, vals = rng.uniform(size=(12,3)), rng.normal(size=(12,4))
def prefix_reads(ms, vs):
    N, D = np.zeros((3,4)), np.zeros(3)
    reads = []
    for m,v in zip(ms,vs):
        S = np.divide(N,D[:,None],out=np.zeros_like(N),where=D[:,None]>EPS)
        reads.append(m.max()*np.sum(norm_positive(m)[:,None]*S,axis=0))
        N += m[:,None]*v
        D += m
    return np.array(reads)
before = prefix_reads(mus,vals)
changed_m, changed_v = mus.copy(), vals.copy()
changed_m[6:] = rng.uniform(size=(6,3))
changed_v[6:] = rng.normal(size=(6,4))
after = prefix_reads(changed_m,changed_v)
assert np.array_equal(before[0], np.zeros(4))
assert np.array_equal(before[:6],after[:6])
record("prefix 인과성·읽기 후 쓰기", {"unchanged_prefix_reads": 6, "first_read_zero": True})

# 14. Sequential decay equals an independently calculated closed form.
lam = .7
seq_N, seq_D = np.zeros((3,4)), np.zeros(3)
for m,v in zip(mus,vals):
    seq_N = lam*seq_N+m[:,None]*v
    seq_D = lam*seq_D+m
powers = lam**np.arange(11,-1,-1)
closed_N = np.einsum("i,ij,ik->jk",powers,mus,vals)
closed_D = (powers[:,None]*mus).sum(axis=0)
decay_error = float(max(np.max(np.abs(seq_N-closed_N)),np.max(np.abs(seq_D-closed_D))))
assert decay_error < 1e-14
record("최근 기억의 닫힌 형태", {"max_error": decay_error})

# 15. Scaling N/D preserves a current mean, not the next write.
N,D = 10.,10.
base_next = (N+100)/(D+1)
scaled_next = (.1*N+100)/(.1*D+1)
assert N/D == (.1*N)/(.1*D) and base_next == 10 and scaled_next == 50.5
record("N·D 임의 rescale 반례", {"next_base": base_next, "next_rescaled": scaled_next})

# 16. VP rotation and inverse at intermediate times and exact endpoints.
z0, eps = rng.normal(size=(16,512)), rng.normal(size=(16,512))
vp_error = 0.
for t in [0,.1,.25,.5,.75,1]:
    a,s = vp(t)
    zt, velocity = a*z0+s*eps, a*eps-s*z0
    vp_error = max(vp_error,float(np.max(np.abs(a*zt-s*velocity-z0))),
                   float(np.max(np.abs(s*zt+a*velocity-eps))))
assert vp_error < 1e-14 and vp(0)==(1.,0.) and vp(1)==(0.,1.)
record("VP 직교 회전·역변환·끝점", {"max_error": vp_error})

# 17. Oracle velocity tests only algebra, never learned denoising performance.
def oracle_ddim(start,times):
    z = start.copy()
    for t,t2 in zip(times,times[1:]):
        a,s = vp(t)
        eps_now = (z-a*z0)/s
        v = a*eps_now-s*z0
        clean, noise = a*z-s*v, s*z+a*v
        a2,s2 = vp(t2)
        z = a2*clean+s2*noise
    return z
base_error = float(np.max(np.abs(oracle_ddim(eps,[1,.75,.5,.25,0])-z0)))
repair_noise = rng.normal(size=z0.shape)
a,s = vp(.5)
repair_error = float(np.max(np.abs(oracle_ddim(a*z0+s*repair_noise,[.5,.375,.25,.125,0])-z0)))
assert base_error < 1e-12 and repair_error < 1e-12
record("정답 velocity의 DDIM 기본·보정 경로", {"base_error": base_error, "repair_error": repair_error, "oracle_only": True})

# 18. Readonly view is not immutable; bytes-backed snapshot has no writable owner.
owner = np.arange(5,dtype="<f4")
view = owner.view()
view.flags.writeable=False
sealed = np.frombuffer(owner.tobytes(),dtype="<f4")
meta = MappingProxyType({"source_version":4,"raw":b"source"})
def snapshot_hash():
    return hashlib.sha256(sealed.tobytes()+meta["raw"]+str(meta["source_version"]).encode()).hexdigest()
digest = snapshot_hash()
owner[0] = 99
assert view[0] == 99 and sealed[0] == 0 and snapshot_hash() == digest
try:
    sealed.flags.writeable=True
    raise AssertionError("immutable bytes must reject writable flag")
except ValueError:
    pass
try:
    meta["source_version"] = 5
    raise AssertionError("immutable metadata must reject write")
except TypeError:
    pass
record("Commit alias 차단·불변 hash", {"readonly_view_counterexample": True, "sealed_unchanged": True})

# 19. Certificate gates, duplicate suppression, and revoked contribution removal.
statuses = ["SOURCE_RELATIVE_VERIFIED","SOURCE_ASSERTED","PARAMETRIC_UNVERIFIED","HYPOTHESIS","UNRESOLVED","CONFLICT","INVALID"]
def write_gate(status,current=True,authorized=True,novel=True):
    return int(status==statuses[0] and current and authorized and novel)
assert [write_gate(s) for s in statuses] == [1,0,0,0,0,0,0]
assert write_gate(statuses[0],current=False)==0
assert write_gate(statuses[0],authorized=False)==0
assert write_gate(statuses[0],novel=False)==0
weights = np.array([[.2,.8],[.5,.5]])
values = np.array([[1.,3.],[2.,4.]])
aggregate_N = weights.T@values
aggregate_D = weights.sum(axis=0)
removed_N = aggregate_N-weights[0,:,None]*values[0]
removed_D = aggregate_D-weights[0]
assert np.allclose(removed_N,weights[1,:,None]*values[1])
assert np.allclose(removed_D,weights[1])
record("영구 쓰기 인증·중복·철회", {"eligible_statuses":1,"revoked_aggregate_matches_rebuild":True})

# 20. Exact copy and actual snapshot/span binding, without Unicode normalization.
raw = "Cafe\u0301".encode("utf8")
source = b"prefix:"+raw+b":suffix"
span = (7,7+len(raw))
selected = source[slice(*span)]
assert selected == raw and len(raw)==6
forged = b"z"+raw[1:]
assert hashlib.sha256(forged).digest()!=hashlib.sha256(raw).digest()
assert source[slice(*span)] != forged
record("원문 bytes 복사·snapshot span binding", {"byte_length":len(raw),"normalization_applied":False,"forged_span_rejected":True})

# 21. Typed strict-order chain is direction/role/time/scope sensitive.
premises = [("A","GT","B","2026","S"),("B","GT","C","2026","S")]
def verify_chain(proof,claim,all_facts):
    if not proof or len(proof)>8 or any(p not in all_facts for p in proof):
        return False
    s,r,o,t,scope=claim
    if r!="GT" or any(p[1]!=r or p[3]!=t or p[4]!=scope for p in proof):
        return False
    if proof[0][0]!=s or proof[-1][2]!=o:
        return False
    if any(x[2]!=y[0] for x,y in zip(proof,proof[1:])):
        return False
    # Detect cycles in the complete provided source-relative GT graph.
    edges={}
    for a,rr,b,tt,ss in all_facts:
        if rr==r and tt==t and ss==scope:
            edges.setdefault(a,set()).add(b)
    def visit(node,path):
        if node in path:
            return True
        return any(visit(child,path|{node}) for child in edges.get(node,()))
    return not any(visit(node,set()) for node in edges)
assert verify_chain(premises,("A","GT","C","2026","S"),premises)
assert not verify_chain(premises,("C","GT","A","2026","S"),premises)
wrong=[premises[0],("C","GT","B","2026","S")]
assert not verify_chain(wrong,("A","GT","C","2026","S"),wrong)
assert not verify_chain(premises,("A","GT","C","2025","S"),premises)
cyclic=premises+[("C","GT","A","2026","S")]
assert not verify_chain(premises,("A","GT","C","2026","S"),cyclic)
record("증명 방향·역할·시간·순환", {"valid_chain_accepted":True,"invalid_cases_rejected":4})

# 22. Cross-module dimensions and evidence-pool bound.
assert 19*64==1216 and 8*64==512
assert 4*512+4*128+16+512==3088
assert 2*256+4*128+16==1040
assert 4*256+2*128+16==1296
assert 1216+2*256+2==1730 and 512+2*256+1==1025
assert 1216+2*256+1==1729
assert 3*128+2+2*16+256+7==681
assert 16+8*16+16==160 and 16+16+64+1==97
assert np.zeros((19,64)).reshape(1216).shape == (1216,)
token_queries=np.zeros((2,1216),dtype=np.float32)
commit_rows=np.zeros((3,512),dtype=np.float32)
Q=token_queries@np.zeros((1216,1216),dtype=np.float32).T
K=commit_rows@np.zeros((1216,512),dtype=np.float32).T
assert Q.reshape(2,19,64).shape==(2,19,64)
assert K.reshape(3,19,64).shape==(3,19,64)
assert (np.zeros((16,512))@np.zeros((512,512)).T).shape==(16,512)
record("인터페이스 차원·근거 캐시 예산", {"relation_input":3088,"message_input":1040,"context_input":1296,"evidence_pool_max":160,"refiner_context_max":97})

spec=(ROOT/"FDT_vNext_Formal_Architecture_Spec_KR.md").read_text()
in_fence=False
outside=[]
for line in spec.splitlines():
    if line.startswith("~~~"):
        in_fence=not in_fence
    elif not in_fence:
        outside.append(line)
spec="\n".join(outside)
blocks=re.findall(r"\\\[([\s\S]*?)\\\]",spec)
tags=[int(x) for x in re.findall(r"\\tag\{E(\d+)\}",spec)]
assert tags==list(range(1,66)) and len(blocks)==65
for block in blocks:
    stripped=re.sub(r"\\[{}]", "", block)
    level=0
    for char in stripped:
        if char=="{": level+=1
        elif char=="}": level-=1
        assert level>=0
    assert level==0
    assert re.findall(r"\\begin\{(\w+)\}",block)==re.findall(r"\\end\{(\w+)\}",block)
assert len(results)==22
report={"date":"2026-10-02","seed":SEED,"dtype":"float64","numpy_version":np.__version__,"passed":len(results),
        "failed":0,"scope":"Algebra and protocol examples; no learned-model capability test",
        "structural_lint":{"equations":65,"balanced_math":True,"sequential_tags":True},
        "checks":results}
(ROOT/"math_audit_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
~~~


## 30. 2026-10-03 최종 점검 결과

이 점검은 v1.0의 전체 본문과 65개 수식을 대조하고 기존 대수·프로토콜 검사 22개를 재실행한 것이다. 실제 FDT 신경망 코드와 학습된 가중치를 시험한 것으로 해석하지 않는다. 마지막 검토에서 발견한 구현 계약의 빈 부분을 v1.1에서 보완했다.

| 점검 항목 | 발견한 문제 | v1.1 처리 |
|---|---|---|
| 무입력 결과 참조 | E42가 Exact table만 참조하면 근거 없는 개념은 NULL 외에 선택 경로가 없음 | episode-local ConceptRef 16개를 별도로 추가; 결과 후보 최대144, source identity128 유지 |
| 행렬 방향 | Claim context의 source·entity 집합에 vector projection 표기를 그대로 쓰면 batch shape가 모호함 | V·Φ의 행렬에 Wᵀ를 오른쪽 곱; Claim context 최대320행 명시 |
| 의미 좌표계 | 앵커 주소만 버전 관리하면 value·Claim feature 변경 후에도 예전 누적값을 재사용할 위험 | 전체 checkpoint representation epoch를 비교; old N,D와 cached embedding 비활성화 |
| 동시 Writer | transaction 밖에서 gate·중복을 판정하면 동일 주장 두 번 기록·revision 유실 가능 | certificate·CAS·중복·aggregate 갱신을 한 transaction 안에서 처리 |
| 최초 출력 action | 빈 prefix에서는 action head에 줄 언어 hidden state가 정의되지 않음 | 출력되지 않는 1216차원 start vector와 position0 지정 |
| Commit tensor 반환 | immutable bytes라도 reader가 변경 가능한 공유 tensor를 얻으면 격리 계약이 깨짐 | 신경망 reader에는 반드시 독립 tensor 복사본 반환 |
| 실패 반환 | typed fallback이 전체 원장을 출력하면 일반 출력 예산을 우회할 수 있음 | 같은 token·byte·UTF-8·permit 한계; 최소 상태조차 불가능하면 빈 text와 구조적 상태 |
| 비용 | E65에서 context projection·pair 관계 MLP 등이 빠짐 | dense projection, relation/message MLP, condition projection, memory gate·binding 비용 추가 |
| 의사코드 | budget·episodeID·epoch와 입력 변수명이 충분히 정의되지 않음 | explicit config·captured snapshot·stage seed·claim reference bank로 연결 |

보완된 ConceptRef는 사람이 이름 붙인 개념 사전이나 단일 앵커 label이 아니다. 슬롯 의미 feature와 μ를 보존하며 동일 앵커 활성으로 identity를 합치지 않는다. source 없는 주장도 구조적으로 표현하되 HardVerifier가 이를 verified로 승인하지 않는다.

representation epoch 변경 때 원문과 인증 이력은 보존한다. 새 checkpoint에서 다시 만든 embedding·claim feature·μ·value만 활성 조회와 aggregate에 사용한다. 기존 값의 평균을 새 좌표계로 옮기는 자동 의미 보존 변환을 가정하지 않는다.

E65의 새 항에서 d_rel=3088, f_rel=512, d_msg=1040, d_ctx=1296이다. MemoryGates는 prior gate(1025→128→1), 문맥 gate(1026→128→1), bank score(768→128→1), value projection·mix의 비용을 포함한다. Binding은 512→128 projection과 최대128개의 ID key score를 포함한다. ExactLookup에는 검색 외에 active16개의 source selector(768→256→1) 비용도 합산한다. Fuzzy key view·F 행렬을 미리 계산하면 별도로 cache 생성 비용과 checkpoint 변경 시 재생성 비용을 기록한다.

예를 들어 d_z=512, m=16, p=97인 4블록에서 Q/K/V/O dense projection만 세어도 304,087,040 MAC이다. ordered-pair relation MLP와 message projection은 474,087,424 MAC이다. 이는 정의한 행렬 크기의 연산 수이며 전체 모델 FLOPs·실측 시간·속도 우수성 수치가 아니다.

추가 계약 검사 6개가 통과했다. 따라서 **기존 22개 + 추가 6개 = 28개 검사 그룹 통과**다. 추가 검사의 대상은 참조 공간, epoch, transaction과 byte budget의 작은 프로토타입이다. 신경망이 의미 있는 개념을 만들거나 실제 proof를 정확하게 선택한다는 증거는 아니다.

| 번호 | 추가 계약 검사 | 결과 |
|---:|---|---|
| 23 | 무입력 ConceptRef의 도달성·episode identity | PASS; without_exact_candidates=17; max_claim_reference_bank=144; concept_is_not_NULL=true |
| 24 | 체크포인트의 value 좌표계·cache epoch | PASS; same_anchor_different_epoch=true; old_contributions_excluded=true |
| 25 | 동시 Writer의 CAS·중복 억제 | PASS; created_records=1; conflict_detected=true; retry_does_not_double_mass=true |
| 26 | copy·fallback 예산·UTF-8 경계 | PASS; copy_not_partially_emitted=true; fallback_bytes=10; zero_budget_empty_text=true |
| 27 | 최초 출력 상태·Commit reader 복사 | PASS; first_action_logits=3; mutable_export_does_not_alias_commit=true |
| 28 | 투영·관계 MLP 비용과 Claim context | PASS; previous_terms=49840128; additional_projection_MACs=304087040; pair_MLP_MACs=474087424; claim_context_max=320 |

**점검 판정:** 명세는 기준 구현을 시작할 수 있는 v1.1 설계 계약으로 정리됐다. 실제 module forward, cache, GPU 경로, gradient, 전체 controller와 응답의 성능은 다음 개발 단계에서 검증해야 한다. 문서 검사를 통과한 것을 구현 완료로 표시하지 않는다.

## 31. 개발 순서와 단계별 완료 조건

현재 상태: 설계 문서·reference 수학 검사·작은 계약 프로토타입 완료. 아래 생산용 FDT 모듈은 아직 구현 완료로 간주하지 않는다. 날짜·학습 횟수·최적 데이터 비율을 임의로 확정하지 않고, 선행 단계의 완료 조건에 따라 진행한다.

| 단계 | 개발 내용 | 계획 산출물 | 다음 단계 진입 조건 |
|---:|---|---|---|
| 0 | v1.1 config·schema·접근 권한 고정 | config/schema 문서, version·epoch 규약 | 모든 field의 shape·dtype·NULL·mask·budget·실패 반환이 명시됨 |
| 1 | 코드 기반과 상태 수명 | config, types, snapshot, registry, seed, serialization 모듈 | snapshot을 외부 tensor로 바꿔도 hash 불변; source/model/representation revision 불일치 차단 |
| 2 | 수학 reference kernel | router, N/D memory, T/F, max-product, VP/DDIM 함수 | float64 reference와 tensor 경로 동치; μ·ρ·γ·NULL·decay·rank·부호 검사 통과 |
| 3 | Exact·identity·인증 기반 | raw store, span index, ID table, typed premises, certificate schema | bytes·version·인가·동명이인·중복·철회 fixture 통과; 자동 parser 결과를 trusted premise로 승격하지 않음 |
| 4 | 언어 입력 backbone | Local attention, Fuzzy read/write, section/goal encoder | 실제 local 구현; prefix 미래 누출0; 문서 경계 격리; streaming과 full 경로 허용 오차 내 일치 |
| 5 | 입력·무입력 initializer | InputInit, PriorInit, NULL goal, stage RNG | 양쪽 모두16×512; source·context 없음에서도 유한 결과; target answer 접근0 |
| 6 | 관계·binding·결과 상태 형식 | RelationOperator, Claim/Proof Head, ConceptRef, OutcomeCodec, HardVerifier | ordered role·NULL·time/scope 타입과 shape가 닫힘; trusted fixture의 verifier 오승인0; Codec 목표는 inference에 유입되지 않음 |
| 7 | 의미 refiner·diffusion | 4블록 D_phi, 4단계 schedule, 1회 repair | oracle-v algebra 통과; random-weight forward에서 finite·shape·budget·mask 검사; 내부 자연어 출력 호출0 |
| 8 | episode controller와 Commit | seed/snapshot capture, seen cache, 종료·error·immutable C | 후보≤160, source identity≤128, claim refs≤144, claims≤64, steps≤8; 실패도 status와 budget trace로 종료 |
| 9 | 자연어 실현·정확 복사·출력 검사 | shared-backbone realization mode, adapter5, action/copy heads, fallback | Memory/Encoder 호출 차단; C hash 불변; start·EOS·UTF-8·원문복사·재tokenize·3회 한계 검사 통과 |
| 10 | 인증된 영구 기억 Writer | transactional manifest·aggregate, CAS, dedup, epoch invalidation | 미검증 쓰기0, 동일 claim mass 중복0, stale cert/epoch 쓰기0, crash 후 manifest·N/D 일치 |
| 11 | 전체 통합과 재현 검증 | 전체 요청 fixture, negative/abstain 경로, checkpoint/replay trace | 정상·부재·모순·무입력·overflow 모두 종료; 같은 환경에서 snapshot·seed replay; source·answer bypass0 |
| 12 | 자원 측정·최적화·구현 버전 고정 | module별 param/MAC/time/peak memory 보고, 검증된 kernel, implementation tag | 전체 요청 비용을 실측; 최적화 전후 결과 동치; 목표 장비에서 지원된 길이만 표시; 실패 fixture 재실행 통과 |

0단계에서 문서상 선택은 완료했으나, 구현용 typed schema와 config validator는 1단계 코드에 반영해야 한다. 2·3단계는 외부 원문과 작은 숫자 fixture로 개발하고 학습 데이터를 소비하지 않는다. 4–10단계의 신경망 forward는 임의 초기화 또는 인가된 기존 가중치로 인터페이스만 시험한다. 미학습 네트워크의 의미 정답률을 이 단계의 완료 조건으로 두지 않는다.

6단계 OutcomeCodec은 표상 형식과 forward shape를 닫는 작업이다. 무작위 가중치의 Codec가 의미를 정확히 압축·복원한다고 주장하지 않는다. Codec와 Realizer의 실제 의미 보존은 향후 표상 학습 단계에서 검증한다. 모든 단계는 reference 경로를 먼저 만들고 최적화 경로를 나중에 추가한다.

최적화 전에 Local Attention이 mask만 씌운 N×N dense allocation인지 확인한다. 식이 local이라는 사실만으로 실제 O(NW) memory 경로가 구현됐다고 보고하지 않는다. 16K는 설계의 I/O 상한이고 현재 장비·학습·구현이 이미 지원한다는 인증이 아니다. 짧은 길이에서 시작해 전체 forward의 메모리·지연을 측정한 뒤 지원 길이를 늘린다.

미분이 있는 수학 kernel의 backward는 별도 검사한다. hard Top-K, max, NULL threshold는 경계에서 미분 불가능하므로 tie·선택 집합 변화는 discrete 정책으로 검사한다. 선택 집합이 고정된 연속 branch에서 float64 finite-difference/gradcheck를 수행하고, 이 검사를 optimizer update나 학습으로 혼동하지 않는다.

## 32. 통합 검증에서 반드시 포함할 시나리오

| 시나리오 | 확인할 결과 |
|---|---|
| 관찰 입력·직접 typed 사실 | Exact ID·source version·verified record 보존 |
| 동일 이름의 다른 개체 | 자동 identity 병합·역할 교환 금지 |
| source 없는 질문·무입력 prior | ConceptRef·미검증 상태 표현, 영구 사실 쓰기0 |
| 앵커 모두 invalid·근거 후보0 | μ·ρ·γ·source context에 NaN 없이 NULL 처리 |
| source의 반대 관계·시간/범위 충돌 | CONFLICT·UNRESOLVED 보존, 강제 verified 금지 |
| 정상 proof와 역방향·순환 proof | 올바른 chain만 정의한 rule 범위에서 승인 |
| source 변경·인증 철회 | 현재 Commit 불변, 다음 episode의 인증 aggregate 갱신 |
| checkpoint 교체·value projection 변경 | 기존 fuzzy aggregate·index epoch를 활성 경로에서 제외 |
| 동시 요청의 같은 사실 저장 | CAS·dedup 이후 단일 기여 |
| 숫자·조합문자·UTF-8 span copy | 원문 bytes 보존; copying correct source와 exact bytes를 별도 검사 |
| 빈 출력 prefix·EOS | start hidden state부터 첫 action, 제한 내 종료 |
| copy 이후 tokenizer 경계 변화 | 실제 고정 tokenizer의 전체 prefix 재인코딩·KV 재구성과 일치 |
| 일반 출력과 fallback budget 초과 | 사실 변경·부분 copy로 예산 우회하지 않고 제한 반환 |
| output style·seed·언어 변화 | 같은 C hash와 근거 보존; 출력 일치 여부는 별도 검사 |
| NaN·Infinity·source/index stale·capacity 초과 | bounded 종료·오류 상태·verified 쓰기 차단 |
| process 중단·Writer crash·replay | atomic manifest/aggregate·같은 환경의 재현성과 stage seed 보존 |

특정 fixture의 오승인0은 그 fixture 집합에서의 결과다. 열린 자연어 verifier 전체의 오류율0이나 보편적 truth oracle를 주장하지 않는다. 실제 응답 모델을 통해서만 평가할 의미·관계·환각 지표는 학습 뒤에 측정한다.

## 33. 개발 완료 후에만 진행할 학습 설계

12단계의 구현 검증 뒤 별도의 학습 명세를 작성한다. 여기서 target encoder의 관리, gradient detach/continuation, 문서 경계·정답 누출 방지, objective와 평균·가중 규칙, source split과 실제 노출 token 계수를 확정한다. 이 문서에서는 학습을 실행하거나 데이터셋 비율을 변경하지 않는다.

학습 순서는 다음을 기준으로 설계한다.

1. **언어·앵커·문맥 기억의 기반:** 문맥의 identity·방향·부정·시간 변화와 앵커 활성의 구별을 검증한다.
2. **의미 표상과 언어 실현:** Codec의 잠재 상태와 Commit만으로 필요한 사실·관계를 표현하는지 확인한다. 이 단계에서 표상 capacity가 부족하면 크기·구조를 새 버전으로 수정한다.
3. **Fuzzy-Guided Diffusion:** 입력 조건에서 의미 상태를 복원하고 실제 rollout에서 근거·개체·관계가 보존되는지 평가한다.
4. **통합·장문:** 앞선 모듈의 의미 성능이 확인된 뒤 공동 학습과 길이 확장을 검토한다.

oracle 상태의 언어 표현이 충분하지 않으면 denoiser 성능부터 높이는 순서로 진행하지 않는다. 표상 보존 실패와 diffusion 복원 실패를 별도 실패로 기록한다. 비교용 recurrent core 등은 diffusion을 제거하자는 결론이 아니라 선택한 중심 구조의 효과를 평가하기 위한 대조군이다.

## 34. 검사 범위와 재현 환경

NumPy reference는 기존 문서의 검사 코드를 사용한다. 신경망 구현은 PyTorch 기반을 개발 기준으로 삼되, 실제 환경에서 버전·backend·dtype·kernel·장비와 설정을 lockfile/manifest에 고정한다. seed만으로 다른 장비·버전 간 bitwise 동일성을 약속하지 않는다. 같은 환경·같은 연산 경로의 replay와, 서로 다른 구현 경로의 선언된 수치 오차를 구분한다.

공식 구현 참고:
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch gradcheck mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)

다음 추가 검사 코드를 final_review_checks.py로 저장하면 계약 prototype 6개를 재실행할 수 있다. 이는 생산용 FDT 모듈 구현이 아니다.

~~~python
"""
Supplementary v1.1 contract checks. Prototypes only, not FDT production modules.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Barrier
import hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parent
checks = []

def check(name, detail):
    checks.append({"name":name,"status":"PASS","detail":detail})

# 1. The pointer space must contain concepts even with no exact entities.
Z = np.zeros((16,512))
Z[:,:16] = np.eye(16)
W = np.eye(128,512)
concept_keys = Z @ W.T
keys = np.vstack([np.zeros((1,128)),concept_keys])
query = concept_keys[0]
logits = keys @ query / np.sqrt(128)
assert int(logits.argmax()) == 1 and keys.shape == (17,128)
assert 128+16 == 144
concept_ids=[("episodeA",i) for i in range(16)]
assert concept_ids[0] != ("episodeB",0)
check("무입력 ConceptRef의 도달성·episode identity",
      {"without_exact_candidates":17,"max_claim_reference_bank":144,"concept_is_not_NULL":True})

# 2. Address equality does not imply value-coordinate compatibility.
A = np.eye(4)
old_value, new_value = np.eye(4), np.eye(4)*2
def epoch(a,value):
    return hashlib.sha256(a.tobytes()+value.tobytes()).hexdigest()
old_epoch,new_epoch=epoch(A,old_value),epoch(A,new_value)
assert old_epoch != new_epoch
records=[{"epoch":old_epoch,"mu":np.array([.2,.8]),"v":np.array([2.,3.])}]
active=[r for r in records if r["epoch"]==new_epoch]
assert active == []
check("체크포인트의 value 좌표계·cache epoch",
      {"same_anchor_different_epoch":True,"old_contributions_excluded":True})

# 3. CAS and canonical dedup must be inside a transaction.
class Store:
    def __init__(self):
        self.version=0;self.records=set();self.mass=np.zeros(2);self.lock=Lock()
    def write(self,expected_version,key):
        with self.lock:
            if expected_version!=self.version:return "VERSION_CONFLICT"
            if key in self.records:return "DUPLICATE"
            self.records.add(key);self.mass += np.array([.2,.8]);self.version += 1
            return "CREATED"
store=Store();barrier=Barrier(2)
def client(_):
    expected=store.version
    barrier.wait(timeout=5)
    return store.write(expected,("A","GT","B","positive","t","scope"))
with ThreadPoolExecutor(max_workers=2) as pool:
    responses=list(pool.map(client,[0,1]))
assert sorted(responses)==["CREATED","VERSION_CONFLICT"]
assert store.write(store.version,("A","GT","B","positive","t","scope"))=="DUPLICATE"
assert len(store.records)==1 and np.allclose(store.mass,[.2,.8])
check("동시 Writer의 CAS·중복 억제",
      {"created_records":1,"conflict_detected":True,"retry_does_not_double_mass":True})

# 4. Fallback is subject to the same byte and UTF-8 limits as generation.
byte_limit=10
prefix=b"x:"
copy="Cafe\u0301".encode("utf8")
assert len(prefix+copy)==8
next_span="😀".encode("utf8")
assert len(prefix+copy+next_span)>byte_limit
def bounded_status(limit):
    status=b"UNRESOLVED"
    return status if len(status)<=limit else b""
assert bounded_status(10).decode("utf8")=="UNRESOLVED"
assert bounded_status(0)==b""
assert (prefix+copy).decode("utf8")=="x:Cafe\u0301"
check("copy·fallback 예산·UTF-8 경계",
      {"copy_not_partially_emitted":True,"fallback_bytes":10,"zero_budget_empty_text":True})

# 5. A first output action needs a start state; tensor exports must be copies.
start=np.ones(1216);mode=np.zeros(1216);style=np.zeros(1216)
u0=start+mode+style
assert u0.shape==(1216,) and (np.zeros((3,1216))@u0).shape==(3,)
sealed=np.arange(16,dtype="<f4").tobytes()
original_hash=hashlib.sha256(sealed).digest()
exported=np.frombuffer(sealed,dtype="<f4").copy()
exported[0]=999
assert hashlib.sha256(sealed).digest()==original_hash
assert np.frombuffer(sealed,dtype="<f4")[0]==0
check("최초 출력 상태·Commit reader 복사",
      {"first_action_logits":3,"mutable_export_does_not_alias_commit":True})

# 6. Dense neural projections and pair MLP costs cannot be omitted.
m,p,d,f=16,97,512,1408
old=4*(m*m*d+m*p*d+m*d*f)
projection=4*(6*m+2*p)*d*d
relation=m*m*(3088*512+512*9+1040*256)
assert projection>0 and relation>0 and old+projection+relation>old
assert 16+160+144==320
check("투영·관계 MLP 비용과 Claim context",
      {"previous_terms":old,"additional_projection_MACs":projection,
       "pair_MLP_MACs":relation,"claim_context_max":320})

report={"date":"2026-10-03","passed":len(checks),"failed":0,
        "scope":"Supplementary schema/transaction/numeric prototypes; no NN semantic performance test",
        "checks":checks}
assert report["passed"]==6
(ROOT/"final_review_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
~~~
