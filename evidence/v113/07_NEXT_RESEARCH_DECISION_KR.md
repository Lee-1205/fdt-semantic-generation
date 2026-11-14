2026-11-14. ROOT 직접/에이전트0. VERIFIED=코드·실행·독립재계산, OBSERVED=저장측정, INFERRED=해석, PROPOSED=미실행계획.

# 07. 다음 연구 판단

**VERIFIED / OBSERVED** 단순반복이아닌두수정·유한학습·전체평가를수행했다. 짝차이수식으로조건반응을키웠고고정Dcontent피드백으로TRAIN언어내용이개선됐다. TRAIN정상관계core34/80+명확한typo1, 역전짝bothcorrect8/32는이전보다진전이다. 하지만DEV긍정은아직해결되지않았고NLL도악화해전체품질개선이나일반화완료를선언하지않는다.

**INFERRED** 현재다음병목은134행/27답문에대한조건부적합에서새개체·관계조합과표현으로넘어가는일반화다. dev_pair의정답positive16개전부학습에없고12개에서학습답문을대신출력했다. 표현·위치패널은학습에있는fact/answer도틀리므로새답문조합뿐아니라배치·질문읽기도다뤄야한다. 이근거로같은134행을더오래반복하는것을다음주목적으로삼지않는다. 의미기억의내용이부재하거나디퓨전이불가능하다는판정도하지않는다.

**PROPOSED 다음별도목적**

1. 원래TRAIN분할의허용된fact/group으로새source-only문장을작성하고, 관계방향/주체/목적어/질문순서/문맥위치를균형화한다. 관측DEV의실제행·heldpair/group·family4/5를TRAIN으로옮기지않는다. 미관측답문구성은별도사전등록분할로둔다. 기존384중탈락250행을무단복구하지않는다.
2. 사람지시대로새TRAIN후보도고정E→D왕복검사를먼저수행해통과한행과실패행을별도manifest로보존한다. 이선별은codec검증이며semanticgeneralization증거가아니고선별bias를보고한다. 정답E(Y)는loss-onlyfile, 조건packet은source-only로계속분리한다.
3. 셋업전에성분·literal/family/pairgroup중복·label균형·실제노출을계산한다. 새실행은별도코드/정확복구/고정time·call·output상한/C100GiB종료보호를검증한ONEtrial만쓴다. 지금아직새data/runner/학습은없으며본CLOSED예산을재사용하지않는다.
4. 같은질문·서로다른합법적K_C의기억필수episode는기본Transformation일반화의증거가생긴뒤별도실험으로한다. K_Pcertifiedwriter/장기통합·미래Exactcopy/Composition·장문사고는성적에섞지않는다. 입력·출력가중치는여전히고정한다.

기존V111/V112/V113fullstates·Adam/scaler/RNG·실패·raw는모두보존했다. V113의훈련성과만으로전역기본모델이나외부발표성적을자동변경하지않는다. 타이머재개0이다.

실제가능한안전검증명령(프로젝트루트):
```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m pytest -q -p no:cacheprovider tests/test_answer_codec_fidelity_v113.py
Get-Content -LiteralPath 'runs/answer_codec_fidelity_20261114_v113/root_result.json' -Raw
Get-FileHash -LiteralPath 'runs/answer_codec_fidelity_20261114_v113/checkpoints/state_0268.pt' -Algorithm SHA256
```
닫힌owner/worker명령을다시실행하지않는다. 다음자료분기용미구현명령을만들어제출하지않는다.
