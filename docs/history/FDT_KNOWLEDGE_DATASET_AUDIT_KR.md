# FDT DIANA 지식 데이터 감사 — 2026-09-30

## 현재 판정

6개 지정 데이터셋의 **실제 split, 첫 행 스키마, 현재 저장소 SHA**를 공식 Dataset Viewer와 Hub API로 확인했다. 각 첫 문서 1개를 기존 FDT tokenizer로 정규화·분할했다. 이것은 데이터 시스템의 **소량 입력 smoke**이며, 전체 자료의 품질·사용 권한·100M 고유 토큰 확보를 검증한 결과가 아니다. 현재 어떤 문서도 optimizer 입력으로 승인하지 않았다.

| 소스 / 지정 config | 확인한 저장소 SHA | 실제 주요 필드 | 첫 문서 청크 / loss 위치 | 권리 상태 |
|---|---|---|---:|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) / sample-10BT | `87f09149...` | `text,id,dump,url,file_path,language,language_score,token_count,score,int_score` | 2 / 865 | dataset card `odc-by`; 개별 웹 문서 권리 미확인, **review_required** |
| [Structured Wikipedia](https://huggingface.co/datasets/wikimedia/structured-wikipedia) / enwiki_namespace_0 | `417c267b...` | `name,identifier,sections,abstract,references,license,url,...` | 3 / 888 | 첫 문서 `CC-BY-SA-4.0` 기록, 의무 검토 전 |
| [LibreTexts](https://huggingface.co/datasets/common-pile/libretexts_filtered) / default | `70388bca...` | `text,id,metadata{title,book_url,license,...}` | 4 / 1,591 | 첫 문서 `CC BY 4.0` 기록, 의무 검토 전 |
| [Project Gutenberg](https://huggingface.co/datasets/common-pile/project_gutenberg_filtered) / default | `3cdf6879...` | `text,id,metadata{title,language,license,...}` | 67 / 24,107 | 첫 문서 `Public Domain` 기록, 관할·원문 확인 전 |
| [arXiv](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered) / default | `033cf7f5...` | `text,id,metadata{authors,license,url,...}` | 78 / 29,666 | 첫 문서 `Public Domain` 기록, 변환·권리 검토 전 |
| [StackExchange](https://huggingface.co/datasets/common-pile/stackexchange_filtered) / default | `c0ac7373...` | `text,id,metadata{site,authors,all_licenses,license,...}` | 1 / 285 | 첫 문서 `CC BY-SA 3.0` 기록, 선택적 ≤5% |

전체 저장소 SHA, 타입, 샘플 hash/URL/토큰 계수는 [스키마 근거](../LOCAL_ASSETS.md)와 [실제 소량 검증](../LOCAL_ASSETS.md)에 저장했다. Dataset Viewer 첫 행 API는 요청 리비전을 고정하지 못했으므로 샘플 행이 위 Hub SHA에서 나왔다는 증명은 아니다. 본문에는 첫 문서만 사용했으므로 출처별 길이 분포나 실제 혼합 비율로 해석하지 않는다. LibreTexts 고정 SHA 리비전은 `datasets.load_dataset(..., streaming=True)`로 실제 한 행을 읽었다. 나머지 5개는 Dataset Viewer 실제 행으로 파서를 시험했으며 `datasets` 스트리밍 경로는 아직 미검증이다.

## 구현

- [데이터 파이프라인](WORKSPACE_ROOT/work/diana_knowledge_pipeline.py): SHA 고정 스트리밍, 6종 정규화, Wikipedia `sections` JSON 계층 파싱, 문서별 출처·라이선스 보존, 문서 단위 해시 분리, exact/normalized/near-duplicate 거절(SQLite MinHash 영속 인덱스 포함), 실제 tokenizer 기반 의미 단위 청킹, 제한적 인접 overlap의 loss masking, 인위적 경계 EOS 제외, 토큰별 혼합 부족량 보고, 가변 길이 padding.
- [설정](WORKSPACE_ROOT/work/diana_knowledge_config.json): 초안 30/25/20/15/10, StackExchange 0, 학습 비활성.
- [제한 수집기](WORKSPACE_ROOT/work/run_diana_bounded_collect.py): 출력 위치·문서 수·스캔 행 수·본문 크기를 제한하고, SQLite 중복 검사 후 권리 검토용 청크 manifest만 저장한다. `training_eligible=false`이며 실제 token IDs/학습 shard를 공개하지 않는다. 고정 리비전 LibreTexts 한 문서를 실제 스트리밍으로 수집한 [단일 문서 결과](../LOCAL_ASSETS.md)는 1청크/1,558 loss 위치/문서 권리 미검토로 기록됐다.
- [CPU 검사](WORKSPACE_ROOT/work/test_diana_knowledge_pipeline.py) 7개 통과. 실물 6개 첫 행의 tokenizer/chunk smoke 통과. 최대 512, overlap 64는 **데이터 smoke 설정**이지 모델 장문 훈련 길이 결정이 아니다.

## 아직 충족하지 못한 조건

1. SQLite MinHash 영속 인덱스를 추가했지만 아직 실제 규모에서 속도/false-negative율이 측정되지 않았고, 기존 FDT 학습·평가 자료와의 오염 검사가 필요하다. 근사 중복은 완전한 의미 중복 인증이 아니다.
2. 일반 text 소스의 장·절은 보수적 문단/Markdown 표제 휴리스틱이다. 실물 첫 행에서 Gutenberg preamble, LibreTexts 수식·표, arXiv 변환 마크업, StackExchange 질문/답 역할 미검증 플래그가 발생했다. 이를 아직 완전히 분리하지 않는다. Wikipedia 표·infobox·참고문헌은 불완전 평문화를 피하려고 현재 본문에서 제외한다.
3. 첫 행의 라이선스 문자열 기록은 사용 권한 승인과 다르다. FineWeb 개별 원문은 특히 `review_required`다. 문서별 권리, attribution, 파생물 조건을 검사한 뒤에만 `reviewed_allowed`로 바꾼다.
4. 데이터 구성 중복 노출, 실제 30/25/20/15/10 비율, 구조적 품질·언어 분포, 고유 토큰 규모는 아직 측정하지 않았다. 부족한 출처를 다른 출처로 조용히 대체하지 않는다.
5. 긴 한 문장을 token fallback으로 자를 때 의미 경계가 없다는 사실을 기록하지만 원문 문자 오프셋 복원·완전한 문장 구분은 후속 구현이 필요하다.

**판정: 선택적 소량 수집 준비 완료, 대규모 학습 자료는 미승인.**
