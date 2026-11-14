# FDT Exact 개발용 생성 경로

`work/astra_exact_development_generation.py`에서 저장된 Base와 Exact reader를 함께 불러와 응답을 생성할 수 있다. 기존 생성 스크립트는 변경하지 않았다.

입력 JSON은 `records`와 `query` 두 필드만 받는다. 정답·source label은 입력받지 않는다.

```json
{
  "records": [
    {"name": "item_1", "value": "ZX123456"},
    {"name": "item_2", "value": "LV876543"}
  ],
  "query": "Copy the value of item_1 exactly. Stop after that value."
}
```

프로젝트 폴더에서 다음 형태로 실행한다. 입력 파일은 별도로 저장하며 출력 파일이 이미 있으면 덮어쓰지 않는다.

```powershell
& '<USER_HOME>\AppData\Local\Programs\Python\Python310\python.exe' -B -X utf8 work/astra_exact_development_generation.py --input request.json --output response.json
```

- 등록 record 전체와 원래 query를 모델에 전달해 복사를 한 번 승인한다. 이후 생성된 문장으로 다시 승인하지 않는다.
- 승인되면 기존 cursor가 선택한 payload와 EOS를 출력한다. 거절되면 같은 원래 prompt에서 실제 Base 응답을 생성한다.
- `--exact-off`는 같은 등록 prompt의 Base 출력 비교용이다. 자연어 문장 이어쓰기와 등록 문서의 prefix 형식은 별도다.
- 기본 생성 한도는 32 tokens. 종료 이유, 사용 경로, 승인 정보, 원문 출력과 token trace가 결과 JSON에 기록된다.
- checkpoint·adapter·tokenizer·기존 source SHA를 확인한다. 가중치가 바뀌면 해당 adapter의 검증도 다시 필요하다.

현재는 **개발용 기능**이다. 알려진 설명·금지 요청 오복사와 Base97의 유지 평가 실패가 남아 있다. Literal identifier 주소 선택을 포함하는 hybrid이며, 학습된 범용 source 검색이나 임의 문서 자동 추출을 의미하지 않는다. 이번 연결은 자동 production 승격이 아니다. 최신 수치는 `ASTRA_EXACT_QUERY_READER_RESULT_20260927_KR.md`와 생성 통합 보고서에서 확인한다.
