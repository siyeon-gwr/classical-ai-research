# 주문 4 — 지시문 · 검색 설정 문서화 (KJDH-2026-010)

- 대상 저장소: `siyeon-gwr/askmind-api` @ `5eccd44a27d2781781d6b5dbd7d3f7ef32a8c05d` (태그 `kjdh-2026-010-revision-base`)
- 성격: 읽기 전용. 코드/색인 미수정.
- 관련 산출물: 본 문서 + `prompts/` (지시문 before/after 원문·diff)

---

## A. 분절 지시문 — `scripts/nosa/2_segment.py` (최종 커밋 `c6bc0a1`, 2026-07-13)

| 구성요소 | 위치 | 내용 |
|---|---|---|
| 시스템 프롬프트 `SYSTEM` | `2_segment.py:76-88` | 발화단위 정의(완결·최소성), 인용 판정 규칙, JSON 출력 스키마(`seq/quoted_speaker/function/hanmun/warning`) |
| few-shot 조립 `build_fewshot()` | `2_segment.py:60-73` | 골드 파일에서 입력(hanmun 연결)·출력(JSON) 예시 생성 |
| few-shot 골드 소스 | `2_segment.py:28-31` → `itongseol_units.json`(理通說), `answer_kwonsinwon_units.json`(答權信元) | 예시 본문은 이 두 JSON에 있음(스크립트에 하드코딩 아님) |
| user 프롬프트 조립 `make_user_prompt()` | `2_segment.py:91-97` | `[예시 입력—제목]…[예시 출력]…` + `[분할 대상] 저자/문체/수신자/원문` |
| LLM 호출 | `2_segment.py:100-110` | `model="gpt-4o"`, `temperature=0` |
| 인용 표지 열거 | `2_segment.py:83` | `來喩·示喩·來書·盛敎曰·兄每言 등`(비망라, "등"+재진술 포괄) |

## B. 생성 지시문 — `routers/base_scholar.py` + `routers/scholars_config.py`

| 구성요소 | 위치 | 내용 |
|---|---|---|
| 시스템 프롬프트 템플릿 | `base_scholar.py` `system_prompt = f"""…"""` (생성 함수 `generate_answer`) | 핵심원칙(원문만 근거), 출처 레이어 규칙 1-4, `{cite_rule}`·`{cite_guard_rules}` 주입, 답변 원칙/레이어/금지 |
| 공통 인용 규칙 `cite_rule` | `base_scholar.py` (규칙 5) | `(cite_guard and has_quote)`일 때만 — "남의 말을 옮긴 구간, 본인 주장으로 진술 말라" |
| 노사 전용 태도규칙 `cite_guard_rules` | `scholars_config.py` nosa 블록 | 인용 귀속(「○○曰」·인용구간)·논평 태도(동의/반대/유보 표지)·서술순서(자기견해→인용+화자→논평). `base_scholar.py`가 config 키 있을 때만 주입 |

### 2dfb874 (2026-08-11) 전후 변경 — 원문·diff는 `prompts/`
- **이전 버전**: `prompts/raw/{scholars_config.py,base_scholar.py}.before-256b666.py` (부모 커밋 `256b666`)
- **이후 버전**: `prompts/raw/{…}.after-2dfb874.py`
- **지시문 추출본**: `prompts/nosa_block.{before,after}.txt`, `prompts/base_scholar_system_prompt.{before,after}.txt`
- **diff**: `prompts/scholars_config.py.2dfb874.diff`(+15), `prompts/base_scholar.py.2dfb874.diff`(+7)
- 변경 요지: 2dfb874에서 노사 블록에 `cite_guard_rules`(인용귀속·태도·순서 세칙) 신설, `base_scholar.py`에 그 주입부 추가. **이전 버전엔 두 항목 모두 없음**(삭제 0, 순수 +22). 다른 10개 학자 config는 해당 키가 없어 무영향.

## C. 검색 설정

### C-1. 노사 검색 파라미터 — `routers/scholars_config.py` nosa 블록 (라인 ~378-431 @ 5eccd44)
| 키 | 값 |
|---|---|
| `namespace` / `history_namespace` | `nosa` / `nosa` |
| `default_prefix` | `蘆沙 奇正鎭 性理說` (질의<15자 시 접두) |
| `temperature` | `0.35` |
| `threshold_direct / similar / keyword` | `0.30 / 0.22 / 0.16` |
| `history_min` | `0.18` |
| `keywords` | 理·氣·太極·理通·理一·分殊·主宰·所以然·動靜·心·性 … (가산점용) |
| `expansion_map` | 한국어→한문 표제어 확장(예 "이통"→"理通 理一分殊 理之妙") |

### C-2. 검색 코드 — `routers/base_scholar.py` `search_chunks()` (라인 83-135)
- 질의 정규화(조사·기호 제거) → `expansion_map` 적용 → 짧으면 `default_prefix` 접두.
- 임베딩: `services/embedding.py:get_embedding` = **text-embedding-3-small**(1536차원, dimensions override 없음).
- Pinecone `index.query`: main 네임스페이스 `top_k*3`, history 네임스페이스 `top_k=5`, `include_metadata=True`.
- 반환 매치를 dict로 변환 후 `quoted_speaker` 등 화자 귀속 필드 채움(노사 네임스페이스만 값 존재).

## D. 수신자(addressee) 추출 정규식 — 위치 (주문 3)

**`scripts/nosa/2_segment.py` `extract_addressee()` 단 한 곳.** 다른 스크립트(`2b_resegment_fallback.py:62`, `2c_fix_long.py:66`)는 `seg.extract_addressee()`를 **재사용**만 하고, `3_upload.py:55-56`은 결과 `addressee` 필드를 읽기만 함. 저장소 내 수신자 추출 정규식 리터럴은 아래 2줄뿐:

```python
# scripts/nosa/2_segment.py
49  def extract_addressee(title):
50      m = re.search(r"[答與上寄]([一-鿿]{2,4})", title)   # 答/與/上/寄 + 한자 2~4
51      if m:
52          return m.group(1)
53      m2 = re.search(r"〔[答與上寄]([^〕]+)〕", title)            # 〔…〕 묶음 제목 폴백
54      if m2:
55          nm = re.search(r"[一-鿿]{2,4}$", m2.group(1))   # 묶음 말미 인명
56          return nm.group(0) if nm else ""
57      return ""
```

---

## 산출물 (`classical-ai-research/analysis/kjdh_revision/`)
| 파일 | 내용 |
|---|---|
| `order4_instructions_and_search.md` | 본 문서 |
| `prompts/raw/*.{before-256b666,after-2dfb874}.py` | 두 파일의 2dfb874 전/후 **원문 전체** |
| `prompts/nosa_block.{before,after}.txt` | 노사 config 블록(지시문) 추출본 |
| `prompts/base_scholar_system_prompt.{before,after}.txt` | 시스템 프롬프트 조립부 추출본 |
| `prompts/*.2dfb874.diff` | 두 파일의 전/후 통합 diff |
