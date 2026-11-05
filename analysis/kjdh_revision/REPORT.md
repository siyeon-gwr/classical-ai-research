# KJDH-2026-010 심사 대응 분석 보고서

- 생성일: 2026-11-05 (UTC)
- 성격: **읽기 전용 분석.** 운영 색인(`nosa_units.json`·벡터 DB)·분절/생성 파이프라인 코드를 일절 수정하지 않음.
- 대상 저장소 및 상태:
  | 저장소 | HEAD 커밋 | 비고 |
  |---|---|---|
  | `siyeon-gwr/askmind-api` | `5eccd44a27d2781781d6b5dbd7d3f7ef32a8c05d` | 노사 운영 시스템(분절·생성·색인 코드). shallow→depth≈965로 심화 |
  | `siyeon-gwr/classical-ai-research` | `22a1c5b95691a275abbbcb392ebbbc8c62e367fd` | 현재 작업 디렉터리·학술 재현 repo. 본 산출물 저장 위치 |
- 산출물 디렉터리: `classical-ai-research/analysis/kjdh_revision/`
- 숫자·해시는 모두 실제 파일에서 계산. 미확인/미접근 항목은 **없음**으로 표기(추정 없음).

---

## 태스크 1 — git tag `kjdh-2026-010-submitted` (코드 변경 없음)

두 저장소에 **로컬** 주석 태그 생성(push 미실행 — 별도 확인 대기):

| 저장소 | 태그 | 가리키는 커밋 |
|---|---|---|
| askmind-api | `kjdh-2026-010-submitted` | `5eccd44a27d2781781d6b5dbd7d3f7ef32a8c05d` |
| classical-ai-research | `kjdh-2026-010-submitted` | `22a1c5b95691a275abbbcb392ebbbc8c62e367fd` |

> 태그는 커밋을 가리키는 포인터로 코드 변경이 아님. 원격 반영(push)은 되돌리기 번거로운 외부 동작이라 사용자 확인 후 수행.

---

## 태스크 2 — 파일 매니페스트 (경로·수정일·크기·해시)

수정일 = 해당 파일을 마지막으로 변경한 git 커밋의 committer 날짜(ISO). 파일시스템 mtime은 클론 시각이라 무의미하므로 사용하지 않음. sha256은 실제 파일 내용 기준(아래 표는 12자 prefix, 전체는 `task2_file_manifest.tsv`).

| 항목 | 경로(askmind-api) | git 수정일 | 크기(B) | sha256(12) | 최종커밋 |
|---|---|---|---|---|---|
| 제출본 색인 nosa_units.json | scripts/nosa/nosa_units.json | **없음** | **없음** | **없음** | **없음** |
| 운영 벡터DB 적재 색인(Pinecone) | (파일 아님·운영 DB) | **없음** | **없음** | **없음** | **없음** |
| 분절 스크립트 2_segment.py | scripts/nosa/2_segment.py | 2026-07-13T13:46:01+09:00 | 8130 | a3b1e359ebeb | c6bc0a1 |
| 원문 코퍼스 nosa_corpus.json | scripts/nosa/nosa_corpus.json | 2026-07-12T23:49:47+09:00 | 4584771 | fa5773a4305d | 3907b6f |
| 생성지시문 scholars_config.py (태도규칙 cite_guard_rules 포함) | routers/scholars_config.py | 2026-08-11T11:50:50+09:00 | 24589 | 0ca1a719a6e9 | 2dfb874 |
| 생성지시문 base_scholar.py (SYSTEM 프롬프트 조립) | routers/base_scholar.py | 2026-08-11T11:50:50+09:00 | 13532 | 008bd2690dd3 | 2dfb874 |
| 교차보류 스크립트 eval_loo.py | scripts/nosa/eval_loo.py | 2026-07-30T16:37:44+09:00 | 16198 | db8eeae9aeb3 | d3673ac |
| 교차보류 로그 loo_run_*.txt | scripts/nosa/loo_run_*.txt | **없음** | **없음** | **없음** | **없음** |
| 표지 집계 스크립트 count_quotes.py | scripts/nosa/count_quotes.py | 2026-07-30T11:31:18+09:00 | 3082 | c9fc99c560c8 | 3e59da6 |
| sample_unattributed.py | scripts/nosa/sample_unattributed.py | **없음** | **없음** | **없음** | **없음** |
| 원문 수집 crawl_nosa.py | scripts/nosa/crawl_nosa.py | 2026-07-12T23:49:47+09:00 | 6000 | 209262df699e | 3907b6f |
| 원문 수집 crawl_all.py | scripts/nosa/crawl_all.py | 2026-07-12T23:49:47+09:00 | 5685 | 524342325658 | 3907b6f |
| 목차 스캔 scan_toc.py | scripts/nosa/scan_toc.py | 2026-07-12T23:49:47+09:00 | 3624 | ef1a50d7f5af | 3907b6f |

**없음 항목 사유**
- `nosa_units.json`: git에 **한 번도 커밋된 적 없음**(`git log --all` 0건). `2_segment.py` 설계상 결과를 R2/Render 임시디스크에만 저장 → 제출본 파일이 저장소에 존재하지 않음.
- 운영 벡터 DB 색인: Pinecone. `PINECONE_API_KEY` 미설정·네트워크 범위 밖 → 접근 불가.
- `loo_run_*.txt`, `sample_unattributed.py`: 저장소 어디에도 없음(이력에도 없음).

---

## 태스크 3 — 색인 버전 이력 재구성

색인 파일 자체(`nosa_units.json`)는 버전관리되지 않으므로 **git으로 색인 파일의 변경 이력을 직접 재구성할 수 없음**. 대신 색인을 생성하는 **파이프라인 코드·원문 코퍼스**의 커밋 이력으로 간접 재구성함. (원시 로그: `task3_pipeline_gitlog.txt`)

**노사 파이프라인/코퍼스/업로드 커밋 (전량)**

| 시각(ISO) | 커밋 | 내용 |
|---|---|---|
| 2026-07-12T23:49:47+09:00 | 3907b6f | Add files via upload (코퍼스·crawl·scan 등 최초) |
| 2026-07-13T10:08:24+09:00 | 3786c67 | Add files via upload |
| 2026-07-13T11:20:13+09:00 | 67ed1a3 | Create 2b_resegment_fallback.py |
| 2026-07-13T12:42:35+09:00 | ae6ea13 | Update 2_segment.py |
| 2026-07-13T12:42:52+09:00 | 625c99c | Update 3_upload.py |
| 2026-07-13T13:46:01+09:00 | c6bc0a1 | Update 2_segment.py (분절 스크립트 최종) |
| 2026-07-13T13:46:18+09:00 | 301ba5d | Update 3_upload.py (업로드 스크립트 최종) |
| 2026-07-13T18:55:48+09:00 | 24697c7 | Create 2c_fix_long.py |
| 2026-07-30T11:31:18+09:00 | 3e59da6 | nosa 인용 메타 조사 스크립트 2종(읽기 전용) |
| 2026-07-30T16:37:44+09:00 | d3673ac | LOO 분절 평가 스크립트 eval_loo.py |

**2026-08-01 전후 (07-25~08-10)**: 해당 창의 `scripts/nosa` 커밋은 07-30의 2건(`3e59da6`, `d3673ac`)뿐이며 **모두 읽기 전용 분석 스크립트**임. **분절·업로드·코퍼스 변경 없음** → 이 시점에 색인을 재생성할 코드/코퍼스 변화는 git상 **없음**.

**2026-09 (9월 재실행)**: `scripts/nosa` 9월 커밋 **없음**. → "9월 재실행"에 해당하는 **파이프라인/코퍼스/스크립트 변경의 git 증거 없음**.

**생성지시문(태도규칙)**: `routers/scholars_config.py` 는 2026-08-11 `2dfb874` 에서 변경(노사 전용 인용귀속·태도 규칙 cite_guard_rules 추가). 이는 색인이 아니라 **응답 생성 프롬프트** 변경이며 8/1 이후임.

**중요 한계(추정 아님, 사실 적시)**: 색인 생성·Pinecone 업로드 실행은 Render 셸에서 수동 실행되고 결과가 R2/임시디스크에만 남으므로, **코드 커밋 없이 이뤄진 재실행(예: 9월 운영 재색인)은 git에 흔적이 남지 않음.** 즉 "git 증거 없음"은 "재실행이 없었다"의 증명이 아님. 파일 mtime 기반 재구성도 색인 파일이 저장소에 없어 불가. 확정하려면 R2 객체의 타임스탬프 또는 Pinecone 색인 메타가 필요(태스크 4 참조).

---

## 태스크 4 — 제출본 ↔ 운영 DB 색인 diff (단위 ID·원문)

**수행 불가 — 양측 모두 접근 불가 → 없음.**

| 비교 대상 | 접근 여부 | 근거 |
|---|---|---|
| 제출본 `nosa_units.json` | 접근 불가 | git 0커밋, 작업트리에도 없음(R2/임시디스크에만 존재) |
| 운영 벡터 DB(Pinecone) 적재 색인 | 접근 불가 | `PINECONE_API_KEY` 미설정·네트워크 범위 밖 |

diff를 수행하려면 다음 중 하나가 필요(요청 시 즉시 진행):
1. 제출 시점 `nosa_units.json` 파일 제공(붙여넣기/커밋), **그리고**
2. 운영 Pinecone 색인 덤프 또는 접근 자격증명(네임스페이스 `nosa`).

둘을 받으면 **단위 ID 집합 diff + 동일 ID의 원문(해시) 일치 여부**를 계산해 제출본과 운영본의 동일성을 판정함.

---

## 생성 산출물 (모두 `classical-ai-research/analysis/kjdh_revision/`)

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 본 보고서 |
| `task2_manifest.py` | 태스크 2 매니페스트 생성 스크립트(재현용) |
| `task2_file_manifest.tsv` | 태스크 2 전체 매니페스트(전체 sha256 포함) |
| `task3_pipeline_gitlog.txt` | 태스크 3 원시 git 이력 증거 |

## 미해결/대기

- 태그 **push 미실행**(두 저장소 로컬 태그만). 원격 반영 원하시면 확인 바람.
- 태스크 3 색인 재실행(9월) 확정 및 태스크 4 diff: 제출본 파일 + Pinecone 접근 필요.
- 본 산출물의 git 커밋 여부는 미결정(현재 작업트리 untracked).
