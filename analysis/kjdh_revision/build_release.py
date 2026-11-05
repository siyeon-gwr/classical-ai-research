#!/usr/bin/env python3
"""
KJDH-2026-010 공개 저장소 패키지 빌더 (읽기 전용 소스 → release/ 생성).
- gold 파일의 korean(번역문) 필드는 '번역문 미재배포 방침'에 따라 제거.
- 없는 파일(sample_unattributed.py, loo_run_*.txt, 제출본 nosa_units.json)은 자리/NOTE만.
"""
import json, os, re, shutil

ASK = "/home/user/askmind-api"
ANA = "/home/user/classical-ai-research/analysis/kjdh_revision"
REL = f"{ANA}/release"

def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# clean tree
if os.path.exists(REL):
    shutil.rmtree(REL)
for d in ["data/gold", "scripts", "prompts", "logs"]:
    os.makedirs(f"{REL}/{d}", exist_ok=True)

report = []

# ---- data/nosa_units.json : placeholder (사용자가 제출본으로 교체) ----
w(f"{REL}/data/nosa_units.json", json.dumps({
    "__PLACEHOLDER__": True,
    "note": "심사용 zip의 제출본 nosa_units.json(번역문 없음)으로 이 파일을 교체하세요.",
}, ensure_ascii=False, indent=1) + "\n")
report.append("data/nosa_units.json : 플레이스홀더 생성(제출본 교체 대기)")

# ---- data/gold/ : korean(번역문) 제거 후 복사 ----
for fn in ["itongseol_units.json", "answer_kwonsinwon_units.json"]:
    src = json.load(open(f"{ASK}/scripts/nosa/{fn}", encoding="utf-8"))
    removed = 0
    for u in src.get("units", []):
        if "korean" in u:
            del u["korean"]; removed += 1
    src["_redistribution_note"] = "korean(번역문) 필드는 한국고전번역원 번역 미재배포 방침에 따라 제거됨. hanmun(표점본)과 메타데이터만 포함."
    w(f"{REL}/data/gold/{fn}", json.dumps(src, ensure_ascii=False, indent=1) + "\n")
    report.append(f"data/gold/{fn} : korean 필드 {removed}개 제거 후 포함")

# ---- scripts/ ----
for fn in ["crawl_nosa.py", "2_segment.py", "count_quotes.py", "eval_loo.py", "r2util.py"]:
    s = f"{ASK}/scripts/nosa/{fn}"
    if os.path.exists(s):
        shutil.copy2(s, f"{REL}/scripts/{fn}"); report.append(f"scripts/{fn} : 포함")
    else:
        report.append(f"scripts/{fn} : 없음(원본 부재)")
if not os.path.exists(f"{ASK}/scripts/nosa/sample_unattributed.py"):
    w(f"{REL}/scripts/NOTE_missing.txt",
      "sample_unattributed.py : 저장소에 존재하지 않아 포함하지 못함(없음).\n")
    report.append("scripts/sample_unattributed.py : 없음 → NOTE_missing.txt")

# ---- prompts/ ----
# 분절 지시문 (2_segment.py SYSTEM 리터럴 추출)
seg = open(f"{ASK}/scripts/nosa/2_segment.py", encoding="utf-8").read()
m = re.search(r'SYSTEM\s*=\s*"""(.*?)"""', seg, re.S)
w(f"{REL}/prompts/segmentation_SYSTEM.txt", (m.group(1).strip() if m else "(추출 실패)") + "\n")
shutil.copy2(f"{ASK}/scripts/nosa/genre_policy.json", f"{REL}/prompts/genre_policy.json")
# 생성 지시문 8/11 이전·이후 원문 (앞서 analysis/kjdh_revision/prompts 에서 추출해 둔 것)
for fn in ["nosa_block.before-256b666.txt", "nosa_block.after-2dfb874.txt",
           "base_scholar_system_prompt.before-256b666.txt",
           "base_scholar_system_prompt.after-2dfb874.txt",
           "scholars_config.py.2dfb874.diff", "base_scholar.py.2dfb874.diff"]:
    s = f"{ANA}/prompts/{fn}"
    if os.path.exists(s):
        shutil.copy2(s, f"{REL}/prompts/{fn}")
report.append("prompts/ : segmentation_SYSTEM.txt, genre_policy.json, 생성지시문 before/after + diff 포함")

# ---- logs/ : loo 로그 부재 ----
w(f"{REL}/logs/NOTE_missing.txt",
  "loo_run_itongseol.txt, loo_run_kwonsinwon.txt : 저장소에 로그 파일이 없음(없음).\n"
  "재생성: scripts/eval_loo.py 를 OPENAI_API_KEY 환경에서 각 gold에 대해 실행.\n")
report.append("logs/ : loo_run_*.txt 없음 → NOTE_missing.txt")

print("BUILD STEPS:")
for r in report:
    print("  -", r)
