#!/usr/bin/env python3
"""
KJDH-2026-010 revision — Task 2 file manifest.
Read-only. Computes path / git-commit-date / size / sha256 / last-commit
hash for each target file. Missing files are reported as 없음.
Outputs TSV to analysis/kjdh_revision/task2_file_manifest.tsv.
"""
import hashlib, os, subprocess, sys

ASK = "/home/user/askmind-api"
CLS = "/home/user/classical-ai-research"
OUT = "/home/user/classical-ai-research/analysis/kjdh_revision/task2_file_manifest.tsv"

# (label, repo_root, relpath)  — repo_root None = not a file (e.g. live DB)
ENTRIES = [
    ("제출본 색인 nosa_units.json",        ASK, "scripts/nosa/nosa_units.json"),
    ("운영 벡터DB 적재 색인(Pinecone)",     None, None),
    ("분절 스크립트 2_segment.py",          ASK, "scripts/nosa/2_segment.py"),
    ("원문 코퍼스 nosa_corpus.json",        ASK, "scripts/nosa/nosa_corpus.json"),
    ("생성지시문: scholars_config.py",      ASK, "routers/scholars_config.py"),
    ("생성지시문: base_scholar.py",         ASK, "routers/base_scholar.py"),
    ("교차보류 스크립트 eval_loo.py",       ASK, "scripts/nosa/eval_loo.py"),
    ("교차보류 로그 loo_run_*.txt",         ASK, "scripts/nosa/loo_run_*.txt"),
    ("표지 집계 스크립트 count_quotes.py",  ASK, "scripts/nosa/count_quotes.py"),
    ("sample_unattributed.py",              ASK, "scripts/nosa/sample_unattributed.py"),
    ("원문 수집 crawl_nosa.py",             ASK, "scripts/nosa/crawl_nosa.py"),
    ("원문 수집 crawl_all.py",              ASK, "scripts/nosa/crawl_all.py"),
    ("목차 스캔 scan_toc.py",               ASK, "scripts/nosa/scan_toc.py"),
]

def git(root, *args):
    try:
        return subprocess.check_output(["git", "-C", root, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

rows = []
for label, root, rel in ENTRIES:
    if root is None:
        rows.append([label, "(파일 아님 — 운영 DB)", "없음", "없음", "없음", "없음"])
        continue
    # glob support (for loo_run_*.txt)
    if "*" in rel:
        import glob
        matches = glob.glob(os.path.join(root, rel))
        if not matches:
            rows.append([label, rel, "없음", "없음", "없음", "없음"])
            continue
        rel_list = [os.path.relpath(m, root) for m in matches]
    else:
        rel_list = [rel]
    for r in rel_list:
        full = os.path.join(root, r)
        if not os.path.exists(full):
            rows.append([label, r, "없음", "없음", "없음", "없음"])
            continue
        size = os.path.getsize(full)
        digest = sha256(full)
        meta = git(root, "log", "-1", "--format=%cI|%H", "--", r)
        cdate, chash = (meta.split("|", 1) + ["", ""])[:2] if meta else ("", "")
        rows.append([label, r, cdate or "(이력없음)", str(size), digest, chash or "(이력없음)"])

hdr = ["항목", "경로(repo상대)", "git_수정일(committer ISO)", "크기(bytes)", "sha256", "최종커밋해시"]
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\t".join(hdr) + "\n")
    for row in rows:
        f.write("\t".join(row) + "\n")

# print a compact markdown table (sha256 shortened for console)
print("| 항목 | 경로 | git 수정일 | 크기(B) | sha256(12) | 커밋 |")
print("|---|---|---|---|---|---|")
for label, path, cdate, size, digest, chash in rows:
    sd = digest[:12] if digest not in ("없음",) else "없음"
    ch = chash[:7] if chash not in ("없음", "(이력없음)") else chash
    print(f"| {label} | {path} | {cdate} | {size} | {sd} | {ch} |")
print(f"\nTSV -> {OUT}")
