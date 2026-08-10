#!/usr/bin/env python3
"""
Aggregate 問目 (mummok) works in a Nosajip segmentation-result JSON.

Computes, over the input segments:
    1. number of distinct works (title) whose title contains "問目"
    2. number of utterance-units belonging to those works
    3. of those, units whose speaker field is non-empty
    4. of the attributed units, how many match the title's recipient

Recipient is extracted from the title pattern "答<name>問目" (the name
is the addressee's courtesy name, 字). A speaker counts as a match if
it equals that courtesy name OR its mapped personal name (see
RECIPIENT_ALIASES). Titles that do not match the pattern yield no
recipient, and unmapped names are NOT expanded — missing values are
left missing, never guessed.

Usage:
    python3 count_mummok.py [path/to/segmentation.json]

Defaults to the in-repo sample corpus if no path is given. Note the
sample uses the field name `citation_speaker`; the full segmentation
result uses `quoted_speaker`. Both are tried, in that order of the
SPEAKER_FIELDS list, and the one actually present is reported.
"""

import json
import re
import sys

# --- Configuration ----------------------------------------------------

DEFAULT_PATH = "nosa-rag/sample_corpus/nosa_sample.json"

# Speaker field candidates, tried in order; first present one is used.
SPEAKER_FIELDS = ["quoted_speaker", "citation_speaker"]

# Title -> recipient extraction: 答<字>問目  (e.g. 答鄭季方問目 -> 鄭季方)
RECIPIENT_RE = re.compile(r"答(.+?)問目")

# Courtesy name (字, as written in titles) -> personal name (名).
# Only the mappings explicitly provided; do not add guesses.
RECIPIENT_ALIASES = {
    "鄭季方": "鄭義林",
    "金景範": "金錫龜",
    "鄭厚允": "鄭載圭",
}


# --- Helpers ----------------------------------------------------------

def load_segments(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"] if isinstance(data, dict) else data


def pick_speaker_field(segments):
    """Return the first SPEAKER_FIELDS key that appears on any segment."""
    for field in SPEAKER_FIELDS:
        if any(field in s for s in segments):
            return field
    return None


def extract_recipient(title):
    """Return the addressee 字 from '答<字>問目', or None if no match."""
    if not title:
        return None
    m = RECIPIENT_RE.search(title)
    return m.group(1) if m else None


def speaker_matches_recipient(speaker, recipient):
    """True if speaker equals the recipient 字 or its mapped 名."""
    if not speaker or not recipient:
        return False
    accepted = {recipient}
    if recipient in RECIPIENT_ALIASES:
        accepted.add(RECIPIENT_ALIASES[recipient])
    return speaker in accepted


# --- Aggregation ------------------------------------------------------

def aggregate(segments, speaker_field):
    mummok = [s for s in segments if "問目" in str(s.get("title", ""))]

    distinct_titles = sorted({s.get("title", "") for s in mummok})

    attributed = [
        s for s in mummok
        if speaker_field and str(s.get(speaker_field, "") or "").strip()
    ]

    matched = [
        s for s in attributed
        if speaker_matches_recipient(
            str(s.get(speaker_field, "") or "").strip(),
            extract_recipient(s.get("title", "")),
        )
    ]

    return {
        "distinct_mummok_titles": len(distinct_titles),
        "titles": distinct_titles,
        "units_in_mummok": len(mummok),
        "units_with_speaker": len(attributed),
        "units_speaker_matches_recipient": len(matched),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    segments = load_segments(path)
    speaker_field = pick_speaker_field(segments)

    r = aggregate(segments, speaker_field)

    print(f"input file            : {path}")
    print(f"total segments        : {len(segments)}")
    print(f"speaker field used    : {speaker_field or '(none present)'}")
    print("-" * 52)
    print(f"1. 問目 works (title) : {r['distinct_mummok_titles']}")
    print(f"2. units in those     : {r['units_in_mummok']}")
    print(f"3. with speaker set   : {r['units_with_speaker']}")
    print(f"4. speaker == recipient: {r['units_speaker_matches_recipient']}")
    if r["titles"]:
        print("-" * 52)
        print("problem titles:")
        for t in r["titles"]:
            print(f"  - {t}  (recipient: {extract_recipient(t) or '?'})")


if __name__ == "__main__":
    main()
