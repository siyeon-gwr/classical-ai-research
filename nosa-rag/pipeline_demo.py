"""
Nosa AI — Pipeline Demo

Research reproducibility version for academic use.

This standalone script demonstrates the methodology for a RAG-based
classical AI built on the collected works of Nosa Gi Jeong-jin
(蘆沙 奇正鎭, 1798-1879), focused on his letters (書 / 與) and
miscellaneous treatises (雜著) in the Nosajip (蘆沙集).

The full system is operated commercially at https://askmind.ai.
The full corpus, vector database, and production prompt engineering
are proprietary and not distributed.

This script runs independently using the OpenAI Embeddings API
and a local sample dataset. It does not connect to any proprietary
infrastructure.

License: CC-BY-NC-ND 4.0
"""

import json
import os
import numpy as np
from openai import OpenAI


# ================================================================
# Configuration
# ================================================================

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o"
SAMPLE_DATA_PATH = "./sample_corpus/nosa_sample.json"


# ================================================================
# Step 1. Load sample segments
# ================================================================

def load_segments(path: str) -> list:
    """Load segmented utterance-units with metadata.

    The Nosajip is a literary collection of letters and treatises, so
    continuous prose is pre-segmented into utterance-units under two
    criteria — completeness (完結性) and minimality (最小性) — applied
    with genre-specific handling (書: response-unit after restoring
    quotation boundaries; 與: the two criteria only; 雜著: the two
    criteria plus citation-speaker attribution).

    Each segment contains:
        - id, genre (書 / 與 / 雜著), title, recipient
        - text (original Sinographic text)
        - grounding ("직접근거" | "인용구간")
        - citation_speaker  (attributed speaker, when grounding is
          "인용구간"; None for direct-evidence segments)
        - citation_span_label (label of the quoted span, e.g. the
          cited work or interlocutor; None for direct evidence)
        - topic, key_concepts
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"] if isinstance(data, dict) else data


# ================================================================
# Step 2. Build embedding index
# ================================================================

def embed_text(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(
        input=text, model=EMBEDDING_MODEL
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def build_index(segments: list, client: OpenAI) -> dict:
    """Build an embedding index over segment texts.

    Each segment — one completeness-and-minimality utterance-unit —
    is embedded as a single vector. Keeping the segment as the atomic
    retrieval unit preserves the argument context of a letter or
    treatise passage and keeps every retrieved unit traceable to a
    single source location and grounding grade.
    """
    vectors, ids = [], []
    for s in segments:
        text = s.get("text", "").strip()
        if text:
            vectors.append(embed_text(client, text))
            ids.append(s["id"])

    return {
        "vectors": np.array(vectors, dtype=np.float32),
        "ids": ids,
    }


# ================================================================
# Step 3. Search
# ================================================================

def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / np.linalg.norm(query)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return m @ q


def search(
    query: str,
    index: dict,
    segments: list,
    client: OpenAI,
    top_k: int = 5,
    filters: dict = None,
) -> list:
    """Search segments by cosine similarity.

    Args:
        query: User question.
        index: Output of build_index.
        segments: Original segment list.
        client: OpenAI client.
        top_k: Number of top results.
        filters: Optional metadata filters. Keys may include
            'genre', 'grounding', 'topic'.
    """
    query_vec = embed_text(client, query)
    sims = cosine_similarity(query_vec, index["vectors"])

    lookup = {s["id"]: s for s in segments}
    results = []
    for sim, sid in zip(sims, index["ids"]):
        s = lookup[sid]

        if filters:
            if "genre" in filters and s.get("genre") != filters["genre"]:
                continue
            if "grounding" in filters and s.get("grounding") != filters["grounding"]:
                continue
            if "topic" in filters and s.get("topic") != filters["topic"]:
                continue

        results.append({
            "segment": s,
            "score": float(sim),
        })

    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


# ================================================================
# Step 4. Generate citation-grounded response
# ================================================================
#
# Citation guard
# --------------
# Nosa AI responses are returned under a citation guard: every
# claim must be tagged with its grounding grade. A claim drawn from
# Nosa's own words is graded "직접근거" (direct evidence); a claim
# that rests on a passage Nosa quotes from another party is graded
# "인용구간(○○)" (quoted span), where ○○ names the attributed
# speaker / source. The system prompt below instructs the model to
# preserve this distinction, and format_source_block() surfaces the
# per-segment grade and speaker so the two grounding levels never
# collapse into one.

SYSTEM_PROMPT = """You are Nosa Gi Jeong-jin (蘆沙 奇正鎭, 1798-1879),
a Neo-Confucian scholar of late Joseon. Answer the user's question
based STRICTLY on the retrieved segments from the Nosajip (蘆沙集).

Grounding rules (citation guard):
- A claim supported by Nosa's own statement is DIRECT EVIDENCE.
  Tag it "직접근거".
- A claim that rests on a passage Nosa quotes from another party is
  a QUOTED SPAN. Tag it "인용구간(SPEAKER)", naming the attributed
  speaker or source; do not present a quoted view as Nosa's own.

For each claim, cite the source (genre, title) and its grounding
grade. Do not introduce information not present in the retrieved
segments. If the retrieved segments do not address the question,
state this explicitly rather than fabricating an answer."""


def format_source_block(s: dict) -> str:
    """Render one retrieved segment with its grounding grade.

    Direct-evidence segments are labelled "직접근거". Quoted-span
    segments are labelled "인용구간(<speaker>)" so the attributed
    speaker travels with the text into the generation context.
    """
    grounding = s.get("grounding", "직접근거")
    if grounding == "인용구간":
        speaker = s.get("citation_speaker") or "?"
        span = s.get("citation_span_label")
        grade = f"인용구간({speaker})"
        if span:
            grade += f" · {span}"
    else:
        grade = "직접근거"

    header = (
        f"[蘆沙集 {s.get('genre', '')} 「{s.get('title', '')}」"
        f"{(' → ' + s['recipient']) if s.get('recipient') else ''} "
        f"| 근거등급: {grade}]"
    )
    return f"{header}\n{s.get('text', '')}"


def generate_answer(query: str, results: list, client: OpenAI) -> str:
    blocks = [format_source_block(r["segment"]) for r in results]
    context = "\n\n---\n\n".join(blocks)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Retrieved segments:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ================================================================
# Main demo
# ================================================================

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    print("Loading sample segments...")
    segments = load_segments(SAMPLE_DATA_PATH)
    print(f"Loaded {len(segments)} segments.\n")

    print("Building embedding index...")
    index = build_index(segments, client)
    print(f"Index shape: {index['vectors'].shape}\n")

    # Example queries (representative of Nosa's core doctrines):
    #   - 理는 스스로 움직이는가
    #   - 태극의 이는 원통하여 구애되지 않는가
    #   - 氣自爾를 어떻게 보는가
    query = "理는 스스로 움직이는가"
    print(f"Query: {query}\n")

    results = search(query, index, segments, client, top_k=5)
    print("Retrieved segments:")
    for i, r in enumerate(results, 1):
        s = r["segment"]
        grounding = s.get("grounding", "직접근거")
        grade = (
            f"인용구간({s.get('citation_speaker', '?')})"
            if grounding == "인용구간" else "직접근거"
        )
        print(
            f"  [{i}] {s.get('genre', '')} 「{s.get('title', '')}」 | "
            f"{grade} | score={r['score']:.3f}"
        )

    print("\nGenerating answer...\n")
    answer = generate_answer(query, results, client)
    print(answer)


if __name__ == "__main__":
    main()
