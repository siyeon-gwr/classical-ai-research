"""
Jeongjo AI — Pipeline Demo

Research reproducibility version for academic use.

This standalone script demonstrates the methodology described in:
    Lee, Si-yeon. "Building a King Jeongjo AI from the Ildeungrok
    Corpus: Utterance-Unit RAG and Recorder-Based Perspective
    Analysis." Korea Journal (forthcoming).

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
SAMPLE_DATA_PATH = "./sample_data/ildeungrok_sample.json"


# ================================================================
# Step 1. Load sample utterances
# ================================================================

def load_utterances(path: str) -> list:
    """Load utterance data with metadata.

    Each utterance contains:
        - utterance_id, vol, section, label
        - hanmun (original Sinographic text)
        - kor (Korean translation)
        - recorder_name_hm, recorder_name_kor
        - year_ad, year_ganji, reign_year
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# Step 2. Build dual-index embeddings (Hanmun + Korean)
# ================================================================

def embed_text(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(
        input=text, model=EMBEDDING_MODEL
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def build_dual_index(utterances: list, client: OpenAI) -> dict:
    """Build separate Hanmun and Korean indices.

    The dual-index design supports cross-lingual retrieval: Korean
    natural-language queries match against the Korean index, while
    Hanmun keyword queries match against the Sinographic index. At
    search time, cosine similarities from both indices are aggregated
    per utterance.
    """
    hanmun_vecs, kor_vecs = [], []
    hanmun_ids, kor_ids = [], []

    for u in utterances:
        if u.get("hanmun", "").strip():
            hanmun_vecs.append(embed_text(client, u["hanmun"]))
            hanmun_ids.append(u["utterance_id"])
        if u.get("kor", "").strip():
            kor_vecs.append(embed_text(client, u["kor"]))
            kor_ids.append(u["utterance_id"])

    return {
        "hanmun": np.array(hanmun_vecs, dtype=np.float32),
        "hanmun_ids": hanmun_ids,
        "korean": np.array(kor_vecs, dtype=np.float32),
        "korean_ids": kor_ids,
    }


# ================================================================
# Step 3. Search with metadata filtering
# ================================================================

def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / np.linalg.norm(query)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return m @ q


def search(
    query: str,
    indices: dict,
    utterances: list,
    client: OpenAI,
    top_k: int = 5,
    filters: dict = None,
) -> list:
    """Search across both indices and aggregate per utterance.

    Args:
        query: User question (Korean or Hanmun).
        indices: Output of build_dual_index.
        utterances: Original utterance list.
        client: OpenAI client.
        top_k: Number of top results.
        filters: Optional metadata filters. Keys may include
            'recorder_name', 'year_ad', 'section', 'vol'.
    """
    query_vec = embed_text(client, query)

    hanmun_sims = cosine_similarity(query_vec, indices["hanmun"])
    korean_sims = cosine_similarity(query_vec, indices["korean"])

    scores = {}
    for sim, uid in zip(hanmun_sims, indices["hanmun_ids"]):
        scores.setdefault(uid, {"hanmun": None, "korean": None})
        scores[uid]["hanmun"] = float(sim)
    for sim, uid in zip(korean_sims, indices["korean_ids"]):
        scores.setdefault(uid, {"hanmun": None, "korean": None})
        scores[uid]["korean"] = float(sim)

    lookup = {u["utterance_id"]: u for u in utterances}
    results = []
    for uid, s in scores.items():
        u = lookup[uid]

        if filters:
            if "section" in filters and filters["section"] not in (u.get("section") or ""):
                continue
            if "recorder_name" in filters and u.get("recorder_name_hm") != filters["recorder_name"]:
                continue
            if "year_ad" in filters:
                yr = u.get("year_ad")
                if yr is None:
                    continue
                yf = filters["year_ad"]
                if isinstance(yf, int) and yr != yf:
                    continue
                if isinstance(yf, (list, tuple)) and not (yf[0] <= yr <= yf[1]):
                    continue

        max_score = max(v for v in s.values() if v is not None)
        results.append({
            "utterance": u,
            "hanmun_score": s["hanmun"],
            "korean_score": s["korean"],
            "max_score": max_score,
        })

    results.sort(key=lambda r: -r["max_score"])
    return results[:top_k]


# ================================================================
# Step 4. Generate citation-grounded response
# ================================================================

SYSTEM_PROMPT = """You are King Jeongjo (正祖, r. 1776-1800), the 22nd
ruler of Joseon. Answer the user's question based STRICTLY on the
retrieved utterances from the Ildeungrok. For each claim, cite the
source (volume, section, recorder, year). Do not introduce information
not present in the retrieved utterances. If the retrieved utterances
do not address the question, state this explicitly rather than
fabricating an answer."""


def generate_answer(query: str, results: list, client: OpenAI) -> str:
    blocks = []
    for r in results:
        u = r["utterance"]
        blocks.append(
            f"[Vol.{u['vol']} {u.get('label', '')} | "
            f"Recorder: {u.get('recorder_name_kor', '?')} | "
            f"Year: {u.get('year_ad', '?')}]\n"
            f"Korean: {u.get('kor', '')}\n"
            f"Hanmun: {u.get('hanmun', '')}"
        )
    context = "\n\n---\n\n".join(blocks)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Retrieved utterances:\n{context}\n\nQuestion: {query}"},
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

    print("Loading sample utterances...")
    utterances = load_utterances(SAMPLE_DATA_PATH)
    print(f"Loaded {len(utterances)} utterances.\n")

    print("Building dual-index embeddings...")
    indices = build_dual_index(utterances, client)
    print(f"Hanmun index: {indices['hanmun'].shape}")
    print(f"Korean index: {indices['korean'].shape}\n")

    query = "학문의 근본이란 무엇인가"
    print(f"Query: {query}\n")

    results = search(query, indices, utterances, client, top_k=5)
    print("Retrieved utterances:")
    for i, r in enumerate(results, 1):
        u = r["utterance"]
        print(
            f"  [{i}] Vol.{u['vol']} {u.get('label', '')} | "
            f"{u.get('recorder_name_kor', '?')} ({u.get('year_ad', '?')}) | "
            f"score={r['max_score']:.3f}"
        )

    print("\nGenerating answer...\n")
    answer = generate_answer(query, results, client)
    print(answer)


if __name__ == "__main__":
    main()
