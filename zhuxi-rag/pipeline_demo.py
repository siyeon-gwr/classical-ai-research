"""
Zhuxi AI — Pipeline Demo

Research reproducibility version for academic use.

This standalone script demonstrates the methodology described in:
    Lee, Si-yeon. "Constructing a Zhuxi AI from the Zhuzi Yulei
    Corpus: Utterance Vectorization and Hallucination Control in
    RAG-Based Generation Frameworks." Geunyeok Hanmunhakhoe (under
    review).

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
SAMPLE_DATA_PATH = "./sample_data/zhuzi_yulei_sample.json"


# ================================================================
# Step 1. Load sample utterances
# ================================================================

def load_utterances(path: str) -> list:
    """Load utterance data with metadata.

    Each utterance contains:
        - id, volume_num, item_num
        - text (Sinographic question-answer pair)
        - recorder, recorder_info
        - category, key_concepts
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["utterances"] if isinstance(data, dict) else data


# ================================================================
# Step 2. Build embedding index
# ================================================================

def embed_text(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(
        input=text, model=EMBEDDING_MODEL
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def build_index(utterances: list, client: OpenAI) -> dict:
    """Build embedding index over utterance texts.

    Each utterance — a question-answer pair (問-曰) with recorder
    information — is embedded as a single vector. The utterance unit
    preserves dialogue context and allows source-traceable retrieval.
    """
    vectors, ids = [], []
    for u in utterances:
        text = u.get("text", "").strip()
        if text:
            vectors.append(embed_text(client, text))
            ids.append(u["id"])

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
    utterances: list,
    client: OpenAI,
    top_k: int = 5,
    filters: dict = None,
) -> list:
    """Search utterances by cosine similarity.

    Args:
        query: User question.
        index: Output of build_index.
        utterances: Original utterance list.
        client: OpenAI client.
        top_k: Number of top results.
        filters: Optional metadata filters. Keys may include
            'recorder', 'category', 'volume_num'.
    """
    query_vec = embed_text(client, query)
    sims = cosine_similarity(query_vec, index["vectors"])

    lookup = {u["id"]: u for u in utterances}
    results = []
    for sim, uid in zip(sims, index["ids"]):
        u = lookup[uid]

        if filters:
            if "recorder" in filters and u.get("recorder") != filters["recorder"]:
                continue
            if "category" in filters and u.get("category") != filters["category"]:
                continue
            if "volume_num" in filters:
                vn = u.get("volume_num")
                vf = filters["volume_num"]
                if isinstance(vf, int) and vn != vf:
                    continue
                if isinstance(vf, (list, tuple)) and not (vf[0] <= vn <= vf[1]):
                    continue

        results.append({
            "utterance": u,
            "score": float(sim),
        })

    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


# ================================================================
# Step 4. Generate citation-grounded response
# ================================================================

SYSTEM_PROMPT = """You are Zhu Xi (朱熹, 1130-1200), the great
Neo-Confucian scholar of the Song dynasty. Answer the user's question
based STRICTLY on the retrieved utterances from the Zhuzi Yulei
(朱子語類). For each claim, cite the source (volume, item number,
recorder). Do not introduce information not present in the retrieved
utterances. If the retrieved utterances do not address the question,
state this explicitly rather than fabricating an answer."""


def generate_answer(query: str, results: list, client: OpenAI) -> str:
    blocks = []
    for r in results:
        u = r["utterance"]
        blocks.append(
            f"[朱子語類 卷{u.get('volume_num', '')} 第{u.get('item_num', '')}條 | "
            f"Recorder: {u.get('recorder', '?')}]\n"
            f"{u.get('text', '')}"
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

    print("Building embedding index...")
    index = build_index(utterances, client)
    print(f"Index shape: {index['vectors'].shape}\n")

    query = "명덕이란 무엇입니까"
    print(f"Query: {query}\n")

    results = search(query, index, utterances, client, top_k=5)
    print("Retrieved utterances:")
    for i, r in enumerate(results, 1):
        u = r["utterance"]
        print(
            f"  [{i}] 卷{u.get('volume_num', '')} 第{u.get('item_num', '')}條 | "
            f"{u.get('recorder', '?')} | score={r['score']:.3f}"
        )

    print("\nGenerating answer...\n")
    answer = generate_answer(query, results, client)
    print(answer)


if __name__ == "__main__":
    main()
