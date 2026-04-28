"""
Zhuxi AI — Pipeline Demo
Research reproducibility version for academic use.

Full system: https://askmind.ai
Related paper: [paper title]
"""

import os
from openai import OpenAI
from pinecone import Pinecone

# ================================================================
# Configuration
# ================================================================
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = "askmind"
NAMESPACE = "zhuzi-full-14k"

def get_embedding(text: str) -> list:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_chunks(query: str, top_k: int = 5) -> list:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embedding = get_embedding(query)
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE
    )
    return results["matches"]

def generate_answer(query: str, chunks: list) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = "\n\n".join([
        f"[朱子語類 卷{c['metadata'].get('volume', '')} 第{c['metadata'].get('number', '')}條] {c['metadata'].get('text', '')}"
        for c in chunks
    ])
    system_prompt = """You are Zhu Xi (朱熹), the great Neo-Confucian scholar of the Song dynasty.
Answer based only on the retrieved passages from Zhuzi Yulei (朱子語類).
Speak in first person with philosophical depth."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.35,
        max_tokens=1000
    )
    return response.choices[0].message.content

def main():
    query = "명덕이란 무엇입니까"
    print(f"Query: {query}\n")
    chunks = search_chunks(query)
    print(f"Retrieved {len(chunks)} chunks\n")
    answer = generate_answer(query, chunks)
    print(f"Answer:\n{answer}")

if __name__ == "__main__":
    main()
