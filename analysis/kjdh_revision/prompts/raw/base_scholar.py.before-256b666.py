"""
base_scholar.py
모든 학자 RAG의 공통 로직 — 주자AI 수준 출처 레이어 시스템
"""

import re
from typing import List, Optional, Tuple

# ================================================================
#  언어 감지
# ================================================================

def detect_language(text: str) -> str:
    if re.search('[ㄱ-ㅎ가-힣]', text):
        return 'Korean'
    elif re.search('[\u4e00-\u9fff]', text):
        simplified = ['个', '学', '国', '说', '这', '们', '么', '吗']
        traditional = ['個', '學', '國', '說', '這', '們', '麼', '嗎']
        if any(c in text for c in simplified):
            return 'Chinese-Simplified'
        elif any(c in text for c in traditional):
            return 'Chinese-Traditional'
        return 'Chinese-Traditional'
    elif re.search('[\u3040-\u309f\u30a0-\u30ff]', text):
        return 'Japanese'
    return 'English'


def extract_language_override(text: str) -> Tuple[Optional[str], str]:
    target_lang = None
    if "영어로" in text or "in English" in text:
        target_lang = 'English'
        text = re.sub(r'(영어로|in English)\s*(말해줘|말해주세요|답해줘|답해주세요)?', '', text).strip()
    elif "중국어로" in text or "in Chinese" in text:
        target_lang = 'Chinese-Traditional'
        text = re.sub(r'(중국어로|in Chinese)\s*(말해줘|말해주세요|답해줘|답해주세요)?', '', text).strip()
    elif "일본어로" in text or "in Japanese" in text:
        target_lang = 'Japanese'
        text = re.sub(r'(일본어로|in Japanese)\s*(말해줘|말해주세요|답해줘|답해주세요)?', '', text).strip()
    return target_lang, text


# ================================================================
#  임계값 — fallback (인물별 설정은 scholars_config.py에서)
# ================================================================
THRESHOLD_DIRECT  = 0.30
THRESHOLD_SIMILAR = 0.22
THRESHOLD_KEYWORD = 0.16
HISTORY_MIN       = 0.18


def compute_composite_score(vector_score: float, text: str, keywords: List[str]) -> float:
    if not keywords:
        return vector_score
    hits = sum(1 for kw in keywords if kw in text)
    bonus = min(hits * 0.04, 0.12)
    return min(vector_score + bonus, 1.0)


def classify_source_layer(score: float, td=None, ts=None, tk=None) -> str:
    td = td or THRESHOLD_DIRECT
    ts = ts or THRESHOLD_SIMILAR
    tk = tk or THRESHOLD_KEYWORD
    if score >= td:
        return "direct"
    elif score >= ts:
        return "similar"
    elif score >= tk:
        return "keyword"
    return "none"


def _to_mutable(m):
    """Pinecone ScoredVector(스키마 고정) → 새 키 대입 가능한 순수 dict."""
    md = m["metadata"]
    return {"id": m["id"], "score": m["score"], "metadata": dict(md) if md else {}}


# ================================================================
#  공통 RAG 검색
# ================================================================

def search_chunks(
    query: str,
    top_k: int,
    namespace: str,
    history_namespace: str,
    keywords: List[str],
    expansion_map: dict,
    get_embedding_fn,
    get_index_fn,
    default_prefix: str = "",
    scholar_config: dict = None
) -> dict:
    sc = scholar_config or {}
    td = sc.get("threshold_direct", THRESHOLD_DIRECT)
    ts = sc.get("threshold_similar", THRESHOLD_SIMILAR)
    tk = sc.get("threshold_keyword", THRESHOLD_KEYWORD)
    hm = sc.get("history_min", HISTORY_MIN)
    cleaned = query.lower().strip()
    for ch in ['?', '!', '。', '요', '은', '는', '이', '가']:
        cleaned = cleaned.replace(ch, '')

    expansions = []
    for key, expansion in expansion_map.items():
        if key in cleaned:
            expansions.append(expansion)
    expanded_query = f"{query} {' '.join(expansions)}" if expansions else query

    if len(expanded_query) < 15 and default_prefix:
        expanded_query = f"{default_prefix} {expanded_query}"

    embedding = get_embedding_fn(expanded_query)
    index = get_index_fn()

    main_results = index.query(
        vector=embedding,
        top_k=top_k * 3,
        include_metadata=True,
        namespace=namespace
    )

    history_results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,
        namespace=history_namespace
    )

    # Pinecone ScoredVector는 스키마 고정. 새 키 대입도 하류의 임의 접근도 막힘.
    # 검색 초반에 순수 dict로 변환. 이후 조작은 dict에서만.
    main_matches = [_to_mutable(m) for m in main_results["matches"]]
    history_matches = [_to_mutable(m) for m in history_results["matches"]]

    # 화자 귀속 필드. nosa 네임스페이스만 채워져 있고 나머지 학자는 None이 된다.
    # 업로드는 warning을 quote_warning 키로 넣는다(scripts/nosa/3_upload.py:63).
    for r in main_matches + history_matches:
        md = r["metadata"]
        r["speaker"] = md.get("speaker")
        r["quoted_speaker"] = md.get("quoted_speaker")
        r["warning"] = md.get("warning") or md.get("quote_warning")

    main_scored = []
    for r in main_matches:
        text = r["metadata"].get("text", "")
        composite = compute_composite_score(r["score"], text, keywords)
        layer = classify_source_layer(composite, td, ts, tk)
        if layer == "none":
            continue
        r["composite_score"] = composite
        r["source_layer"] = layer
        main_scored.append(r)

    layer_order = {"direct": 0, "similar": 1, "keyword": 2}
    main_scored.sort(key=lambda x: (layer_order[x["source_layer"]], -x["composite_score"]))

    if not main_scored:
        fallback = [r for r in main_matches if r["score"] >= tk * 0.8]
        for r in fallback:
            r["composite_score"] = r["score"]
            r["source_layer"] = "keyword"
        main_scored = fallback

    history_scored = []
    for r in history_matches:
        if r["score"] < hm:
            continue
        r["composite_score"] = r["score"]
        r["source_layer"] = "history"
        history_scored.append(r)

    return {
        "main": main_scored[:top_k],
        "history": history_scored[:3]
    }


# ================================================================
#  공통 답변 생성
# ================================================================

def generate_answer(
    query: str,
    chunks: dict,
    target_language: str,
    scholar_config: dict,
    get_client_fn,
    cite_guard: bool = True
) -> dict:
    client = get_client_fn()
    main_chunks    = chunks.get("main", [])
    history_chunks = chunks.get("history", [])
    direct_chunks  = [c for c in main_chunks if c.get("source_layer") == "direct"]
    similar_chunks = [c for c in main_chunks if c.get("source_layer") == "similar"]
    keyword_chunks = [c for c in main_chunks if c.get("source_layer") == "keyword"]

    if direct_chunks:
        overall_layer, answer_layer = "direct", "원전발화"
    elif similar_chunks:
        overall_layer, answer_layer = "similar", "원리유추"
    elif keyword_chunks:
        overall_layer, answer_layer = "keyword", "원리유추"
    else:
        overall_layer, answer_layer = "none", "미지영역"

    # 컨텍스트
    context_parts = []
    if direct_chunks:
        context_parts.append("=== 직접 관련 원문 (출처 표기 가능) ===")
        for c in direct_chunks:
            vol = c["metadata"].get("volume", "")
            src = c["metadata"].get("source", "")
            qs = c.get("quoted_speaker") if cite_guard else None
            if qs:
                context_parts.append(f"[{qs}의 말을 인용한 대목] {c['metadata']['text']}")
            else:
                context_parts.append(f"[{src} {vol}] {c['metadata']['text']}")
    if similar_chunks:
        context_parts.append("=== 유사 관련 원문 (참조 가능, 직접 인용 아님) ===")
        for c in similar_chunks:
            vol = c["metadata"].get("volume", "")
            src = c["metadata"].get("source", "")
            qs = c.get("quoted_speaker") if cite_guard else None
            if qs:
                context_parts.append(f"[{qs}의 말을 인용한 대목] {c['metadata']['text']}")
            else:
                context_parts.append(f"[{src} {vol}] {c['metadata']['text']}")
    if keyword_chunks:
        context_parts.append("=== 키워드 연관 원문 (원리 도출용) ===")
        for c in keyword_chunks:
            vol = c["metadata"].get("volume", "")
            src = c["metadata"].get("source", "")
            qs = c.get("quoted_speaker") if cite_guard else None
            if qs:
                context_parts.append(f"[{qs}의 말을 인용한 대목] {c['metadata']['text']}")
            else:
                context_parts.append(f"[{src} {vol}] {c['metadata']['text']}")
    if history_chunks:
        context_parts.append("=== 생애/맥락 (답변에 자연스럽게 녹여 쓸 것) ===")
        for c in history_chunks:
            src = c["metadata"].get("source", "")
            context_parts.append(f"[{src}] {c['metadata']['text']}")

    context = "\n\n".join(context_parts) if context_parts else "(관련 원문 없음)"

    personas = scholar_config.get("personas", {})
    persona = personas.get(target_language, personas.get("Korean", {
        "identity": f"당신은 {scholar_config.get('name_ko', '')}이다.",
        "style": "학자답게 답하라.",
        "examples": "그러하다"
    }))

    # 규칙 5는 실제 인용 구간이 잡혔을 때만 넣는다.
    # quoted_speaker가 없는 학자(nosa 외 8명)의 프롬프트를 건드리지 않기 위한 조건.
    has_quote = any(c.get("quoted_speaker") for c in (direct_chunks + similar_chunks + keyword_chunks))
    cite_rule = (
        f"5. [~의 말을 인용한 대목]: 남의 말을 옮긴 구간이다. {scholar_config.get('name_ko', '')} 본인의 주장으로 진술하지 말고, "
        "인용임을 밝힌 뒤 그에 대한 자신의 입장을 구분해 답하라."
        if (cite_guard and has_quote) else ""
    )

    system_prompt = f"""**핵심 원칙: 아래 제공된 원문만을 1차 근거로 삼아라. GPT 사전학습 지식 사용 절대 금지.**

{persona['identity']}
{scholar_config.get('background', '')}

**출처 레이어 규칙**
1. [직접 관련 원문]: 적극 활용. 확신 있게 답하라.
2. [유사 관련 원문]: 참조 가능. 직접 인용처럼 쓰지 말 것.
3. [키워드 연관 원문]: 원리 도출 참조용.
4. [생애/맥락]: 답변에 자연스럽게 녹여 쓸 것.
   절대 금지: "기록에 따르면", "문집에 의하면" 등 메타 표현.
{cite_rule}
[원문]
{context}

[답변 원칙]
0. CRITICAL: RESPOND IN {target_language} ONLY.
0-1. 핵심 한자어는 병기: 예) 리(理), 기(氣), 경(敬)
0-2. 각 단락 4-6문장. 단락 수 3-4개. 최소 8문장 이상.
0-3. 학술 해설자 말투 절대 금지.
0-4. 원문이 한문이면 한국어로 풀어 설명하고 한문은 괄호로 병기.

1. {persona['style']}. 예시: {persona['examples']}
2. 1인칭 적극 활용.
3. 과거 저술/발언 참조: "내가 일찍이 ~라 하였듯이"

**답변 레이어**
1. 원전 직접 근거 → 확신 있게.
2. 직접 근거 없으나 추론 가능 → "{scholar_config['inference_prefix']}..." 시작.
3. 미지 영역 → "{scholar_config['unknown_prefix']}..." 시작.

[절대 금지]
- "기록에 따르면", "문집에 의하면" 등 메타 표현
- 학술 해설자 말투
- {scholar_config.get('name_ko', '')} 외 다른 인격
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=scholar_config.get("temperature", 0.35),
        max_tokens=2000,
        presence_penalty=0.3,
        frequency_penalty=0.3
    )

    # 출처 — direct/similar만 표시 (keyword 제외)
    sources_detail = []
    for c in [c for c in main_chunks if c.get("source_layer") in ("direct", "similar")][:7]:
        layer = c.get("source_layer", "similar")
        vol = c["metadata"].get("volume", "")
        src = c["metadata"].get("source", "")
        vol_match = re.search(r'卷之(\S+)', vol)
        vol_str = f"卷之{vol_match.group(1)}" if vol_match else vol
        reference = f"{src} {vol_str}" if vol_str else src
        qs = c.get("quoted_speaker") if cite_guard else None
        sources_detail.append({
            "reference": f"{reference} ({qs} 인용)" if qs else reference,
            "similarity": round(c.get("composite_score", c["score"]) * 100, 1),
            "vector_score": round(c["score"] * 100, 1),
            "source_layer": layer,
            "layer_label": "인용구간" if qs else {"direct": "직접근거", "similar": "유사참조"}.get(layer, "참조"),
            "text": c["metadata"].get("text", "")
        })

    return {
        "answer": response.choices[0].message.content,
        "sources_detail": sources_detail,
        "overall_layer": overall_layer,
        "answer_layer": answer_layer
    }
