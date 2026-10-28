"""
Retriever: Hybrid Semantic + Title Lexical Search against Qdrant Vector Store.

Features:
  1. Semantic Search: Embed query with SentenceTransformer and query Qdrant.
  2. Out-of-Domain Filter: Immediately rejects non-DSA/web technologies (e.g. React, Vue, Django).
  3. Catalog Title-Match Re-ranking: Boosts lectures whose titles match query keywords.
  4. Catalog Anchor Fallback: Ensures foundational dedicated lectures (like Lecture 11 for Time Complexity)
     are never missed even if Whisper transcripts had noise.
  5. Prompt-Leak Cleaning: Strips hallucinated prompt-bias text from segments.
  6. Optional Adjacent Chunk Context Expansion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.config import settings
from backend.embeddings.embedder import get_embedding_model
from backend.embeddings.vector_store import get_qdrant_client
from backend.models.schemas import QueryRequest, TopicSource
from backend.utils.logging import logger

try:
    from qdrant_client.http import models as qmodels
except ImportError as _e:
    raise ImportError("qdrant-client is not installed. Run: pip install qdrant-client") from _e


# ---------------------------------------------------------------------------
# Stopwords and Domain Guard Vocabulary
# ---------------------------------------------------------------------------

STOPWORDS: Set[str] = {
    "explain", "what", "is", "how", "does", "in", "the", "and", "for", "to",
    "of", "a", "an", "c++", "dsa", "about", "can", "you", "tell", "me", "please",
    "love", "babbar", "lecture", "video", "kya", "hai", "kaise", "batao",
    "with", "using", "from", "concept", "concepts", "solution", "questions", "part"
}

# Technologies explicitly outside the scope of Love Babbar's C++ Placement DSA Course
OUT_OF_DOMAIN_KEYWORDS: Set[str] = {
    "react", "reactjs", "vue", "vuejs", "angular", "angularjs",
    "django", "flask", "fastapi", "spring", "springboot", "express", "expressjs",
    "nodejs", "node.js", "nextjs", "nuxt", "svelte", "html", "css", "tailwind",
    "bootstrap", "flutter", "reactnative", "android", "swift", "kotlin",
    "mongodb", "postgresql", "mysql", "machine learning", "deep learning", "nlp"
}

ABBREVIATIONS: Dict[str, str] = {
    "bst": "binary search tree",
    "dp": "dynamic programming",
    "ll": "linked list",
    "oops": "object oriented programming oops",
    "oop": "object oriented programming",
    "stl": "standard template library stl",
    "tle": "time limit exceeded",
    "bfs": "breadth first search",
    "dfs": "depth first search"
}

# Regex to detect Whisper prompt leak in chunk text
PROMPT_LEAK_REGEX = re.compile(
    r"^(?:(?:binary search|linear search|recursion|time complexity|space complexity|arrays|linear arrays|love babbar|c\+\+ placement|element|start|end|mid)[,\s\-–]*)+",
    re.IGNORECASE
)

_VIDEOS_CATALOG_CACHE: Optional[List[Dict[str, Any]]] = None


def get_videos_catalog() -> List[Dict[str, Any]]:
    """Loads and caches playlist metadata from data/manifests/videos.json."""
    global _VIDEOS_CATALOG_CACHE
    if _VIDEOS_CATALOG_CACHE is not None:
        return _VIDEOS_CATALOG_CACHE

    v_path = settings.videos_json_path
    if v_path.exists():
        try:
            with open(v_path, "r", encoding="utf-8") as f:
                _VIDEOS_CATALOG_CACHE = json.load(f)
                return _VIDEOS_CATALOG_CACHE
        except Exception as exc:
            logger.warning(f"Failed to load videos.json: {exc}")
    _VIDEOS_CATALOG_CACHE = []
    return _VIDEOS_CATALOG_CACHE


def extract_query_keywords(query: str) -> List[str]:
    """Extracts informative content words from query, expanding abbreviations."""
    clean_q = query.lower()
    for abbr, expanded in ABBREVIATIONS.items():
        clean_q = re.sub(r"\b" + re.escape(abbr) + r"\b", expanded, clean_q)

    tokens = re.findall(r"[a-zA-Z]{3,}", clean_q)
    return [t for t in tokens if t not in STOPWORDS]


def is_out_of_domain_query(query: str) -> bool:
    """Checks if query specifically inquires about non-DSA technologies."""
    q_lower = query.lower()
    for tech in OUT_OF_DOMAIN_KEYWORDS:
        if re.search(r"\b" + re.escape(tech) + r"\b", q_lower):
            return True
    return False


def compute_title_match(query_kws: List[str], title: str, query: str = "") -> float:
    """
    Computes a 0.0 - 1.0 relevance overlap between query keywords and lecture title.
    Includes smart priority adjustments for general vs specialized topics.
    """
    if not query_kws:
        return 0.0

    t_words = set(re.findall(r"[a-zA-Z0-9]{2,}", title.lower()))
    t_tokens = [w for w in t_words if w not in STOPWORDS]
    if not t_tokens:
        return 0.0

    matches = sum(1 for kw in query_kws if kw in t_words)
    if matches == 0:
        return 0.0

    ratio = max(matches / len(t_tokens), matches / min(len(query_kws), 3))

    # If user query is generic (e.g. "time complexity") but title is specialized ("recursive"),
    # slightly penalize specialized sub-topics so foundational lecture (Lecture 11) is #1
    if "time" in query_kws and "complexity" in query_kws:
        if "recursion" not in query_kws and "recursive" in t_words:
            ratio *= 0.75

    return min(1.0, round(ratio, 4))


def clean_chunk_text(text: str) -> str:
    """Strips hallucinated prompt leak prefix if present."""
    if not text:
        return ""
    cleaned = PROMPT_LEAK_REGEX.sub("", text).strip()
    return cleaned if len(cleaned) > 20 else text


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def embed_query(query: str) -> List[float]:
    """Embeds a single query string using the same model as the chunks."""
    model = get_embedding_model()
    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


# ---------------------------------------------------------------------------
# Adjacent chunk helpers
# ---------------------------------------------------------------------------

def _fetch_adjacent_chunks(
    video_id: str,
    start_time: float,
    end_time: float,
    collection_name: str,
    window_seconds: float = 120.0,
) -> Optional[str]:
    """Fetches adjacent transcript text within window_seconds."""
    qc = get_qdrant_client()
    try:
        result = qc.scroll(
            collection_name=collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="video_id",
                        match=qmodels.MatchValue(value=video_id),
                    ),
                    qmodels.FieldCondition(
                        key="start_time",
                        range=qmodels.Range(
                            gte=max(0.0, start_time - window_seconds),
                            lte=end_time + window_seconds,
                        ),
                    ),
                ]
            ),
            limit=6,
            with_payload=True,
            with_vectors=False,
        )
        points = result[0]
        adjacent = sorted(
            [p for p in points if not (
                p.payload.get("start_time") == start_time
                and p.payload.get("end_time") == end_time
            )],
            key=lambda p: p.payload.get("start_time", 0.0),
        )
        if not adjacent:
            return None
        return " ... ".join(clean_chunk_text(p.payload.get("text", "")) for p in adjacent).strip() or None
    except Exception as exc:
        logger.warning(f"Adjacent chunk fetch failed for {video_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core search with Hybrid Re-ranking & Relevance Guards
# ---------------------------------------------------------------------------

def search(
    request: QueryRequest,
    collection_name: Optional[str] = None,
) -> List[TopicSource]:
    """
    Performs hybrid semantic + title lexical search and returns ranked TopicSources.
    Guarantees out-of-domain rejection and foundational lecture prioritization.
    """
    # Guard 1: Detect explicit out-of-domain tech queries (e.g. React, Angular, Django)
    if is_out_of_domain_query(request.query):
        logger.info(f"[/search] Out-of-domain query detected: '{request.query}'. Returning 0 sources.")
        return []

    col = collection_name or settings.QDRANT_COLLECTION
    qc = get_qdrant_client()

    query_kws = extract_query_keywords(request.query)
    query_vector = embed_query(request.query)

    search_limit = max(request.top_k * 10, 60)
    # Search Qdrant with a generous candidate limit
    try:
        if hasattr(qc, "query_points"):
            res = qc.query_points(
                collection_name=col,
                query=query_vector,
                limit=search_limit,
                with_payload=True,
            )
            raw_hits = res.points
        else:
            raw_hits = qc.search(
                collection_name=col,
                query_vector=query_vector,
                limit=search_limit,
                with_payload=True,
            )
    except Exception as exc:
        logger.error(f"Qdrant search failed: {exc}")
        return []

    # Guard 2: Identify any catalog lectures that strongly match the query by title
    catalog = get_videos_catalog()
    strong_title_matches: List[Tuple[Dict[str, Any], float]] = []
    for v in catalog:
        t_score = compute_title_match(query_kws, v.get("title", ""), request.query)
        if t_score >= 0.40:
            strong_title_matches.append((v, t_score))
    strong_title_matches.sort(key=lambda x: x[1], reverse=True)

    # If there are strong catalog title matches (e.g. Lecture 110 for "knapsack"),
    # check if their chunks exist in raw_hits. If not, fetch chunks directly from Qdrant.
    hit_video_ids = {h.payload.get("video_id") for h in raw_hits if h.payload}
    injected_hits = []
    for v_meta, t_score in strong_title_matches[:3]:
        v_id = v_meta.get("video_id")
        if v_id and v_id not in hit_video_ids:
            try:
                scroll_res = qc.scroll(
                    collection_name=col,
                    scroll_filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(key="video_id", match=qmodels.MatchValue(value=v_id))]
                    ),
                    limit=3,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in scroll_res[0]:
                    # Create a synthetic hit with high title-driven relevance
                    class SyntheticHit:
                        pass
                    syn = SyntheticHit()
                    syn.id = pt.id
                    syn.score = 0.55 + (t_score * 0.35)
                    syn.payload = pt.payload
                    injected_hits.append(syn)
            except Exception as e:
                logger.warning(f"Could not inject chunks for {v_id}: {e}")

    all_hits = injected_hits + list(raw_hits)

    # Re-rank candidates based on hybrid score: 0.60 * vector_score + 0.40 * title_match
    scored_candidates = []
    seen_chunks = set()

    effective_threshold = request.similarity_threshold

    for hit in all_hits:
        payload = hit.payload or {}
        chunk_id = payload.get("chunk_id", str(hit.id))
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        v_score = float(hit.score)
        v_title = payload.get("video_title", "")
        t_match = compute_title_match(query_kws, v_title, request.query)

        # Clean prompt-leak from text
        raw_text = payload.get("text", "")
        cleaned_text = clean_chunk_text(raw_text)

        # If text had heavy prompt leak that falsely boosted vector score for unrelated video, penalize
        if raw_text != cleaned_text and t_match == 0:
            v_score *= 0.65

        # Hybrid score calculation:
        # Vector score is the primary semantic signal. Title match provides an additive boost.
        hybrid_score = max(v_score, (v_score * 0.70) + (t_match * 0.30))

        if t_match >= 0.50:
            hybrid_score += (t_match * 0.15)

        # Cap hybrid score to valid probability range [0.0, 0.98]
        hybrid_score = min(0.98, hybrid_score)

        if hybrid_score < effective_threshold:
            continue

        scored_candidates.append({
            "hit": hit,
            "payload": payload,
            "hybrid_score": round(hybrid_score, 4),
            "cleaned_text": cleaned_text
        })

    # Sort descending by hybrid score
    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

    sources: List[TopicSource] = []
    for cand in scored_candidates[:request.top_k]:
        payload = cand["payload"]
        video_id = payload.get("video_id", "")
        start_time = float(payload.get("start_time", 0.0))
        end_time = float(payload.get("end_time", 0.0))

        adjacent_context: Optional[str] = None
        if request.expand_adjacent:
            adjacent_context = _fetch_adjacent_chunks(
                video_id=video_id,
                start_time=start_time,
                end_time=end_time,
                collection_name=col,
            )

        sources.append(
            TopicSource(
                lecture_number=payload.get("lecture_number"),
                title=payload.get("video_title", ""),
                video_id=video_id,
                start_time=start_time,
                end_time=end_time,
                timestamp_range=(
                    f"{payload.get('start_timestamp_str', '')} - "
                    f"{payload.get('end_timestamp_str', '')}"
                ),
                youtube_url=payload.get("youtube_timestamp_url", ""),
                relevance_score=cand["hybrid_score"],
                matched_transcript_snippet=cand["cleaned_text"],
                adjacent_context=adjacent_context,
            )
        )

    logger.info(f"[/search] Query '{request.query}' -> Returning {len(sources)} verified source(s).")
    return sources
