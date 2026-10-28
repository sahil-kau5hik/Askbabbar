"""
LLM Generator: Builds a grounded answer from retrieved TopicSources.

Supports:
  - provider="gemini"  -> Google Gemini via google-generativeai
  - provider="groq"    -> Groq inference (llama/mixtral) via groq SDK
  - provider="openai"  -> OpenAI-compatible APIs

The prompt is carefully crafted so the model:
  1. Only answers from the retrieved transcript context (RAG-grounded)
  2. References lecture numbers and timestamps in its answer
  3. Gracefully says "not found in lectures" when context is insufficient
  4. Never hallucinates code or algorithms not present in transcripts
"""

from __future__ import annotations

import time
from typing import List, Optional

from backend.config import settings
from backend.models.schemas import QueryResponse, TopicSource
from backend.utils.logging import logger


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert DSA (Data Structures and Algorithms) teaching assistant
for Love Babbar's Complete C++ Placement DSA Course on YouTube.

You are given transcript excerpts and metadata from Love Babbar's lectures to answer the student's question.
Guidelines:
- Ground your answer in the concepts covered in Love Babbar's lectures provided below.
- Always cite the specific Lecture Number and timestamps where the topic is taught.
- Explain concepts clearly, step-by-step, and educationally (in the energetic, encouraging style Love Babbar teaches).
- If C++ code examples or formulas (e.g., mid = s + (e - s)/2, Big-O notations, loop patterns) are taught in the lecture, include well-formatted code blocks.
- If the student asks about a concept that is completely absent from the provided lectures (or not covered in C++ DSA), clearly inform them and suggest relevant C++/DSA topics.
"""


def _build_context_block(sources: List[TopicSource]) -> str:
    """Formats TopicSources into a readable context block for the LLM."""
    if not sources:
        return "No relevant transcript excerpts found."

    blocks = []
    for i, src in enumerate(sources, start=1):
        lecture_info = f"Lecture {src.lecture_number}" if src.lecture_number else src.title
        block = (
            f"--- Source {i}: {lecture_info} | {src.timestamp_range} ---\n"
            f"{src.matched_transcript_snippet}"
        )
        if src.adjacent_context:
            block += f"\n[Adjacent context]: {src.adjacent_context}"
        blocks.append(block)

    return "\n\n".join(blocks)


def _build_user_prompt(query: str, sources: List[TopicSource]) -> str:
    context = _build_context_block(sources)
    return (
        f"QUESTION: {query}\n\n"
        f"TRANSCRIPT EXCERPTS:\n{context}\n\n"
        f"Please answer the question based solely on the transcript excerpts above."
    )


def _assess_confidence(sources: List[TopicSource], answer: str) -> str:
    """Simple heuristic confidence label."""
    if not sources:
        return "low"
    best_score = max(s.relevance_score for s in sources)
    if best_score >= 0.70:
        return "high"
    elif best_score >= 0.45:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _generate_gemini(system: str, user: str) -> str:
    """Generates answer via Google Gemini."""
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        ) from e

    api_key = settings.LLM_API_KEY.strip()
    if not api_key:
        raise ValueError("LLM_API_KEY is not set in .env (needed for Gemini).")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=settings.LLM_MODEL,
        system_instruction=system,
    )
    for attempt in range(2):
        try:
            response = model.generate_content(user)
            return response.text.strip()
        except Exception as exc:
            if "429" in str(exc) and attempt == 0:
                time.sleep(3.0)
                continue
            raise


def _generate_groq(system: str, user: str) -> str:
    """Generates answer via Groq Cloud."""
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("groq SDK not installed. Run: pip install groq") from e

    api_key = settings.GROQ_API_KEY.strip() or settings.LLM_API_KEY.strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


def _generate_openai(system: str, user: str) -> str:
    """Generates answer via OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai is not installed. Run: pip install openai") from e

    api_key = settings.LLM_API_KEY.strip()
    if not api_key:
        raise ValueError("LLM_API_KEY is not set in .env.")

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    sources: List[TopicSource],
    provider: Optional[str] = None,
) -> QueryResponse:
    """
    Generates a grounded RAG answer from retrieved TopicSources.

    Args:
        query: original user query
        sources: list of TopicSource from retriever
        provider: "gemini" | "groq" | "openai" (defaults to settings.LLM_PROVIDER)

    Returns:
        QueryResponse with answer, confidence, and sources
    """
    prov = (provider or settings.LLM_PROVIDER).lower().strip()
    user_prompt = _build_user_prompt(query, sources)

    logger.info(f"Generating answer via provider='{prov}', model='{settings.LLM_MODEL}' ...")

    if not sources:
        # Check if query asks about out-of-scope tech (e.g. React, Web Dev)
        out_of_scope_techs = [
            "react", "reactjs", "vue", "angular", "django", "flask", "spring",
            "node", "express", "html", "css", "flutter", "android", "swift", "kotlin", "machine learning"
        ]
        found_tech = next((t.capitalize() for t in out_of_scope_techs if t in query.lower()), None)
        if found_tech:
            msg = (
                f"Love Babbar's course is dedicated specifically to **C++ and Data Structures & Algorithms (DSA)**.\n\n"
                f"**{found_tech}** is a web development / software framework topic and is **not covered in this 149-lecture DSA course**.\n\n"
                "You can ask any questions regarding C++ or DSA topics from Love Babbar's course, such as:\n"
                "- **Programming & C++**: Variables, Loops, Bitwise Operators, Functions, Pointers, Memory Allocation, OOPs\n"
                "- **Core Algorithms**: Time & Space Complexity (Big-O / TLE), Binary Search, Sorting Algorithms\n"
                "- **Data Structures**: Arrays, 2D Arrays, Strings, Linked Lists, Stacks, Queues, Binary Trees, BST, Heaps, Graphs\n"
                "- **Advanced Topics**: Recursion, Backtracking, Dynamic Programming"
            )
        else:
            msg = (
                f"I couldn't find a clear answer in Love Babbar's lectures for **'{query}'**.\n\n"
                "Please make sure your query is related to **C++ or Data Structures & Algorithms** (e.g., Arrays, Binary Search, Time Complexity, Recursion, Sorting, etc.)."
            )
        return QueryResponse(
            query=query,
            answer=msg,
            confidence="low",
            sources=[],
        )

    has_key = bool(settings.LLM_API_KEY.strip() or settings.GROQ_API_KEY.strip())
    if not has_key:
        logger.info("LLM API key not set. Generating extractive summary from retrieved sources.")
        summary_lines = [
            "> *Note: LLM API key not configured in `.env`. Displaying grounded context directly from lecture transcripts:*",
            "",
            f"Here are the most relevant explanations from Love Babbar's lectures for **{query}**:",
            "",
        ]
        for i, src in enumerate(sources[:3], 1):
            lec = f"Lecture {src.lecture_number}" if src.lecture_number else (src.title or "Lecture")
            summary_lines.append(f"**{i}. {lec} [{src.timestamp_range}]** ([Watch on YouTube]({src.youtube_url})):")
            snippet = src.matched_transcript_snippet.strip()
            summary_lines.append(f"> {snippet}")
            summary_lines.append("")
        answer = "\n".join(summary_lines)
    else:
        try:
            if prov == "gemini":
                answer = _generate_gemini(SYSTEM_PROMPT, user_prompt)
            elif prov == "groq":
                answer = _generate_groq(SYSTEM_PROMPT, user_prompt)
            elif prov in ("openai", "azure"):
                answer = _generate_openai(SYSTEM_PROMPT, user_prompt)
            else:
                raise ValueError(f"Unknown LLM provider: '{prov}'. Use 'gemini', 'groq', or 'openai'.")
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            # Graceful fallback: construct grounded answer from retrieved transcripts directly
            fallback_lines = [
                f"> *Sorry, I encountered an error with the AI model. Displaying verified explanations directly from lecture transcripts:*",
                "",
                f"### Love Babbar Lecture Explanations for: **{query}**",
                "",
            ]
            for i, src in enumerate(sources[:3], 1):
                lec = f"Lecture {src.lecture_number}: {src.title}" if src.lecture_number else src.title
                fallback_lines.append(f"#### {i}. {lec}")
                fallback_lines.append(f"- **Video Timestamp**: [{src.timestamp_range}]({src.youtube_url})")
                fallback_lines.append(f"- **Transcript Excerpt**: \"{src.matched_transcript_snippet}\"")
                if src.adjacent_context:
                    fallback_lines.append(f"- **Additional Context**: {src.adjacent_context}")
                fallback_lines.append("")
            answer = "\n".join(fallback_lines)

    confidence = _assess_confidence(sources, answer)
    logger.info(f"Answer generated. confidence={confidence}, sources={len(sources)}")

    return QueryResponse(
        query=query,
        answer=answer,
        confidence=confidence,
        sources=sources,
    )
