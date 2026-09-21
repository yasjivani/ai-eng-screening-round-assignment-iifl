"""Orchestration: retrieval -> grounded LLM answer -> validated response."""

from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from .llm import LLMError, complete
from .retriever import Hit, get_retriever
from .schema import AgentResponse, Confidence, LLMAnswer, weakest

TOP_K = 4
MIN_SCORE = 3.0
HIGH_SCORE = 8.0

ESCALATION_TEXT = (
    "I can't answer this from the published policy documents. "
    "I'm routing it to a human support executive who can help."
)

SYSTEM_PROMPT = """You are a customer support assistant for IIFL Finance, an Indian NBFC.

You answer ONLY from the policy extracts supplied in the user message.

Rules:
- Never use outside knowledge or guess a number, rate, fee or date.
- If the extracts do not fully answer the question, set grounded=false and confidence=low.
- Account-specific questions about a customer's own balance, EMI date, loan status,
  account number or transaction cannot be answered from generic policy documents.
- If the extracts cover a similar but different product, do not substitute it.
- Keep the answer under 90 words.
- Quote figures exactly as written, including whether GST is extra.
- chunk_id must be the id of the single extract relied on most.

Return only the requested structured output."""


def _render_context(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        c = hit.chunk
        blocks.append(
            f"[id: {c['id']}]\n"
            f"Document: {c['doc']} ({c['version']}), "
            f"page {c['page']}, section: {c['section']}\n"
            f"{c['text']}"
        )
    return "\n\n".join(blocks)


def _citation(chunk: dict) -> str:
    return (
        f"{chunk['doc']} {chunk['version']} — "
        f"page {chunk['page']}, {chunk['section']}"
    )


def _retrieval_confidence(hits: list[Hit]) -> Confidence:
    top = hits[0].score
    if top >= HIGH_SCORE:
        return "high"
    if top >= MIN_SCORE * 1.5:
        return "medium"
    return "low"


def _escalate(
    query: str,
    reason: str,
    category: str = "out_of_scope",
    source: str = "none",
) -> AgentResponse:
    return AgentResponse(
        query=query,
        category=category,
        answer=ESCALATION_TEXT,
        source=source,
        confidence="low",
        action="escalate",
        escalation_reason=reason,
    )


def _parse(raw: str) -> LLMAnswer:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) < 2:
            raise ValueError("Malformed code fence.")
        text = parts[1].removeprefix("json").strip()

    return LLMAnswer.model_validate(json.loads(text))


def answer_question(
    query: str,
    retriever=None,
    llm: Callable[[str, str], str] = complete,
) -> AgentResponse:
    query = (query or "").strip()

    if len(query) < 5:
        return _escalate(query, "Empty or too-short question.")

    try:
        retriever = retriever or get_retriever()
        hits = retriever.search(query, k=TOP_K)
    except (FileNotFoundError, ValueError) as exc:
        return _escalate(query, f"Retriever unavailable: {exc}")

    if not hits or hits[0].score < MIN_SCORE:
        return _escalate(query, "No policy content matched the question.")

    top_chunk = hits[0].chunk
    category = top_chunk.get("category", "other_policy")

    try:
        raw = llm(
            SYSTEM_PROMPT,
            f"Policy extracts:\n\n{_render_context(hits)}\n\n"
            f"Customer question: {query}",
        )
    except LLMError as exc:
        return _escalate(
            query,
            f"LLM unavailable: {exc}",
            category=category,
        )

    try:
        parsed = _parse(raw)
    except (json.JSONDecodeError, ValidationError, IndexError, ValueError) as exc:
        return _escalate(
            query,
            f"Model returned unusable output: {exc}",
            category=category,
        )

    if not parsed.grounded:
        return _escalate(
            query,
            "Retrieved policy text does not answer the question.",
            category=category,
        )

    cited = next(
        (h.chunk for h in hits if h.chunk["id"] == parsed.chunk_id),
        top_chunk,
    )
    confidence = weakest(_retrieval_confidence(hits), parsed.confidence)

    if confidence == "low":
        return _escalate(
            query,
            "Confidence below the threshold for an automated reply.",
            category=category,
            source=_citation(cited),
        )

    return AgentResponse(
        query=query,
        category=category,
        answer=parsed.answer.strip(),
        source=_citation(cited),
        confidence=confidence,
        action="respond",
    )
