from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Confidence = Literal["high", "medium", "low"]
Action = Literal["respond", "escalate"]

CATEGORIES = (
    "interest_rates_and_charges",
    "tax",
    "co_lending",
    "out_of_scope",
    "other_policy",
)


class LLMAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=1000)
    chunk_id: str | None
    confidence: Confidence
    grounded: bool


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    category: str
    answer: str
    source: str
    confidence: Confidence
    action: Action
    escalation_reason: str | None = None


def weakest(a: Confidence, b: Confidence) -> Confidence:
    order = {"low": 0, "medium": 1, "high": 2}
    return a if order[a] <= order[b] else b
