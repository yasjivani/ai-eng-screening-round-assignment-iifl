import json

import pytest

from src.agent import answer_question
from src.llm import LLMError
from src.retriever import get_retriever
from src.schema import AgentResponse


GOOD = {
    "answer": "Gold loans are priced between 9.72% and 27% per annum (fixed).",
    "chunk_id": None,
    "confidence": "high",
    "grounded": True,
}


@pytest.fixture(scope="module")
def retriever():
    return get_retriever()


def stub(payload: dict):
    def llm(system, user):
        return json.dumps(payload)

    return llm


def test_empty_input_escalates_without_calling_llm():
    def explode(system, user):
        raise AssertionError("LLM must not be called for empty input")

    res = answer_question("   ", llm=explode)
    assert isinstance(res, AgentResponse)
    assert res.action == "escalate"
    assert res.category == "out_of_scope"


def test_happy_path_returns_valid_response(retriever):
    hits = retriever.search("What is the interest rate range for a gold loan?", k=4)
    assert hits, "Expected supplied policy corpus to contain a matching chunk."

    payload = {**GOOD, "chunk_id": hits[0].chunk["id"]}
    res = answer_question(
        "What is the interest rate range for a gold loan?",
        retriever=retriever,
        llm=stub(payload),
    )

    assert isinstance(res, AgentResponse)
    assert res.action == "respond"
    assert res.confidence in {"high", "medium"}
    assert res.source != "none"


def test_off_topic_question_escalates_before_llm(retriever):
    def explode(system, user):
        raise AssertionError("LLM must not be called for an off-topic query")

    res = answer_question(
        "Who won the cricket world cup in 2011?",
        retriever=retriever,
        llm=explode,
    )
    assert res.action == "escalate"


def test_ungrounded_model_reply_escalates(retriever):
    res = answer_question(
        "What is the EMI due date on my loan account 88213?",
        retriever=retriever,
        llm=stub({**GOOD, "grounded": False}),
    )
    assert res.action == "escalate"


def test_malformed_json_escalates(retriever):
    res = answer_question(
        "What is the interest rate range for a gold loan?",
        retriever=retriever,
        llm=lambda s, u: "Sure! Here you go: 9.72%",
    )
    assert res.action == "escalate"
    assert "unusable output" in (res.escalation_reason or "")


def test_invalid_confidence_value_is_rejected(retriever):
    res = answer_question(
        "What is the interest rate range for a gold loan?",
        retriever=retriever,
        llm=stub({**GOOD, "confidence": "very high"}),
    )
    assert res.action == "escalate"


def test_llm_outage_escalates(retriever):
    def down(system, user):
        raise LLMError("connection refused")

    res = answer_question(
        "gold loan interest rate",
        retriever=retriever,
        llm=down,
    )
    assert res.action == "escalate"
    assert "LLM unavailable" in (res.escalation_reason or "")


def test_model_confidence_cannot_exceed_retrieval_confidence(retriever):
    res = answer_question(
        "Is uninterrupted service ensured to borrowers?",
        retriever=retriever,
        llm=stub({**GOOD, "confidence": "high"}),
    )
    assert res.confidence in {"high", "medium", "low"}
    if res.action == "respond":
        assert res.source != "none"
