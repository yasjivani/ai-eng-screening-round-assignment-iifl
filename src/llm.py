"""OpenAI adapter kept separate from the agent for easy testing/injection."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from .schema import LLMAnswer

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class LLMError(RuntimeError):
    pass


def complete(system: str, user: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=LLMAnswer,
            temperature=0,
        )
        message = response.choices[0].message

        if message.parsed is None:
            raise LLMError("OpenAI returned no structured result.")

        return message.parsed.model_dump_json()
    except Exception as exc:
        if isinstance(exc, LLMError):
            raise
        raise LLMError(str(exc)) from exc
