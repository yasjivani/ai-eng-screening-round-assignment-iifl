"""Simple CLI for the policy-aware customer support agent."""

from __future__ import annotations

import argparse
import json

from .agent import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question against the supplied IIFL policy documents."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Customer question. Omit it for interactive mode.",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("Customer question: ").strip()

    result = answer_question(question)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
