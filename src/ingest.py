"""Build a small searchable policy chunk index from PDFs in data/raw/."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "chunks.json"

# Keep chunks comfortably below the context size while preserving page/source
# traceability. The corpus is intentionally small for this assignment.
MAX_CHARS = 1800
MIN_CHARS = 80


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            candidate = f"{current}\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
                continue

        if current:
            chunks.append(current)
            current = ""

        # Long tables/paragraphs: split on sentence-ish boundaries first.
        if len(paragraph) > max_chars:
            parts = re.split(r"(?<=[.!?])\s+", paragraph)
            buffer = ""
            for part in parts:
                candidate = f"{buffer} {part}".strip() if buffer else part
                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = part[:max_chars]
            if buffer:
                current = buffer
        else:
            current = paragraph

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHARS]


def infer_category(doc_name: str, section: str, text: str) -> str:
    value = f"{doc_name} {section} {text}".lower()

    if "tax" in value or "gst" in value:
        return "tax"
    if "co-lending" in value or "colending" in value or "co lending" in value:
        return "co_lending"
    if any(term in value for term in ("interest", "rate", "charge", "fee", "penal")):
        return "interest_rates_and_charges"
    return "other_policy"


def extract_pdf(pdf_path: Path) -> list[dict]:
    records: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            text = clean_text(raw)
            if not text:
                continue

            # The first non-empty line is a useful lightweight section/title hint.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            section = lines[0][:160] if lines else pdf_path.stem

            for index, chunk_text in enumerate(split_into_chunks(text), start=1):
                chunk_id = f"{pdf_path.stem}:p{page_number}:c{index}"
                records.append(
                    {
                        "id": chunk_id,
                        "doc": pdf_path.name,
                        "version": "provided",
                        "page": page_number,
                        "section": section,
                        "category": infer_category(
                            pdf_path.stem, section, chunk_text
                        ),
                        "text": chunk_text,
                    }
                )

    return records


def build_index() -> list[dict]:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {RAW_DIR}. Put the supplied policy PDFs there."
        )

    chunks: list[dict] = []
    for pdf_path in pdfs:
        chunks.extend(extract_pdf(pdf_path))

    if not chunks:
        raise ValueError("PDFs were found but no extractable text was produced.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunks


def main() -> None:
    chunks = build_index()
    documents = sorted({c["doc"] for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(documents)} PDF(s).")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
