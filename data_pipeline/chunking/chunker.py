"""
Generic chunking.

Works on `Unit` objects, so it is identical for a constitutional article
and a statutory section. Units longer than MAX_CHUNK_CHARS are split on
paragraph boundaries into parts that each keep the parent's citation
path — one vector for 24,000 characters of unrelated schedule text is
semantically meaningless and expensive to rerank.
"""

from datetime import date

from backend.app.core.config import MAX_CHUNK_CHARS
from data_pipeline.ir import Document, Unit


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on line boundaries without exceeding max_chars per part."""

    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    length = 0

    for line in text.split("\n"):

        # A single line longer than the budget has to be hard-split.
        if len(line) > max_chars:
            if current:
                parts.append("\n".join(current))
                current, length = [], 0

            for start in range(0, len(line), max_chars):
                parts.append(line[start:start + max_chars])

            continue

        # +1 for the newline that will rejoin them.
        if current and length + len(line) + 1 > max_chars:
            parts.append("\n".join(current))
            current, length = [], 0

        current.append(line)
        length += len(line) + 1

    if current:
        parts.append("\n".join(current))

    return [part for part in parts if part.strip()]


def build_chunks(
    document: Document,
    as_at: str | None = None,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    """Convert a parsed document into storable chunks."""

    as_at = as_at or date.today().isoformat()

    chunks: list[dict] = []

    for unit in document.units:
        base_id = f"{document.document_id}-{unit.slug()}"

        parts = split_text(unit.text, max_chars=max_chars)

        for index, part in enumerate(parts, start=1):

            chunk_id = (
                base_id
                if len(parts) == 1
                else f"{base_id}-part-{index}"
            )

            chunks.append(
                _build_chunk(
                    document=document,
                    unit=unit,
                    chunk_id=chunk_id,
                    content=part,
                    part=index,
                    part_count=len(parts),
                    as_at=as_at,
                )
            )

    return chunks


def _build_chunk(
    document: Document,
    unit: Unit,
    chunk_id: str,
    content: str,
    part: int,
    part_count: int,
    as_at: str,
) -> dict:
    return {
        "chunk_id": chunk_id,

        "document_id": document.document_id,
        "document_type": document.document_type,
        "title": document.title,
        "jurisdiction": document.jurisdiction,
        "language": document.language,

        # Versioning — what lets a future query filter out repealed law
        # and lets an answer state the date its authority is current to.
        "version": document.version,
        "effective_date": document.effective_date,
        "in_force": document.in_force,
        "as_at": as_at,

        "source_name": document.source_name,
        "source_url": document.source_url,

        # Citation
        "unit_type": unit.unit_type,
        "unit_number": unit.number,
        "unit_title": unit.title,
        "path": unit.path,
        "citation": " — ".join(unit.path),
        "part": part,
        "part_count": part_count,

        "content": content,
    }
