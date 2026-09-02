"""
Validation that applies to any legal document.

Two levels: checks on cleaned text, and checks on generated chunks.
Both return a list of error strings so the caller can report them all
at once rather than failing on the first.
"""

import re

from backend.app.core.config import MAX_CHUNK_CHARS
from data_pipeline.chunking.ids import point_id


REQUIRED_CHUNK_FIELDS = [
    "chunk_id",
    "document_id",
    "document_type",
    "title",
    "jurisdiction",
    "language",
    "version",
    "effective_from",
    "effective_to",
    "in_force",
    "as_at",
    "source_name",
    "unit_type",
    "path",
    "content",
]


# ---------------------------------------------------------
# CLEANED TEXT
# ---------------------------------------------------------

def validate_cleaned_text(text: str) -> list[str]:
    errors: list[str] = []

    if not text.strip():
        return ["Document is empty"]

    if re.search(r"--- PAGE \d+ ---", text):
        errors.append("Forbidden content still present: page markers")

    for name, char in {
        "fi ligature": "ﬁ",
        "fl ligature": "ﬂ",
        "ff ligature": "ﬀ",
        "ffi ligature": "ﬃ",
        "ffl ligature": "ﬄ",
    }.items():
        if char in text:
            errors.append(f"Forbidden content still present: {name}")

    control = sorted(
        {
            f"U+{ord(char):04X}"
            for char in text
            if ord(char) < 32 and char not in "\n\t"
        }
    )

    if control:
        errors.append(
            "Suspicious control characters found: " + ", ".join(control)
        )

    if re.search(r"\n\s*\n\s*\n", text):
        errors.append("Multiple consecutive blank lines detected")

    return errors


# ---------------------------------------------------------
# CHUNKS
# ---------------------------------------------------------

def validate_chunks(
    chunks: list[dict],
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    errors: list[str] = []

    if not chunks:
        return ["No chunks were generated."]

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]

    duplicates = sorted(
        {cid for cid in chunk_ids if chunk_ids.count(cid) > 1}
    )

    if duplicates:
        errors.append(
            "Duplicate chunk IDs: " + ", ".join(map(str, duplicates[:5]))
        )

    # Distinct chunk ids must produce distinct point ids.
    point_ids = {point_id(cid) for cid in chunk_ids if cid}

    if len(point_ids) != len(set(chunk_ids)):
        errors.append("Point ID collision detected across chunks.")

    for index, chunk in enumerate(chunks, start=1):

        missing = [
            field
            for field in REQUIRED_CHUNK_FIELDS
            if field not in chunk
        ]

        if missing:
            errors.append(
                f"Chunk {index} ({chunk.get('chunk_id')}) missing: "
                + ", ".join(missing)
            )

        content = chunk.get("content", "")

        if not content.strip():
            errors.append(
                f"Chunk {index} ({chunk.get('chunk_id')}) has empty content."
            )

        if len(content) > max_chars:
            errors.append(
                f"Chunk {chunk.get('chunk_id')} exceeds {max_chars} "
                f"characters ({len(content)})."
            )

        if not chunk.get("path"):
            errors.append(
                f"Chunk {chunk.get('chunk_id')} has no citation path."
            )

    return errors
