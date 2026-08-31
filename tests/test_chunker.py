"""
Chunking: oversized units must be split, and every chunk must carry a
complete, citable payload.

Schedule 6 of the Constitution was a single 24,146-character chunk —
one vector for pages of unrelated transitional provisions.
"""

from data_pipeline.chunking.chunker import build_chunks, split_text
from data_pipeline.ir import Document, Unit
from data_pipeline.validators.generic import (
    REQUIRED_CHUNK_FIELDS,
    validate_chunks,
)


def make_document(units: list[Unit]) -> Document:
    return Document(
        document_id="test-doc-2026",
        title="Test Document, 2026",
        document_type="constitution",
        jurisdiction="Kenya",
        language="English",
        version="2026",
        effective_date="2026-01-01",
        in_force=True,
        source_name="Kenya Law",
        source_url=None,
        units=units,
    )


def test_short_text_is_not_split():
    assert split_text("a short article", max_chars=100) == [
        "a short article"
    ]


def test_long_text_splits_within_the_cap():
    text = "\n".join(f"line {n} of the schedule" for n in range(500))

    parts = split_text(text, max_chars=400)

    assert len(parts) > 1
    assert all(len(part) <= 400 for part in parts)


def test_split_preserves_all_content():
    text = "\n".join(f"paragraph {n}" for n in range(200))

    parts = split_text(text, max_chars=300)

    assert "\n".join(parts) == text


def test_single_overlong_line_is_hard_split():
    parts = split_text("x" * 1000, max_chars=250)

    assert all(len(part) <= 250 for part in parts)
    assert "".join(parts) == "x" * 1000


def test_oversized_unit_becomes_multiple_chunks():
    unit = Unit(
        unit_type="schedule",
        number="6",
        title="SIXTH SCHEDULE",
        path=["SIXTH SCHEDULE"],
        text="\n".join(f"transitional provision {n}" for n in range(400)),
    )

    chunks = build_chunks(make_document([unit]), max_chars=1000)

    assert len(chunks) > 1
    assert all(len(chunk["content"]) <= 1000 for chunk in chunks)

    # Parts stay individually addressable and keep the parent citation.
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(set(ids)) == len(ids)
    assert all(chunk["citation"] == "SIXTH SCHEDULE" for chunk in chunks)
    assert all(chunk["part_count"] == len(chunks) for chunk in chunks)


def test_single_part_unit_keeps_a_clean_id():
    unit = Unit(
        unit_type="article",
        number="49",
        title="Rights of arrested persons",
        path=["Chapter Four", "Article 49"],
        text="(1) An arrested person has the right to remain silent.",
    )

    chunks = build_chunks(make_document([unit]))

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == (
        "test-doc-2026-chapter-four-article-49"
    )
    assert "part" not in chunks[0]["chunk_id"]


def test_chunks_carry_every_required_field():
    unit = Unit(
        unit_type="article",
        number="1",
        title="Sovereignty of the people",
        path=["Chapter One", "Article 1"],
        text="All sovereign power belongs to the people of Kenya.",
    )

    chunks = build_chunks(make_document([unit]))

    for field in REQUIRED_CHUNK_FIELDS:
        assert field in chunks[0], f"missing {field}"

    assert validate_chunks(chunks) == []


def test_versioning_fields_reach_the_payload():
    """Without these, repealed law cannot be told from current law."""

    unit = Unit(
        unit_type="article",
        number="1",
        title="Title",
        path=["Chapter One", "Article 1"],
        text="Some legal text.",
    )

    chunk = build_chunks(make_document([unit]), as_at="2026-08-31")[0]

    assert chunk["version"] == "2026"
    assert chunk["effective_date"] == "2026-01-01"
    assert chunk["in_force"] is True
    assert chunk["as_at"] == "2026-08-31"


def test_validator_catches_duplicate_chunk_ids():
    unit = Unit(
        unit_type="article",
        number="1",
        title="Title",
        path=["Chapter One", "Article 1"],
        text="Some legal text.",
    )

    chunks = build_chunks(make_document([unit]))
    duplicated = chunks + chunks

    assert any("Duplicate" in error for error in validate_chunks(duplicated))


def test_validator_catches_oversized_chunks():
    chunk = {
        field: "x" for field in REQUIRED_CHUNK_FIELDS
    }
    chunk["path"] = ["Chapter One"]
    chunk["content"] = "x" * 5000

    errors = validate_chunks([chunk], max_chars=4000)

    assert any("exceeds" in error for error in errors)
