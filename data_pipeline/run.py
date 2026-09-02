"""
Pocket Lawyer ingestion pipeline.

    python -m data_pipeline.run --document constitution-of-kenya-2010
    python -m data_pipeline.run --all
    python -m data_pipeline.run --document X --from chunk
    python -m data_pipeline.run --document X --recreate

Stages:

    extract -> clean -> structure -> chunk -> store

Each stage writes its artifact to disk, so `--from` can resume without
redoing expensive work. Only `store` needs Qdrant running.
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date

from backend.app.core.config import COLLECTION_NAME, MAX_CHUNK_CHARS
from data_pipeline import adapters
from data_pipeline.chunking.chunker import build_chunks
from data_pipeline.cleaners import base as cleaner_base
from data_pipeline.extractors.pdf_extractor import extract_text
from data_pipeline.ir import Document
from data_pipeline.metadata import build_metadata, write_metadata
from data_pipeline.registry import DocumentEntry, get_document, load_registry
from data_pipeline.validators import generic as generic_validator
from data_pipeline.vectorstore.collection import (
    ensure_collection,
    get_client,
    point_count,
)
from data_pipeline.vectorstore.ingest import (
    delete_document,
    ingest_chunks,
    load_embedding_model,
)


STAGES = ["extract", "clean", "structure", "chunk", "store"]


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

def banner(text: str) -> None:
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def report(errors: list[str], label: str) -> None:
    if errors:
        print(f"[FAIL] {label}")
        for error in errors:
            print(f"       {error}")
        raise SystemExit(1)

    print(f"[PASS] {label}")


# ---------------------------------------------------------
# STAGES
# ---------------------------------------------------------

def stage_extract(entry: DocumentEntry) -> None:
    print(f"Extracting {entry.raw_path.name}...")

    extract_text(entry.raw_path, entry.extracted_path)


def stage_clean(entry: DocumentEntry) -> None:
    print("Cleaning extracted text...")

    spec = adapters.cleaner_spec(entry.cleaner)

    text = entry.extracted_path.read_text(encoding="utf-8")

    cleaned = cleaner_base.clean(text, spec)

    entry.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    entry.cleaned_path.write_text(cleaned, encoding="utf-8")

    report(
        generic_validator.validate_cleaned_text(cleaned),
        "Generic text validation",
    )

    validator = adapters.validator(entry.validator)

    if validator and hasattr(validator, "validate_cleaned_text"):
        report(
            validator.validate_cleaned_text(cleaned),
            f"{entry.document_type.title()} text validation",
        )

    print(f"       {len(cleaned):,} characters -> {entry.cleaned_path.name}")


def stage_structure(entry: DocumentEntry) -> Document:
    print("Detecting legal structure...")

    parse = adapters.structure_parser(entry.structure)

    text = entry.cleaned_path.read_text(encoding="utf-8")

    document = parse(text, entry)

    validator = adapters.validator(entry.validator)

    if validator and hasattr(validator, "validate_document"):
        report(
            validator.validate_document(document),
            "Document structure validation",
        )

    entry.structure_path.parent.mkdir(parents=True, exist_ok=True)
    entry.structure_path.write_text(
        json.dumps(asdict(document), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    by_type: dict[str, int] = {}
    for unit in document.units:
        by_type[unit.unit_type] = by_type.get(unit.unit_type, 0) + 1

    summary = ", ".join(f"{count} {name}s" for name, count in by_type.items())

    print(f"       {summary} -> {entry.structure_path.name}")

    return document


def stage_chunk(entry: DocumentEntry, as_at: str) -> list[dict]:
    print("Chunking...")

    raw = json.loads(entry.structure_path.read_text(encoding="utf-8"))

    document = Document(**{**raw, "units": []})

    from data_pipeline.ir import Unit

    document.units = [Unit(**unit) for unit in raw["units"]]

    chunks = build_chunks(document, as_at=as_at)

    report(generic_validator.validate_chunks(chunks), "Chunk validation")

    entry.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    entry.chunks_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    longest = max(len(chunk["content"]) for chunk in chunks)
    split = sum(1 for chunk in chunks if chunk["part_count"] > 1)

    print(
        f"       {len(chunks)} chunks "
        f"(longest {longest:,} chars, cap {MAX_CHUNK_CHARS:,}; "
        f"{split} from split units)"
    )

    return chunks


def stage_store(
    entry: DocumentEntry,
    chunks: list[dict],
    as_at: str,
    recreate: bool = False,
) -> None:
    print(f"Storing into '{COLLECTION_NAME}'...")

    client = get_client()

    # Validate the collection schema before anything destructive runs.
    ensure_collection(client, recreate=recreate)

    before = point_count(client)

    # Load the model before deleting too, so a model failure cannot
    # leave the collection emptied.
    model = load_embedding_model()

    # Delete so a document whose chunk set shrank leaves no orphaned
    # vectors. With deterministic IDs this keeps ingestion idempotent.
    delete_document(client, entry.document_id)

    written = ingest_chunks(client, model, chunks)

    after = point_count(client)

    if written != len(chunks):
        report(
            [f"Wrote {written} points for {len(chunks)} chunks."],
            "Vector count",
        )

    print(f"[PASS] {written} vectors stored")
    print(f"       collection total: {before} -> {after}")

    metadata = build_metadata(entry, chunk_count=len(chunks), as_at=as_at)
    write_metadata(entry, metadata)

    print(f"       metadata -> {entry.metadata_path.name}")


# ---------------------------------------------------------
# DRIVER
# ---------------------------------------------------------

def run_document(
    entry: DocumentEntry,
    start: str,
    recreate: bool = False,
) -> None:
    banner(f"POCKET LAWYER — INGEST: {entry.document_id}")

    as_at = date.today().isoformat()

    begin = STAGES.index(start)

    chunks: list[dict] | None = None

    if begin <= STAGES.index("extract"):
        stage_extract(entry)

    if begin <= STAGES.index("clean"):
        stage_clean(entry)

    if begin <= STAGES.index("structure"):
        stage_structure(entry)

    if begin <= STAGES.index("chunk"):
        chunks = stage_chunk(entry, as_at=as_at)

    if begin <= STAGES.index("store"):
        if chunks is None:
            chunks = json.loads(
                entry.chunks_path.read_text(encoding="utf-8")
            )

        stage_store(entry, chunks, as_at=as_at, recreate=recreate)

    print()
    print(f"[PASS] {entry.document_id} ingested")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Pocket Lawyer ingestion pipeline."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--document", help="document_id from documents.yaml")
    group.add_argument(
        "--all",
        action="store_true",
        help="ingest every registered document",
    )

    parser.add_argument(
        "--from",
        dest="start",
        choices=STAGES,
        default="extract",
        help="resume from this stage (default: extract)",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "drop and rebuild the collection before storing; needed "
            "when its vector schema has changed"
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered documents and exit",
    )

    args = parser.parse_args()

    if args.list:
        for entry in load_registry():
            print(f"{entry.document_id}  ({entry.document_type})")
        return

    entries = (
        load_registry() if args.all else [get_document(args.document)]
    )

    for index, entry in enumerate(entries):
        try:
            # Only the first document may recreate the collection —
            # otherwise `--all --recreate` would wipe each document as
            # the next one is ingested.
            run_document(
                entry,
                start=args.start,
                recreate=args.recreate and index == 0,
            )

        except SystemExit:
            raise

        except Exception as error:
            print()
            print(f"[ERROR] {entry.document_id}: {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
