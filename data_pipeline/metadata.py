"""
Document metadata and checksums.

Generic replacement for the old Constitution-specific metadata builder.
Checksums are recorded at each pipeline stage so a later run can tell
whether a source document actually changed.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

from backend.app.core.config import PROJECT_ROOT
from data_pipeline.registry import DocumentEntry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_metadata(
    entry: DocumentEntry,
    chunk_count: int,
    as_at: str | None = None,
) -> dict:
    as_at = as_at or date.today().isoformat()

    files = {
        "raw": entry.raw_path,
        "extracted": entry.extracted_path,
        "cleaned": entry.cleaned_path,
    }

    return {
        "document_id": entry.document_id,
        "title": entry.title,
        "document_type": entry.document_type,
        "jurisdiction": entry.jurisdiction,
        "language": entry.language,

        "source": {
            "name": entry.source_name,
            "url": entry.source_url,
        },

        "version": entry.version,
        "effective_from": entry.effective_from,
        "effective_to": entry.effective_to,
        "in_force": entry.in_force,

        "files": {
            name: _relative(path)
            for name, path in files.items()
        },

        "checksums": {
            f"{name}_sha256": sha256(path)
            for name, path in files.items()
            if path.exists()
        },

        "processing": {
            "status": "ingested",
            "chunk_count": chunk_count,
            "as_at": as_at,
        },
    }


def write_metadata(entry: DocumentEntry, metadata: dict) -> None:
    entry.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    entry.metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
