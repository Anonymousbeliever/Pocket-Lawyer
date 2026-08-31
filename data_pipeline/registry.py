"""
The document registry.

Loads `documents.yaml` and resolves each entry's adapter names to the
callables that clean, structure and validate that document type.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.app.core.config import PROJECT_ROOT


REGISTRY_PATH = Path(__file__).resolve().parent / "documents.yaml"


@dataclass
class DocumentEntry:
    """One registered source document."""

    document_id: str
    title: str
    document_type: str
    jurisdiction: str
    language: str

    version: str
    effective_date: str | None
    in_force: bool

    source_name: str
    source_url: str | None

    raw_path: Path

    cleaner: str
    structure: str
    validator: str

    # ---------------------------------------------------------
    # DERIVED PATHS
    # ---------------------------------------------------------

    @property
    def slug(self) -> str:
        """Filename stem used for this document's pipeline artifacts."""

        return self.document_id

    @property
    def extracted_path(self) -> Path:
        return PROJECT_ROOT / "data" / "extracted" / f"{self.slug}.txt"

    @property
    def cleaned_path(self) -> Path:
        return PROJECT_ROOT / "data" / "cleaned" / f"{self.slug}.txt"

    @property
    def structure_path(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / f"{self.slug}_structure.json"

    @property
    def chunks_path(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / f"{self.slug}_chunks.json"

    @property
    def metadata_path(self) -> Path:
        return PROJECT_ROOT / "data" / "metadata" / f"{self.slug}.json"


def _parse_entry(raw: dict) -> DocumentEntry:
    required = [
        "document_id",
        "title",
        "document_type",
        "jurisdiction",
        "language",
        "version",
        "in_force",
        "raw_path",
        "cleaner",
        "structure",
        "validator",
    ]

    missing = [field for field in required if field not in raw]

    if missing:
        raise ValueError(
            f"Registry entry {raw.get('document_id', '<unknown>')} "
            f"is missing: {', '.join(missing)}"
        )

    source = raw.get("source") or {}

    return DocumentEntry(
        document_id=raw["document_id"],
        title=raw["title"],
        document_type=raw["document_type"],
        jurisdiction=raw["jurisdiction"],
        language=raw["language"],
        version=str(raw["version"]),
        effective_date=raw.get("effective_date"),
        in_force=bool(raw["in_force"]),
        source_name=source.get("name", "Unknown"),
        source_url=source.get("url"),
        raw_path=PROJECT_ROOT / raw["raw_path"],
        cleaner=raw["cleaner"],
        structure=raw["structure"],
        validator=raw["validator"],
    )


def load_registry(path: Path = REGISTRY_PATH) -> list[DocumentEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")

    raw_entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []

    if not isinstance(raw_entries, list):
        raise ValueError(
            "documents.yaml must contain a list of document entries."
        )

    entries = [_parse_entry(raw) for raw in raw_entries]

    ids = [entry.document_id for entry in entries]

    duplicates = sorted({i for i in ids if ids.count(i) > 1})

    if duplicates:
        raise ValueError(
            "Duplicate document_id in registry: "
            + ", ".join(duplicates)
        )

    return entries


def get_document(document_id: str) -> DocumentEntry:
    for entry in load_registry():
        if entry.document_id == document_id:
            return entry

    known = ", ".join(e.document_id for e in load_registry())

    raise KeyError(
        f"Unknown document_id: {document_id}. Registered: {known}"
    )
