"""
The document registry.

Loads `documents.yaml` and resolves each entry's adapter names to the
callables that clean, structure and validate that document type.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.app.core.config import DOCUMENTS_DIR, PROJECT_ROOT


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
    effective_from: str | None
    effective_to: str | None
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
        """
        Identity of this document *version*.

        Mirrors the chunk id prefix exactly, so a chunk id can always be
        traced back to the directory its artifacts live in. Two versions
        of the same Act therefore never share a directory - which
        matters most for metadata.json, the provenance record.
        """

        return f"{self.document_id}@v{self.version}"

    @property
    def document_dir(self) -> Path:
        return DOCUMENTS_DIR / self.slug

    @property
    def extracted_path(self) -> Path:
        return self.document_dir / "extracted.txt"

    @property
    def cleaned_path(self) -> Path:
        return self.document_dir / "cleaned.txt"

    @property
    def structure_path(self) -> Path:
        return self.document_dir / "structure.json"

    @property
    def chunks_path(self) -> Path:
        return self.document_dir / "chunks.json"

    @property
    def metadata_path(self) -> Path:
        return self.document_dir / "metadata.json"


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
        effective_from=raw.get("effective_from"),
        effective_to=raw.get("effective_to"),
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

    # Uniqueness is per document *version*, not per document. Two
    # versions of the same Act are two entries sharing a document_id.
    slugs = [entry.slug for entry in entries]

    duplicates = sorted({s for s in slugs if slugs.count(s) > 1})

    if duplicates:
        raise ValueError(
            "Duplicate document version in registry: "
            + ", ".join(duplicates)
        )

    # Exactly one version of a document may be the operative one.
    for document_id in {entry.document_id for entry in entries}:
        in_force = [
            entry.slug
            for entry in entries
            if entry.document_id == document_id and entry.in_force
        ]

        if len(in_force) > 1:
            raise ValueError(
                f"More than one version of {document_id} is marked "
                "in_force: " + ", ".join(sorted(in_force))
            )

    return entries


def get_document(reference: str) -> DocumentEntry:
    """
    Resolve a document reference to one registered version.

    Accepts either an exact version slug ("employment-act-2007@v2022")
    or a bare document id, which resolves to the version currently in
    force.
    """

    entries = load_registry()

    for entry in entries:
        if entry.slug == reference:
            return entry

    matches = [e for e in entries if e.document_id == reference]

    if len(matches) == 1:
        return matches[0]

    if matches:
        current = [e for e in matches if e.in_force]

        if len(current) == 1:
            return current[0]

        raise KeyError(
            f"{reference} has several registered versions and none is "
            "uniquely in force. Name one: "
            + ", ".join(sorted(e.slug for e in matches))
        )

    known = ", ".join(sorted(e.slug for e in entries))

    raise KeyError(f"Unknown document: {reference}. Registered: {known}")
