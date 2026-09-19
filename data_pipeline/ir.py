"""
The common intermediate representation for a legal document.

Every structure parser converts its document type into this shape, and
every stage after structure parsing (chunking, embedding, storage,
validation) works only on this shape.

That is the whole point: the Constitution's Chapter -> Article and an
Act's Part -> Section both collapse into `path` plus `unit_type`, so a
new document type needs a new structure adapter and nothing else.
"""

from dataclasses import dataclass, field


@dataclass
class Unit:
    """
    One citable legal unit — an article, a section, a schedule.

    `path` is the human-readable breadcrumb used for citation, e.g.
    ["Chapter Four", "Article 49"]. It is what lets the answer say
    where a passage came from without the chunker knowing anything
    about constitutions.
    """

    unit_type: str
    number: str
    title: str
    path: list[str]
    text: str

    def slug(self) -> str:
        """Stable, readable identifier fragment for this unit."""

        parts = [
            part.lower().replace(" ", "-")
            for part in self.path
        ]

        return "-".join(parts)


@dataclass
class Document:
    """A single legal document and its citable units."""

    document_id: str
    title: str
    document_type: str
    jurisdiction: str
    language: str

    # Temporal validity. `effective_to` is None while a version is the
    # operative one; setting it, with in_force False, is how a
    # superseded version stays queryable for "what did the law say in
    # 2015" without being returned by default.
    version: str
    effective_from: str | None
    effective_to: str | None
    in_force: bool

    source_name: str
    source_url: str | None

    units: list[Unit] = field(default_factory=list)

    def unit_count(self, unit_type: str | None = None) -> int:
        if unit_type is None:
            return len(self.units)

        return sum(
            1
            for unit in self.units
            if unit.unit_type == unit_type
        )


def document_from_entry(entry, units: list[Unit]) -> Document:
    """
    Build a Document from a registry entry and the units a parser found.

    Every structure adapter ends the same way: copy eleven fields off the
    entry and attach the units. Doing that by hand in each adapter means
    a new field has to be remembered in N places, and a forgotten one
    fails silently — the document simply loses a payload field. This is
    the one place to change.
    """

    return Document(
        document_id=entry.document_id,
        title=entry.title,
        document_type=entry.document_type,
        jurisdiction=entry.jurisdiction,
        language=entry.language,
        version=entry.version,
        effective_from=entry.effective_from,
        effective_to=entry.effective_to,
        in_force=entry.in_force,
        source_name=entry.source_name,
        source_url=entry.source_url,
        units=units,
    )
