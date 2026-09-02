"""
The document-type adapter seam.

Everything else in the pipeline is generic. Three things are not, and
cannot be: where a document's operative text begins, how its hierarchy
is parsed, and what "structurally complete" means for it.

Registering a new document type happens here and nowhere else. The
orchestrator does not need to change, and neither does any stage after
structure parsing - an Act's Part/Section and the Constitution's
Chapter/Article both become `path` plus `unit_type` in the IR.

Adding a type:

    1. write cleaners/<type>.py exposing a CleanerSpec named SPEC
    2. write structure/<type>.py exposing parse(text, entry) -> Document
    3. optionally write validators/<type>.py
    4. add one line to each table below
"""

from typing import Protocol

from data_pipeline.cleaners import constitution as constitution_cleaner
from data_pipeline.cleaners.base import CleanerSpec
from data_pipeline.ir import Document
from data_pipeline.registry import DocumentEntry
from data_pipeline.structure import constitution as constitution_structure
from data_pipeline.validators import constitution as constitution_validator


class StructureParser(Protocol):
    """
    The contract every structure adapter must satisfy.

    Stated explicitly so a mismatched parser fails at lookup with a
    clear message, rather than deep inside a pipeline run.
    """

    def __call__(self, text: str, entry: DocumentEntry) -> Document:
        ...


# ---------------------------------------------------------
# REGISTRATIONS
# ---------------------------------------------------------

CLEANERS: dict[str, CleanerSpec] = {
    "constitution": constitution_cleaner.SPEC,
}

STRUCTURE_PARSERS: dict[str, StructureParser] = {
    "constitution": constitution_structure.parse,
}

# Optional. A type with no entry here gets generic validation only.
VALIDATORS: dict[str, object] = {
    "constitution": constitution_validator,
}


# ---------------------------------------------------------
# LOOKUP
# ---------------------------------------------------------

def _resolve(table: dict, name: str, kind: str):
    if name not in table:
        known = ", ".join(sorted(table)) or "(none registered)"

        raise KeyError(
            f"No {kind} adapter named {name!r}. Registered: {known}. "
            f"Add it to data_pipeline/adapters.py."
        )

    return table[name]


def cleaner_spec(name: str) -> CleanerSpec:
    return _resolve(CLEANERS, name, "cleaner")


def structure_parser(name: str) -> StructureParser:
    parser = _resolve(STRUCTURE_PARSERS, name, "structure")

    if not callable(parser):
        raise TypeError(
            f"Structure adapter {name!r} is not callable; it must be "
            "parse(text, entry) -> Document."
        )

    return parser


def validator(name: str):
    """Type-specific validator, or None when the type has none."""

    return VALIDATORS.get(name)


def registered_types() -> list[str]:
    return sorted(
        set(CLEANERS) | set(STRUCTURE_PARSERS) | set(VALIDATORS)
    )
