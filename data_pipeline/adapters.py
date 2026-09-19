"""
The document-type adapter seam.

Everything else in the pipeline is generic. Three things are not, and
cannot be: where a document's operative text begins, how its hierarchy
is parsed, and what "structurally complete" means for it.

Registering a new document type happens here and nowhere else. The
orchestrator does not need to change, and neither does any stage after
structure parsing - an Act's Part/Section and the Constitution's
Chapter/Article both become `path` plus `unit_type` in the IR.

Adding a document:

    1. write cleaners/<document>.py exposing a CleanerSpec named SPEC
    2. reuse an existing structure parser, or write structure/<type>.py
       exposing parse(text, entry) -> Document for a new hierarchy
    3. write validators/<document>.py, or use "none" to opt out
    4. add one line to each table below

Note the asymmetry, learned from adding the second document. The
structure parser is genuinely per-*type* and reusable: every Kenya Law
Act shares the Part -> Section shape, so "act" is written once. The
cleaner spec is per-*document*, because where the operative text begins
and which running header repeats both name the document itself.
"""

from typing import Protocol

from data_pipeline.cleaners import constitution as constitution_cleaner
from data_pipeline.cleaners import (
    criminal_procedure_code as criminal_procedure_code_cleaner,
)
from data_pipeline.cleaners import penal_code as penal_code_cleaner
from data_pipeline.cleaners.base import CleanerSpec
from data_pipeline.ir import Document
from data_pipeline.registry import DocumentEntry
from data_pipeline.structure import act as act_structure
from data_pipeline.structure import constitution as constitution_structure
from data_pipeline.validators import constitution as constitution_validator
from data_pipeline.validators import (
    criminal_procedure_code as criminal_procedure_code_validator,
)
from data_pipeline.validators import penal_code as penal_code_validator


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

# Cleaner specs are per-document, not per-type: their patterns embed the
# document's own title and running header, and a CleanerSpec is static
# data with no access to the registry entry.
CLEANERS: dict[str, CleanerSpec] = {
    "constitution": constitution_cleaner.SPEC,
    "criminal-procedure-code": criminal_procedure_code_cleaner.SPEC,
    "penal-code": penal_code_cleaner.SPEC,
}

# Structure parsers are per-type and reusable. "act" handles the
# Part -> Section shape Kenya Law uses for every consolidated Act.
STRUCTURE_PARSERS: dict[str, StructureParser] = {
    "constitution": constitution_structure.parse,
    "act": act_structure.parse,
}

# Optional. A type with no entry here gets generic validation only.
VALIDATORS: dict[str, object] = {
    "constitution": constitution_validator,
    "criminal-procedure-code": criminal_procedure_code_validator,
    "penal-code": penal_code_validator,
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
    """
    Type-specific validator, or None when the type has none.

    `registry.py` makes `validator` a required field, so an unregistered
    name here is a typo rather than a deliberate opt-out — and silently
    returning None would downgrade the document to generic validation
    without saying so. The sentinel "none" is how a type opts out on
    purpose.
    """

    if name in VALIDATORS:
        return VALIDATORS[name]

    if name in ("none", "generic"):
        return None

    raise KeyError(
        f"No validator adapter named {name!r}. Registered: "
        f"{', '.join(sorted(VALIDATORS))}. Use 'none' to opt out "
        "deliberately, or add it to data_pipeline/adapters.py."
    )


def registered_types() -> list[str]:
    return sorted(
        set(CLEANERS) | set(STRUCTURE_PARSERS) | set(VALIDATORS)
    )
