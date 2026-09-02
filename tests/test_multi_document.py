"""
The multi-document guarantees.

Until now the pipeline had run on exactly one document of exactly one
type, with three adapter tables holding one entry each — and a
single-entry dictionary is indistinguishable from a hardcoded constant.
"Generic" was a claim about the shape of the code, not a demonstrated
property.

These tests demonstrate it without needing a PDF, Qdrant or an API key:
two synthetic documents of different shapes, and two versions of the
same document, pushed through the real generic chunker.
"""

from data_pipeline.adapters import (
    cleaner_spec,
    registered_types,
    structure_parser,
)
from data_pipeline.chunking.chunker import build_chunks
from data_pipeline.chunking.ids import point_id
from data_pipeline.ir import Document, Unit
from data_pipeline.validators.generic import validate_chunks

import pytest


def constitution_shaped(version: str = "2010") -> Document:
    """Chapter -> Article, like the Constitution."""

    return Document(
        document_id="constitution-of-kenya-2010",
        title="Constitution of Kenya, 2010",
        document_type="constitution",
        jurisdiction="Kenya",
        language="English",
        version=version,
        effective_from="2010-08-27",
        effective_to=None,
        in_force=True,
        source_name="Kenya Law",
        source_url=None,
        units=[
            Unit(
                unit_type="article",
                number="49",
                title="Rights of arrested persons",
                path=["Chapter Four", "Article 49"],
                text="An arrested person has the right to remain silent.",
            )
        ],
    )


def act_shaped(version: str = "2007") -> Document:
    """Part -> Section, like an Act. A different hierarchy entirely."""

    return Document(
        document_id="employment-act-2007",
        title="Employment Act, 2007",
        document_type="act",
        jurisdiction="Kenya",
        language="English",
        version=version,
        effective_from="2008-06-02",
        effective_to=None,
        in_force=True,
        source_name="Kenya Law",
        source_url=None,
        units=[
            Unit(
                unit_type="section",
                number="45",
                title="Unfair termination",
                path=["Part V", "Section 45"],
                text="No employer shall terminate a contract unfairly.",
            )
        ],
    )


def ids_of(document: Document) -> list[str]:
    return [c["chunk_id"] for c in build_chunks(document)]


# ---------------------------------------------------------
# TWO DOCUMENTS
# ---------------------------------------------------------

def test_two_documents_produce_disjoint_chunk_ids():
    assert not set(ids_of(constitution_shaped())) & set(ids_of(act_shaped()))


def test_two_documents_produce_disjoint_point_ids():
    """
    The original bug: positional ids meant document #2 overwrote #1.
    """

    constitution = {point_id(c) for c in ids_of(constitution_shaped())}
    act = {point_id(c) for c in ids_of(act_shaped())}

    assert not constitution & act


# ---------------------------------------------------------
# TWO VERSIONS OF ONE DOCUMENT
#
# The regression this whole refactor exists to prevent.
# ---------------------------------------------------------

def test_two_versions_of_one_act_do_not_collide():
    enacted = ids_of(act_shaped(version="2007"))
    amended = ids_of(act_shaped(version="2022"))

    assert not set(enacted) & set(amended)
    assert {point_id(c) for c in enacted} != {point_id(c) for c in amended}


def test_version_is_part_of_chunk_identity():
    assert ids_of(act_shaped(version="2022")) == [
        "employment-act-2007@v2022-part-v-section-45"
    ]


def test_same_document_and_version_is_reproducible():
    """Deterministic ids are what make re-ingestion idempotent."""

    assert ids_of(act_shaped()) == ids_of(act_shaped())
    assert point_id(ids_of(act_shaped())[0]) == point_id(
        ids_of(act_shaped())[0]
    )


# ---------------------------------------------------------
# A DIFFERENT HIERARCHY THROUGH THE SAME GENERIC CODE
# ---------------------------------------------------------

def test_act_hierarchy_flows_through_the_generic_chunker():
    """
    Part/Section must need no special handling. If this passes, the IR
    abstraction holds for a document type nothing was written for.
    """

    chunk = build_chunks(act_shaped())[0]

    assert chunk["unit_type"] == "section"
    assert chunk["citation"] == "Part V — Section 45"
    assert chunk["path"] == ["Part V", "Section 45"]
    assert validate_chunks([chunk]) == []


def test_both_shapes_pass_the_same_validator():
    for document in (constitution_shaped(), act_shaped()):
        assert validate_chunks(build_chunks(document)) == []


# ---------------------------------------------------------
# ADAPTER CONTRACT
# ---------------------------------------------------------

def test_unknown_document_type_fails_with_the_known_list():
    with pytest.raises(KeyError, match="Registered"):
        structure_parser("no-such-type")

    with pytest.raises(KeyError, match="Registered"):
        cleaner_spec("no-such-type")


def test_registered_types_are_discoverable():
    assert "constitution" in registered_types()
