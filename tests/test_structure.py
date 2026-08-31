"""
Structure adapter and registry.

The Constitution parse is checked against the real cleaned artifact
when it exists — 264 articles and 6 schedules, none missing. These
tests skip rather than fail on a fresh checkout where the pipeline has
not been run yet.
"""

import pytest

from data_pipeline.registry import get_document, load_registry
from data_pipeline.structure import constitution as constitution_structure
from data_pipeline.validators import constitution as constitution_validator


DOCUMENT_ID = "constitution-of-kenya-2010"


@pytest.fixture(scope="module")
def entry():
    return get_document(DOCUMENT_ID)


@pytest.fixture(scope="module")
def document(entry):
    if not entry.cleaned_path.exists():
        pytest.skip(
            "cleaned artifact missing; run "
            "`python -m data_pipeline.run --document "
            f"{DOCUMENT_ID}` first"
        )

    text = entry.cleaned_path.read_text(encoding="utf-8")

    return constitution_structure.parse(text, entry)


# ---------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------

def test_registry_loads():
    entries = load_registry()

    assert entries
    assert all(entry.document_id for entry in entries)


def test_registry_rejects_unknown_document():
    with pytest.raises(KeyError):
        get_document("no-such-document")


def test_registry_derives_artifact_paths(entry):
    assert entry.cleaned_path.name == f"{DOCUMENT_ID}.txt"
    assert entry.chunks_path.name == f"{DOCUMENT_ID}_chunks.json"


# ---------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------

def test_all_264_articles_are_parsed(document):
    assert document.unit_count("article") == 264


def test_all_6_schedules_are_parsed(document):
    assert document.unit_count("schedule") == 6


def test_no_article_number_is_missing(document):
    numbers = sorted(
        int(unit.number)
        for unit in document.units
        if unit.unit_type == "article"
    )

    assert numbers == list(range(1, 265))


def test_document_validator_passes(document):
    assert constitution_validator.validate_document(document) == []


def test_article_49_is_intact(document):
    """The worked example used throughout the project."""

    article = next(
        unit
        for unit in document.units
        if unit.unit_type == "article" and unit.number == "49"
    )

    assert article.title == "Rights of arrested persons"
    assert article.path == ["Chapter Four", "Article 49"]

    # Every paragraph of 49(1), including (e) which is easy to drop.
    for marker in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]:
        assert marker in article.text

    assert "to remain silent" in article.text
    assert "twenty-four hours" in article.text


def test_units_carry_citation_paths(document):
    assert all(unit.path for unit in document.units)


def test_unit_slug_is_citation_shaped(document):
    article = next(
        unit
        for unit in document.units
        if unit.unit_type == "article" and unit.number == "49"
    )

    assert article.slug() == "chapter-four-article-49"
