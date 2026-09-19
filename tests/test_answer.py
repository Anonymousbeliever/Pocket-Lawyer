"""
The structured answer contract.

Polarity is fixed by construction here: the model fills a `verdict`
field and code writes the opening word, so the stated verdict and the
explanation cannot disagree.
"""

import json

import pytest

from backend.app.ai.answer import (
    ANSWER_SCHEMA,
    INSUFFICIENT_ANSWER,
    LegalAnswer,
    cited_sources,
    render,
    verify_citations,
)


ARTICLE_49 = "constitution-of-kenya-2010@v2010-chapter-four-article-49"
ARTICLE_51 = "constitution-of-kenya-2010@v2010-chapter-four-article-51"


def sources() -> list[dict]:
    return [
        {"chunk_id": ARTICLE_49, "unit_title": "Rights of arrested persons"},
        {"chunk_id": ARTICLE_51, "unit_title": "Rights of persons detained"},
    ]


def answer(**overrides) -> LegalAnswer:
    defaults = dict(
        sufficient=True,
        question_type="polar",
        verdict="no",
        explanation=(
            "Article 49(1)(a) gives an arrested person the right to be "
            "informed promptly of the reason for the arrest."
        ),
        qualifications=[],
        cited_chunk_ids=[ARTICLE_49],
    )
    defaults.update(overrides)
    return LegalAnswer(**defaults)


# ---------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------

def test_schema_satisfies_openai_strict_mode():
    """Strict mode requires every property listed in `required`."""

    properties = set(ANSWER_SCHEMA["properties"])
    required = set(ANSWER_SCHEMA["required"])

    assert properties == required
    assert ANSWER_SCHEMA["additionalProperties"] is False


def test_verdict_enum_covers_non_polar_questions():
    verdicts = ANSWER_SCHEMA["properties"]["verdict"]["enum"]

    assert "not_applicable" in verdicts
    assert {"yes", "no", "it_depends"} <= set(verdicts)


# ---------------------------------------------------------
# POLARITY — THE REGRESSION
# ---------------------------------------------------------

def test_polar_no_answer_opens_with_no():
    """
    Regression: "Can police arrest me without telling me why?" was
    answered "Yes, the police cannot arrest you without telling you
    why." The opening word is now written by code from the verdict
    field, so it cannot contradict the explanation.
    """

    text = render(answer(verdict="no"))

    assert text.startswith("No.")
    assert not text.startswith("Yes")


def test_polar_yes_answer_opens_with_yes():
    text = render(answer(verdict="yes"))

    assert text.startswith("Yes.")


def test_it_depends_is_rendered():
    text = render(answer(verdict="it_depends"))

    assert text.startswith("It depends.")


def test_open_question_gets_no_polarity_prefix():
    text = render(
        answer(question_type="open", verdict="not_applicable")
    )

    assert not text.startswith(("Yes", "No", "It depends"))
    assert text.startswith("Article 49")


def test_qualifications_are_rendered():
    text = render(
        answer(qualifications=["Subject to the twenty-four hour limit."])
    )

    assert "Important qualifications:" in text
    assert "- Subject to the twenty-four hour limit." in text


# ---------------------------------------------------------
# REFUSAL
# ---------------------------------------------------------

def test_insufficient_answer_renders_the_refusal():
    text = render(
        answer(
            sufficient=False,
            verdict="not_applicable",
            explanation="The sources do not address divorce.",
        )
    )

    assert text.startswith(INSUFFICIENT_ANSWER)
    assert "do not address divorce" in text


def test_refusal_never_opens_with_a_verdict():
    """A false `sufficient` wins even if verdict was filled in."""

    text = render(answer(sufficient=False, verdict="no"))

    assert not text.startswith("No.")


# ---------------------------------------------------------
# CITATION VERIFICATION
# ---------------------------------------------------------

def test_valid_citations_are_kept():
    result = verify_citations(answer(), sources())

    assert result.cited_chunk_ids == [ARTICLE_49]
    assert result.unverified_citations == []


def test_fabricated_citations_are_removed():
    """Caught deterministically, with no second model call."""

    result = verify_citations(
        answer(cited_chunk_ids=[ARTICLE_49, "employment-act-2007-section-45"]),
        sources(),
    )

    assert result.cited_chunk_ids == [ARTICLE_49]
    assert result.unverified_citations == ["employment-act-2007-section-45"]


def test_cited_sources_returns_only_what_was_cited():
    result = cited_sources(answer(cited_chunk_ids=[ARTICLE_51]), sources())

    assert len(result) == 1
    assert result[0]["chunk_id"] == ARTICLE_51


def test_cited_sources_preserves_citation_order():
    result = cited_sources(
        answer(cited_chunk_ids=[ARTICLE_51, ARTICLE_49]),
        sources(),
    )

    assert [s["chunk_id"] for s in result] == [ARTICLE_51, ARTICLE_49]


def test_no_citations_yields_no_sources():
    assert cited_sources(answer(cited_chunk_ids=[]), sources()) == []


# ---------------------------------------------------------
# PARSING
# ---------------------------------------------------------

def test_parses_a_well_formed_response():
    payload = json.dumps(
        {
            "sufficient": True,
            "question_type": "polar",
            "verdict": "no",
            "explanation": "  Article 49 applies.  ",
            "qualifications": ["A limit."],
            "cited_chunk_ids": [ARTICLE_49],
        }
    )

    result = LegalAnswer.from_json(payload)

    assert result.verdict == "no"
    assert result.explanation == "Article 49 applies."
    assert result.qualifications == ["A limit."]


def test_invalid_json_is_rejected_clearly():
    with pytest.raises(RuntimeError, match="invalid JSON"):
        LegalAnswer.from_json("not json at all")


def test_missing_fields_are_reported():
    with pytest.raises(RuntimeError, match="missing required fields"):
        LegalAnswer.from_json(json.dumps({"sufficient": True}))

