"""
Contextual passage text used for embedding and reranking.

The header must carry the topic label that a chunk's body sometimes
lacks entirely, without ever altering the content sent to the LLM.
"""

from backend.app.core.passage import contextual_text


ARTICLE_16 = {
    "title": "Constitution of Kenya, 2010",
    "citation": "Chapter Three — Article 16",
    "unit_title": "Dual citizenship",
    "content": (
        "A citizen by birth does not lose citizenship by acquiring "
        "the citizenship of another country."
    ),
}

SCHEDULE = {
    "title": "Constitution of Kenya, 2010",
    "citation": "SIXTH SCHEDULE",
    "unit_title": "SIXTH SCHEDULE",
    "content": "Transitional and consequential provisions.",
}


def test_title_reaches_the_embedded_text():
    """
    The regression this exists for: Article 16's body never contains
    the word "dual", so embedding the body alone made it unfindable.
    """

    assert "dual" not in ARTICLE_16["content"].lower()
    assert "dual citizenship" in contextual_text(ARTICLE_16).lower()


def test_content_is_preserved_verbatim():
    text = contextual_text(ARTICLE_16)

    assert text.endswith(ARTICLE_16["content"])


def test_header_precedes_the_content():
    text = contextual_text(ARTICLE_16)

    assert text.startswith("Constitution of Kenya, 2010")
    assert "Chapter Three — Article 16: Dual citizenship" in text


def test_schedule_heading_is_not_duplicated():
    """'SIXTH SCHEDULE: SIXTH SCHEDULE' helps nobody."""

    text = contextual_text(SCHEDULE)

    assert "SIXTH SCHEDULE: SIXTH SCHEDULE" not in text
    assert "SIXTH SCHEDULE" in text


def test_missing_metadata_degrades_to_bare_content():
    assert contextual_text({"content": "some text"}) == "some text"


def test_missing_content_does_not_crash():
    assert "Dual citizenship" in contextual_text(
        {"citation": "Chapter Three — Article 16",
         "unit_title": "Dual citizenship"}
    )


def test_citation_only_chunk_keeps_its_citation():
    text = contextual_text(
        {"citation": "Chapter Four — Article 49", "content": "body"}
    )

    assert text.startswith("Chapter Four — Article 49")
    assert text.endswith("body")
