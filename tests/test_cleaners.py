"""
Generic cleaning operations.

`base.py` is what every document passes through before anything else sees it,
and until the Penal Code arrived it had no tests. These cover the two cuts,
because both fail in the same dangerous way: a marker that silently does not
match ships publisher matter as law.
"""

import pytest

from data_pipeline.cleaners.base import (
    CleanerSpec,
    clean,
    cut_at_end,
    cut_to_start,
    normalize_lines,
    remove_page_markers,
)


# ---------------------------------------------------------
# CUTTING FRONT AND BACK MATTER
# ---------------------------------------------------------

def test_cut_to_start_drops_front_matter():
    text = "LAWS OF KENYA\ncover page\nPENAL CODE\nAssented to on 26 May 1930"

    assert cut_to_start(text, r"(?m)^PENAL CODE").startswith("PENAL CODE")


def test_cut_at_end_drops_trailing_matter():
    """
    The Penal Code ends with an alphabetical index the document itself
    disclaims as "not part of the Act".
    """

    text = (
        "398.\nPunishment of accessories\nSome operative text.\n"
        "INDEX - TO THE PENAL CODE\n"
        "definition of ......................\n256\n"
    )

    result = cut_at_end(text, r"(?m)^INDEX\s*[–—-]\s*TO THE PENAL CODE")

    assert result.endswith("Some operative text.\n")
    assert "definition of" not in result


def test_a_missing_start_marker_is_loud():
    with pytest.raises(ValueError, match="start marker"):
        cut_to_start("no marker here", r"(?m)^PREAMBLE")


def test_a_missing_end_marker_is_loud():
    """
    Silently keeping the text would ship an index as law - the same reason
    `cut_to_start` refuses rather than returning the document unchanged.
    """

    with pytest.raises(ValueError, match="end marker"):
        cut_at_end("no marker here", r"(?m)^INDEX")


def test_both_cuts_apply_together():
    text = "cover\nSTART\nthe law itself\nINDEX\nnot the law\n"

    spec = CleanerSpec(
        start_pattern=r"(?m)^START$",
        end_pattern=r"(?m)^INDEX$",
    )

    result = clean(text, spec)

    assert "the law itself" in result
    assert "cover" not in result
    assert "not the law" not in result


def test_no_patterns_keeps_the_document_whole():
    """Both cuts are optional; the Constitution needs no end marker."""

    result = clean("just the law\n", CleanerSpec())

    assert result.strip() == "just the law"


# ---------------------------------------------------------
# THE OPERATIONS EVERY DOCUMENT GETS
# ---------------------------------------------------------

def test_page_markers_are_removed():
    assert "PAGE" not in remove_page_markers("a\n--- PAGE 12 ---\nb")


def test_ligatures_are_normalised():
    assert "fine" in clean("a ﬁne of ten shillings\n", CleanerSpec())


def test_repeated_lines_collapse():
    """A running header that survives its own pattern would repeat."""

    assert normalize_lines("Kenya\nKenya\nthe law") == "Kenya\nthe law"


def test_verify_markers_catch_an_over_eager_regex():
    """Cheap insurance against a pattern quietly eating half the document."""

    spec = CleanerSpec(verify_markers={"section 1": "Short title"})

    with pytest.raises(ValueError, match="missing expected content"):
        clean("nothing relevant here\n", spec)
