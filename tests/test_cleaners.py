"""
Generic cleaning operations.

`base.py` is what every document passes through before anything else sees it,
and until the Penal Code arrived it had no tests. These cover the two cuts,
because both fail in the same dangerous way: a marker that silently does not
match ships publisher matter as law.
"""

import re

import pytest

from data_pipeline.cleaners import employment_act as employment_act_cleaner
from data_pipeline.cleaners.base import (
    CleanerSpec,
    clean,
    cut_at_end,
    cut_to_start,
    normalize_lines,
    remove_page_markers,
)


# ---------------------------------------------------------
# A START MARKER MUST NOT MATCH THE FRONT MATTER IT IS CUTTING
#
# `cut_to_start` takes the FIRST match, so a pattern that also matches the
# cover keeps the cover - and the table of contents behind it - as law. The
# Employment Act has two decoys: "THE EMPLOYMENT ACT" on the cover and
# "Employment Act" on the licence page.
# ---------------------------------------------------------

EMPLOYMENT_FRONT_MATTER = """LAWS OF KENYA
THE EMPLOYMENT ACT
CAP. 226
Revised and published by the National Council for Law Reporting
Kenya
Employment Act
Cap. 226
Legislation as at 26 April 2024
FRBR URI: /akn/ke/act/2007/11/eng@2024-04-26
Employment Act (Cap. 226)
Contents
Part 1 – PRELIMINARY ....................................................... 1
EMPLOYMENT ACT
CAP. 226
Published in Kenya Gazette Vol. CIX-No. 81 on 9 November 2007
Assented to on 22 October 2007
Part 1 – PRELIMINARY
1.
Short title
"""


def test_employment_act_start_marker_skips_the_cover_and_the_contents():
    cut = cut_to_start(
        EMPLOYMENT_FRONT_MATTER,
        employment_act_cleaner.SPEC.start_pattern,
    )

    assert cut.startswith("EMPLOYMENT ACT\nCAP. 226\nPublished in Kenya")

    # The decoys and everything between them are gone.
    assert "LAWS OF KENYA" not in cut
    assert "THE EMPLOYMENT ACT" not in cut
    assert "Legislation as at" not in cut
    assert "Contents" not in cut


def test_employment_act_running_header_matches_the_real_shape():
    """Header and jurisdiction on consecutive lines, each with stray space."""

    page = "Employment Act (Cap. 226) \n Kenya\n35.\nTermination notice\n"

    stripped = re.sub(
        employment_act_cleaner.SPEC.running_header_pattern, "", page
    )

    assert stripped == "35.\nTermination notice\n"


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
