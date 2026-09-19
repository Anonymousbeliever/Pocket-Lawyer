"""
The Act structure adapter — Part -> Section.

The second document type, and the first real test of the claim that a
new hierarchy needs a new structure adapter and nothing else.

Most of these run on a short inline fixture, so they need no PDF, no
Qdrant and no API key. The ones that check the real Criminal Procedure
Code skip when the pipeline has not been run on this checkout.
"""

import pytest

from data_pipeline.adapters import (
    cleaner_spec,
    registered_types,
    structure_parser,
    validator,
)
from data_pipeline.chunking.chunker import build_chunks
from data_pipeline.ir import Document, Unit
from data_pipeline.registry import get_document
from data_pipeline.structure import act as act_structure
from data_pipeline.validators import generic as generic_validator
from data_pipeline.validators import (
    criminal_procedure_code as cpc_validator,
)


DOCUMENT_ID = "criminal-procedure-code"


# A miniature Act with every shape the real one contains: an opening
# amendment log, Parts with an en dash, a cross-heading, a lettered
# section, sub-paragraphs, a repealed section, and a schedule.
SAMPLE = """CRIMINAL PROCEDURE CODE
CAP. 75
Published in Kenya Gazette Vol. XXXII-No. 27 on 29 May 1930
[Amended by Criminal Procedure (Amendment) Act, 1930 (Act No. 15 of 1930) on 10 June 1930]
Part I – PRELIMINARY
1.
Short title
This Act may be cited as the Criminal Procedure Code.
2.
Interpretation
In this Code, unless the context otherwise requires—
"cognizable offence" means an offence for which a police officer may arrest without warrant.
Part III – GENERAL PROVISIONS ARREST, ESCAPE AND RETAKING
Arrest Generally
21.
Arrest
(1)
In making an arrest the police officer shall actually touch the body of the person to be arrested.
(2)
If the person forcibly resists, the officer may use all means necessary to effect the arrest.
(a)
reasonable force may be used;
(b)
nothing in this section gives a right to cause the death of a person.
36A.
Remand by court
(1)
A police officer may apply to a court for an order to hold a suspect.
(i)
the nature of the offence;
(ii)
the general nature of the evidence.
222.
[Repealed by Act No. 33 of 1963, 1st Sch.]
Part IXA – VICTIM IMPACT STATEMENTS
329A.
Interpretation
In this Part, "victim" has the meaning assigned to it in the Victim Protection Act.
FIRST SCHEDULE
OFFENCES UNDER THE PENAL CODE (Cap. 63)
Section
Offence
"""


@pytest.fixture(scope="module")
def entry():
    return get_document(DOCUMENT_ID)


@pytest.fixture(scope="module")
def sample(entry):
    return act_structure.parse(SAMPLE, entry)


def unit(document, number):
    for candidate in document.units:
        if candidate.number == number:
            return candidate

    return None


# ---------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------

def test_act_and_cpc_are_registered():
    assert "act" in registered_types()
    assert "criminal-procedure-code" in registered_types()

    assert callable(structure_parser("act"))
    assert cleaner_spec("criminal-procedure-code").start_pattern


def test_unknown_validator_is_not_silently_ignored():
    """
    The registry makes `validator` required, so an unregistered name is
    a typo. Returning None would quietly downgrade the document to
    generic validation only.
    """

    with pytest.raises(KeyError, match="Registered"):
        validator("no-such-validator")

    assert validator("none") is None


# ---------------------------------------------------------
# HIERARCHY
# ---------------------------------------------------------

def test_sections_are_found_under_their_part(sample):
    assert unit(sample, "1").path == ["Part I", "Section 1"]
    assert unit(sample, "21").path == ["Part III", "Section 21"]


def test_part_label_not_part_title_is_used_in_the_path(sample):
    """
    Part titles contain en dashes and commas. Keeping them out of the
    path keeps them out of `Unit.slug()` and therefore out of chunk ids.
    """

    for candidate in sample.units:
        assert "–" not in candidate.path[0]
        assert "," not in candidate.path[0]

    assert unit(sample, "21").path[0] == "Part III"


def test_lettered_sections_are_not_missed(sample):
    """36A, 137A-137N and 379A are real sections a digits-only pattern drops."""

    assert unit(sample, "36A") is not None
    assert unit(sample, "36A").title == "Remand by court"
    assert unit(sample, "329A").path == ["Part IXA", "Section 329A"]


def test_section_title_is_taken_from_the_following_line(sample):
    assert unit(sample, "1").title == "Short title"
    assert unit(sample, "21").title == "Arrest"


def test_section_title_is_not_repeated_in_the_content(sample):
    assert not unit(sample, "1").text.startswith("Short title")


# ---------------------------------------------------------
# WHAT IS DELIBERATELY DROPPED
# ---------------------------------------------------------

def test_opening_amendment_log_is_dropped(sample):
    """Gazette details and "[Amended by ...]" are history, not law."""

    for candidate in sample.units:
        assert "Amended by" not in candidate.text
        assert "Kenya Gazette" not in candidate.text


def test_repealed_sections_are_dropped(sample):
    assert unit(sample, "222") is None


def test_schedules_are_not_emitted(sample):
    """
    Act schedules are tables; line-based extraction shreds them into one
    line per cell. Parsing stops at the first schedule heading.
    """

    assert sample.unit_count("schedule") == 0

    for candidate in sample.units:
        assert "OFFENCES UNDER THE PENAL CODE" not in candidate.text


def test_cross_headings_are_not_attributed_to_the_previous_section(sample):
    """
    "Arrest Generally" sits between section 2 and section 21. Without
    detection it lands at the end of section 2's text.
    """

    assert "Arrest Generally" not in unit(sample, "2").text
    assert "Arrest Generally" not in unit(sample, "21").text


# ---------------------------------------------------------
# THE CROSS-HEADING RULE, BOTH WAYS
# ---------------------------------------------------------

def test_a_heading_before_a_section_is_recognised():
    assert act_structure._is_cross_heading("SEARCH WARRANTS", "118.")
    assert act_structure._is_cross_heading("Arrest Generally", "21.")


def test_running_text_before_a_section_is_kept():
    """The rule must not eat the last line of the preceding section."""

    assert not act_structure._is_cross_heading(
        "the court shall consider any objection.", "118."
    )

    # Amendment annotations also sit immediately before section numbers.
    assert not act_structure._is_cross_heading(
        "[Act No. 5 of 2003, s. 61.]", "118."
    )

    # A surviving page number is not a heading.
    assert not act_structure._is_cross_heading("118", "119.")

    # Only lines that actually precede a section qualify.
    assert not act_structure._is_cross_heading(
        "SEARCH WARRANTS", "and thereafter the court may"
    )


# ---------------------------------------------------------
# SUBSECTIONS AND PARAGRAPHS
# ---------------------------------------------------------

def test_subsections_are_rebuilt_with_their_numbering(sample):
    text = unit(sample, "21").text

    assert text.startswith("(1) In making an arrest")
    assert "(2) If the person forcibly resists" in text
    assert "(a) reasonable force may be used;" in text


def test_sub_paragraphs_flatten_rather_than_disappear(sample):
    """
    (i)/(ii) collapse to the same level as (a)/(b) — the hierarchy below
    a subsection is not separately addressable, matching the
    Constitution adapter. The text must still survive intact.
    """

    text = unit(sample, "36A").text

    assert "(i) the nature of the offence;" in text
    assert "(ii) the general nature of the evidence." in text


# ---------------------------------------------------------
# THE GENERIC PIPELINE DOWNSTREAM
# ---------------------------------------------------------

def test_act_units_chunk_without_any_act_specific_code(sample, entry):
    chunks = build_chunks(sample)

    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    expected = f"{entry.document_id}@v{entry.version}-part-iii-section-21"

    assert expected in by_id
    assert by_id[expected]["citation"] == "Part III — Section 21"
    assert by_id[expected]["unit_type"] == "section"


def test_chunk_ids_are_free_of_punctuation(sample):
    """
    `Unit.slug()` only lowercases and replaces spaces, so a Part title
    in the path would leak commas and en dashes into ids and into the
    uuid5 they seed.
    """

    for chunk in build_chunks(sample):
        assert set(chunk["chunk_id"]) <= set(
            "abcdefghijklmnopqrstuvwxyz0123456789-@."
        )


def test_cpc_and_constitution_chunk_ids_are_disjoint(sample):
    """
    The regression the whole ingestion refactor exists to prevent: a
    second document must not overwrite the first.
    """

    constitution = Document(
        document_id="constitution-of-kenya-2010",
        title="Constitution of Kenya, 2010",
        document_type="constitution",
        jurisdiction="Kenya",
        language="English",
        version="2010",
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

    cpc_ids = {chunk["chunk_id"] for chunk in build_chunks(sample)}

    constitution_ids = {
        chunk["chunk_id"] for chunk in build_chunks(constitution)
    }

    assert not cpc_ids & constitution_ids


# ---------------------------------------------------------
# FRONT-MATTER DETECTION
#
# Found by this document: the dot-leader check was Constitution-only
# and unanchored. Acts carry statutory forms whose fill-in blanks look
# identical mid-line, so moving the check into the generic validator
# without anchoring it failed every Act that has forms.
# ---------------------------------------------------------

def test_toc_leaders_are_still_detected():
    leaked = "Contents\nPart I - PRELIMINARY .................... 3\n"

    errors = generic_validator.validate_cleaned_text(leaked)

    assert any("TOC page leaders" in error for error in errors)
    assert any("table of contents" in error for error in errors)


def test_statutory_form_blanks_are_not_mistaken_for_a_toc():
    form = (
        "on the ................ day of ................ 20......, "
        "at the ................ held before me.\n"
    )

    errors = generic_validator.validate_cleaned_text(form)

    assert not any("TOC page leaders" in error for error in errors)


# ---------------------------------------------------------
# THE REAL DOCUMENT
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def real_document(entry):
    if not entry.cleaned_path.exists():
        pytest.skip(
            "cleaned artifact missing; run "
            f"`python -m data_pipeline.run --document {DOCUMENT_ID}` first"
        )

    text = entry.cleaned_path.read_text(encoding="utf-8")

    return act_structure.parse(text, entry)


def test_real_cpc_parses_to_the_expected_shape(real_document):
    assert cpc_validator.validate_document(real_document) == []


def test_real_cpc_text_passes_its_validator(entry):
    if not entry.cleaned_path.exists():
        pytest.skip("cleaned artifact missing")

    text = entry.cleaned_path.read_text(encoding="utf-8")

    assert cpc_validator.validate_cleaned_text(text) == []
