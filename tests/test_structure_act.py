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
from data_pipeline.validators import employment_act as employment_validator
from data_pipeline.validators import (
    law_of_succession_act as succession_validator,
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
# THE OPTIONAL CHAPTER LEVEL
#
# The Penal Code is Part -> Chapter -> Section; the Criminal Procedure Code is
# Part -> Section. Chapter support is additive, so both must work through the
# same parser.
#
# Note the heading shape differs from a Part's: a Part is one line with a dash,
# a Chapter splits across two.
# ---------------------------------------------------------

CHAPTERED = """PENAL CODE
CAP. 63
Assented to on 26 May 1930
[Amended by Penal Code (Amendment) Ordinance, 1930 (Act No. 45 of 1930)]
Part I – GENERAL PROVISIONS
Chapter I
PRELIMINARY
1.
Short title
This Code may be cited as the Penal Code.
Part II – CRIMES
Chapter XV
OFFENCES AGAINST MORALITY
139.
[Deleted by Act No. 3 of 2006, 2nd Sch.]
Chapter XIX
MURDER AND MANSLAUGHTER
203.
Murder
Any person who of malice aforethought causes the death of another person by an
unlawful act or omission is guilty of murder.
204.
Punishment of murder
Any person convicted of murder shall be sentenced to death.
"""


@pytest.fixture(scope="module")
def chaptered(entry):
    return act_structure.parse(CHAPTERED, entry)


def test_the_chapter_sits_between_part_and_section(chaptered):
    assert unit(chaptered, "203").path == [
        "Part II",
        "Chapter XIX",
        "Section 203",
    ]


def test_chapters_reset_across_parts(chaptered):
    assert unit(chaptered, "1").path == ["Part I", "Chapter I", "Section 1"]


def test_the_chapter_title_does_not_leak_into_the_section(chaptered):
    """
    "OFFENCES AGAINST MORALITY" sits on its own line after the heading. With
    no section open it must be dropped, not accumulated.
    """

    for candidate in chaptered.units:
        assert "MURDER AND MANSLAUGHTER" not in candidate.text
        assert "OFFENCES AGAINST MORALITY" not in candidate.text


def test_a_chapter_heading_is_not_mistaken_for_a_part(chaptered):
    """Both are Roman numerals; only the Part carries a dash and a title."""

    labels = {unit.path[0] for unit in chaptered.units}

    assert labels == {"Part I", "Part II"}


def test_chapters_keep_chunk_ids_clean(chaptered):
    """
    An extra path level must not smuggle punctuation into `Unit.slug()` -
    labels only, never the chapter's title.
    """

    for chunk in build_chunks(chaptered):
        assert set(chunk["chunk_id"]) <= set(
            "abcdefghijklmnopqrstuvwxyz0123456789-@."
        )

    assert unit(chaptered, "203").slug() == "part-ii-chapter-xix-section-203"


def test_repealed_sections_are_still_dropped_under_chapters(chaptered):
    assert unit(chaptered, "139") is None


def test_an_act_without_chapters_is_unchanged(sample):
    """
    The Criminal Procedure Code regression. Chapter support is additive: with
    no Chapter headings the path stays two levels, exactly as before.
    """

    for candidate in sample.units:
        assert len(candidate.path) == 2

    assert unit(sample, "21").path == ["Part III", "Section 21"]


# ---------------------------------------------------------
# MIXED ARABIC AND ROMAN PART NUMBERING
#
# Found by the Employment Act, which opens with "Part 1 – PRELIMINARY" in
# Arabic and then switches to Roman for Parts II to XIII.
#
# The failure this guards is silent, not loud. A Roman-only PART_PATTERN does
# not mislabel Part 1 - it fails to match, so `_parse_tree` goes on treating
# the text as the Act's opening block and drops every section until the next
# Part heading. Sections 1 to 3 disappeared, taking the definitions in
# section 2 with them, and the only downstream symptom was a section count
# three short.
# ---------------------------------------------------------

MIXED_NUMBERING = """EMPLOYMENT ACT
CAP. 226
Published in Kenya Gazette Vol. CIX-No. 81 on 9 November 2007
[Amended by Employment (Amendment) Act, 2022 (Act No. 15 of 2022)]
An Act of Parliament to declare the fundamental rights of employees
Part 1 – PRELIMINARY
1.
Short title
This Act may be cited as the Employment Act.
2.
Interpretation
In this Act, unless the context otherwise requires—
"basic salary" means an employee's gross salary excluding allowances.
Part VI – TERMINATION AND DISMISSAL
35.
Termination notice
(1)
A contract of service may be terminated by either party giving notice.
Part XIII – MISCELLANEOUS PROVISIONS
90.
Rules
The Cabinet Secretary may make rules generally for the better carrying out
of the provisions of this Act.
91.
[Spent]
"""


@pytest.fixture(scope="module")
def mixed(entry):
    return act_structure.parse(MIXED_NUMBERING, entry)


def test_an_arabic_part_number_is_matched(mixed):
    assert unit(mixed, "1").path == ["Part 1", "Section 1"]


def test_sections_under_an_arabic_part_are_not_dropped(mixed):
    """
    The regression itself. Before Arabic support these two were discarded
    as opening-block text, in silence.
    """

    assert unit(mixed, "1") is not None
    assert unit(mixed, "2") is not None

    assert "basic salary" in unit(mixed, "2").text


def test_arabic_and_roman_parts_coexist_in_one_act(mixed):
    labels = {candidate.path[0] for candidate in mixed.units}

    assert labels == {"Part 1", "Part VI", "Part XIII"}


def test_an_arabic_part_keeps_chunk_ids_clean(mixed):
    assert unit(mixed, "1").slug() == "part-1-section-1"

    for chunk in build_chunks(mixed):
        assert set(chunk["chunk_id"]) <= set(
            "abcdefghijklmnopqrstuvwxyz0123456789-@."
        )


def test_spent_sections_are_dropped(mixed):
    """
    The third spelling of "no longer operative", after repealed and deleted.
    Previously excluded only as an accident of having no body.
    """

    assert unit(mixed, "91") is None


def test_roman_only_acts_are_unaffected_by_arabic_support(sample):
    """
    The Criminal Procedure Code regression. "III" cannot match `\\d+`, so
    every Act already ingested must parse exactly as it did before - same
    labels, therefore same chunk ids.
    """

    assert unit(sample, "1").path == ["Part I", "Section 1"]
    assert unit(sample, "21").path == ["Part III", "Section 21"]
    assert unit(sample, "329A").path == ["Part IXA", "Section 329A"]


def test_a_part_reference_in_running_text_is_not_a_heading():
    """
    "Part IX or Division 1 of Part IV of that Act;" appears mid-sentence in
    the Employment Act. A Part heading needs a dash straight after the
    numeral, which prose does not supply.
    """

    assert not act_structure.PART_PATTERN.match(
        "Part IX or Division 1 of Part IV of that Act;"
    )
    assert not act_structure.PART_PATTERN.match("Part 5 of the Schedule")

    assert act_structure.PART_PATTERN.match("Part 1 – PRELIMINARY")


# ---------------------------------------------------------
# SECTIONS THAT BELONG TO NO PART
#
# Found by the Law of Succession Act, which has no Part I: its divisions begin
# at "Part II – WILLS" and sections 1 to 4 sit above them ungrouped.
#
# The guard that drops everything before the first Part exists for a good
# reason - the gazette block and the "[Amended by ...]" log are not law - so
# it cannot simply be removed. A SECTION heading opens an unparted bucket;
# anything else up there is still discarded.
# ---------------------------------------------------------

NO_FIRST_PART = """LAW OF SUCCESSION ACT
CAP. 160
Published in Kenya Gazette Vol. LXXIV-No. 55 on 17 November 1972
[Amended by Statute Law (Miscellaneous Amendments) Act, 1975 (Act No. 7 of 1975)]
An ACT of Parliament to consolidate the law relating to succession
1.
Short title
This Act may be cited as the Law of Succession Act.
3.
Interpretation
In this Act, unless the context otherwise requires—
"will" means the legal declaration by a person of his wishes.
Part II – WILLS
5.
Persons capable of making wills and freedom of testation
(1)
Any person who is of sound mind and not a minor may dispose of all or any of
his free property by will.
Part VIII – MISCELLANEOUS
96.
Sane murderer not to share in victim's estate
A person who has been convicted of murdering another shall not be entitled to
share in the estate of the murdered person.
"""


@pytest.fixture(scope="module")
def unparted(entry):
    return act_structure.parse(NO_FIRST_PART, entry)


def test_sections_above_the_first_part_survive(unparted):
    """The regression. Before this, sections 1 and 3 were dropped silently."""

    assert unit(unparted, "1") is not None
    assert unit(unparted, "3") is not None

    assert "legal declaration" in unit(unparted, "3").text


def test_an_unparted_section_gets_a_one_level_path(unparted):
    """
    Deliberately short rather than invented. Synthesising "Part I" so every
    path looks alike would put a division in the citation that the Act does
    not contain.
    """

    assert unit(unparted, "3").path == ["Section 3"]
    assert unit(unparted, "3").slug() == "section-3"


def test_parted_sections_are_unaffected_by_the_unparted_bucket(unparted):
    assert unit(unparted, "5").path == ["Part II", "Section 5"]
    assert unit(unparted, "96").path == ["Part VIII", "Section 96"]


def test_the_opening_block_is_still_dropped_when_there_is_no_part_i(unparted):
    """
    The amendment log and the long title precede section 1 here, with no Part
    heading to stop them. They must not be swept into the first section.
    """

    for candidate in unparted.units:
        assert "Amended by" not in candidate.text
        assert "ACT of Parliament" not in candidate.text


def test_unparted_sections_keep_chunk_ids_clean(unparted):
    for chunk in build_chunks(unparted):
        assert set(chunk["chunk_id"]) <= set(
            "abcdefghijklmnopqrstuvwxyz0123456789-@."
        )


def test_acts_that_open_with_a_part_gain_no_unparted_sections(sample):
    """
    The Criminal Procedure Code regression. Its first Part precedes its first
    section, so every unit keeps a Part in its path and no chunk id moves.
    """

    for candidate in sample.units:
        assert len(candidate.path) >= 2
        assert candidate.path[0].startswith("Part ")


# ---------------------------------------------------------
# A WRAPPED SENTENCE MUST NOT END THE ACT
#
# Found by the Children Act. A schedule heading stops the parse, so a false
# positive here does not mislabel a unit - it TRUNCATES THE DOCUMENT.
#
# Section 218 ends "...shall be carried out in accordance with the Fifth
# Schedule.", the line wrapped, and "Fifth Schedule." landed at the start of
# a line. The old case-insensitive prefix match took it for the start of the
# schedules and silently discarded sections 219 to 250 - the whole of Part
# XV, Children in Conflict with the Law.
# ---------------------------------------------------------

def test_a_wrapped_schedule_reference_is_not_a_schedule_heading():
    assert not act_structure.SCHEDULE_PATTERN.match("Fifth Schedule.")
    assert not act_structure.SCHEDULE_PATTERN.match("Fifth Schedule")
    assert not act_structure.SCHEDULE_PATTERN.match(
        "as set out in the First Schedule to this Act."
    )


def test_real_schedule_headings_still_match():
    """Bare, and with the bracketed source reference Kenya Law uses."""

    assert act_structure.SCHEDULE_PATTERN.match("FIRST SCHEDULE")
    assert act_structure.SCHEDULE_PATTERN.match("FIFTH SCHEDULE [s. 218(5)]")
    assert act_structure.SCHEDULE_PATTERN.match(
        "THIRD SCHEDULE [s. 66(2), 144 (aa), 174, 186(6)]"
    )


def test_the_act_is_not_truncated_by_a_wrapped_reference():
    """End to end: the sentence fragment must not stop the parse."""

    text = """CHILDREN ACT
CAP. 141
Published in Kenya Gazette Vol. CXXIV-No. 146 on 29 July 2022
Part XIV – ADOPTION
218.
Assessment of a child
(1)
The assessment of a child for purposes of this section shall be carried out in
accordance with the
Fifth Schedule.
219.
Criminal liability of a child
A child under twelve years of age shall not be criminally responsible.
FIRST SCHEDULE [s. 8(1)]
BEST INTEREST CONSIDERATIONS
1.
This paragraph belongs to a schedule and must not be emitted.
"""

    document = act_structure.parse(text, get_document("children-act"))

    numbers = {candidate.number for candidate in document.units}

    assert "219" in numbers, "the wrapped reference truncated the Act"
    assert "1" not in numbers, "schedule content leaked into the units"


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


EMPLOYMENT_ID = "employment-act"


@pytest.fixture(scope="module")
def employment_entry():
    return get_document(EMPLOYMENT_ID)


@pytest.fixture(scope="module")
def real_employment_act(employment_entry):
    if not employment_entry.cleaned_path.exists():
        pytest.skip(
            "cleaned artifact missing; run "
            f"`python -m data_pipeline.run --document {EMPLOYMENT_ID}` first"
        )

    text = employment_entry.cleaned_path.read_text(encoding="utf-8")

    return act_structure.parse(text, employment_entry)


def test_employment_act_is_registered():
    assert "employment-act" in registered_types()

    assert cleaner_spec("employment-act").start_pattern
    assert validator("employment-act") is not None


def test_real_employment_act_parses_to_the_expected_shape(real_employment_act):
    assert employment_validator.validate_document(real_employment_act) == []


def test_real_employment_act_text_passes_its_validator(employment_entry):
    if not employment_entry.cleaned_path.exists():
        pytest.skip("cleaned artifact missing")

    text = employment_entry.cleaned_path.read_text(encoding="utf-8")

    assert employment_validator.validate_cleaned_text(text) == []


SUCCESSION_ID = "law-of-succession-act"


@pytest.fixture(scope="module")
def succession_entry():
    return get_document(SUCCESSION_ID)


@pytest.fixture(scope="module")
def real_succession_act(succession_entry):
    if not succession_entry.cleaned_path.exists():
        pytest.skip(
            "cleaned artifact missing; run "
            f"`python -m data_pipeline.run --document {SUCCESSION_ID}` first"
        )

    text = succession_entry.cleaned_path.read_text(encoding="utf-8")

    return act_structure.parse(text, succession_entry)


def test_succession_act_is_registered():
    assert "law-of-succession-act" in registered_types()

    assert cleaner_spec("law-of-succession-act").start_pattern
    assert validator("law-of-succession-act") is not None


def test_real_succession_act_parses_to_the_expected_shape(real_succession_act):
    assert succession_validator.validate_document(real_succession_act) == []


def test_real_succession_act_text_passes_its_validator(succession_entry):
    if not succession_entry.cleaned_path.exists():
        pytest.skip("cleaned artifact missing")

    text = succession_entry.cleaned_path.read_text(encoding="utf-8")

    assert succession_validator.validate_cleaned_text(text) == []


# ---------------------------------------------------------
# DOCUMENTS SIX, SEVEN AND EIGHT
#
# Parametrised rather than given a fixture each: all three reuse `act`
# unchanged, so the only per-document knowledge is which validator to call.
# ---------------------------------------------------------

LAST_THREE = [
    ("children-act", "children_act"),
    ("marriage-act", "marriage_act"),
    ("small-claims-court-act", "small_claims_court_act"),
]


@pytest.mark.parametrize("document_id, validator_module", LAST_THREE)
def test_last_three_documents_are_registered(document_id, validator_module):
    assert document_id in registered_types()

    assert cleaner_spec(document_id).start_pattern
    assert validator(document_id) is not None


@pytest.mark.parametrize("document_id, validator_module", LAST_THREE)
def test_last_three_parse_to_their_expected_shape(document_id, validator_module):
    import importlib

    entry = get_document(document_id)

    if not entry.cleaned_path.exists():
        pytest.skip(
            "cleaned artifact missing; run "
            f"`python -m data_pipeline.run --document {document_id}` first"
        )

    module = importlib.import_module(
        f"data_pipeline.validators.{validator_module}"
    )

    text = entry.cleaned_path.read_text(encoding="utf-8")

    assert module.validate_cleaned_text(text) == []
    assert module.validate_document(act_structure.parse(text, entry)) == []


def test_only_the_first_four_succession_sections_lack_a_part(real_succession_act):
    """
    The whole Act must not quietly fall into the unparted bucket if a Part
    heading ever stops matching. Exactly four sections belong to no Part.
    """

    unparted = {
        u.number for u in real_succession_act.units if len(u.path) == 1
    }

    assert unparted == {"1", "2", "3", "4"}
