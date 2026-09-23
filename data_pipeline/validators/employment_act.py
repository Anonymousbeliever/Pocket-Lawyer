"""
Employment Act specific validation.

Generic text and chunk checks live in `generic`. What is specific here is
knowing that this Act must contain sections 1 to 92 with no gaps, thirteen
Parts, the four lettered sections a digits-only pattern would silently skip,
and - the reason this file earns its place - that Part 1 is numbered in
Arabic where every later Part is Roman.

Without this a parse failure is silent: a broken pattern returns fewer units,
or none, and the only complaint downstream is "No chunks were generated".

All counts below were measured against the 2024-04-26 consolidation, not
estimated.
"""

import re


REQUIRED_MARKERS = {
    "act title": "EMPLOYMENT ACT",
    "section 1": "Short title",
    "section 2": "Interpretation",
    "section 35": "Termination notice",
    "section 44": "Summary dismissal",
    "section 45": "Unfair termination",
    "section 90": "Rules",
    "first part": "Part 1",
    "last part": "Part XIII",
}

EXPECTED_SECTIONS = list(range(1, 93))

EXPECTED_PARTS = 13

# Sections inserted by amendment carry a letter suffix and are the most
# likely thing to be lost by a too-narrow pattern. 29A is the only live one;
# the 31 series is repealed but its headings still appear in the text.
SAMPLE_LETTERED_SECTIONS = ["29A", "31A", "31B", "31C"]

SECTION_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")

LETTERED_SECTION_HEADING = re.compile(r"(?m)^(\d{1,3}[A-Z]{1,2})\.\s*$")

# Arabic OR Roman. Part 1 is "Part 1"; Parts II to XIII are Roman. A
# Roman-only pattern here would report twelve Parts and call that a pass.
PART_HEADING = re.compile(r"(?m)^Part\s+(?:\d+|[IVXLCDM]+[A-Z]?)\s*[–—-]")

# The Act has no Chapter level - it is Part -> Section, like the Criminal
# Procedure Code.
CHAPTER_HEADING = re.compile(r"(?m)^Chapter\s+[IVXLCDM]+[A-Z]?$")

# Measured: 96 section headings (1-92 plus 29A, 31A, 31B, 31C), of which five
# carry no operative text - 31A deleted, 31B and 31C repealed, 91 and 92 spent.
EXPECTED_LIVE_SECTIONS = 91

# Part 1's sections are the specific thing the Arabic numbering endangers.
# If `act.py` ever stops matching "Part 1", these vanish silently and
# section 2 - the definitions of "employee", "wages" and "termination" -
# goes with them.
PART_ONE_SECTIONS = ["1", "2", "3"]


def validate_cleaned_text(text: str) -> list[str]:
    errors: list[str] = []

    errors.extend(_required_content(text))
    errors.extend(_section_sequence(text))
    errors.extend(_divisions(text))

    return errors


def _required_content(text: str) -> list[str]:
    return [
        f"Missing required content: {name}"
        for name, marker in REQUIRED_MARKERS.items()
        if marker not in text
    ]


def _section_sequence(text: str) -> list[str]:
    errors: list[str] = []

    numbers = [int(match) for match in SECTION_HEADING.findall(text)]

    if not numbers:
        return ["No section headings detected."]

    missing = [n for n in EXPECTED_SECTIONS if n not in set(numbers)]

    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})

    if missing:
        errors.append(
            "Missing section numbers: "
            + ", ".join(map(str, missing[:20]))
        )

    if duplicates:
        errors.append(
            "Duplicate section numbers: "
            + ", ".join(map(str, duplicates[:20]))
        )

    lettered = set(LETTERED_SECTION_HEADING.findall(text))

    absent = [s for s in SAMPLE_LETTERED_SECTIONS if s not in lettered]

    if absent:
        errors.append(
            "Lettered sections not detected: " + ", ".join(absent)
        )

    return errors


def _divisions(text: str) -> list[str]:
    errors: list[str] = []

    parts = len(PART_HEADING.findall(text))
    chapters = len(CHAPTER_HEADING.findall(text))

    if parts != EXPECTED_PARTS:
        errors.append(f"Expected {EXPECTED_PARTS} Part headings, found {parts}.")

    if chapters:
        errors.append(
            f"Found {chapters} Chapter heading(s); this Act has none."
        )

    return errors


def validate_document(document) -> list[str]:
    """Structural expectations for the parsed Act."""

    errors: list[str] = []

    sections = document.unit_count("section")

    if sections != EXPECTED_LIVE_SECTIONS:
        errors.append(
            f"Expected {EXPECTED_LIVE_SECTIONS} live sections, "
            f"parsed {sections}."
        )

    if document.unit_count("schedule"):
        errors.append(
            "Schedules were emitted; the Employment Act has none."
        )

    # The Arabic "Part 1" regression, caught by name rather than by a count
    # that would merely look three short.
    parsed = {unit.number for unit in document.units}

    missing_part_one = [s for s in PART_ONE_SECTIONS if s not in parsed]

    if missing_part_one:
        errors.append(
            "Part 1 sections missing: "
            + ", ".join(missing_part_one)
            + " - 'Part 1' is Arabic where later Parts are Roman, so a "
            "Roman-only PART_PATTERN drops them silently."
        )

    # Every citation path is Part -> Section. A third level would mean the
    # Chapter branch fired on a document that has no Chapters.
    wrong_depth = [
        unit.path
        for unit in document.units
        if len(unit.path) != 2
    ]

    if wrong_depth:
        errors.append(
            f"{len(wrong_depth)} section(s) have a citation path that is "
            f"not Part -> Section, e.g. {wrong_depth[0]}"
        )

    return errors
