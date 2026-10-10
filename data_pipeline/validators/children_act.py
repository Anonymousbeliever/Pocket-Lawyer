"""
Children Act specific validation.

Generic text and chunk checks live in `generic`. What is specific here is
knowing that this Act runs from section 1 to 250 with no gaps, has sixteen
Parts, no lettered sections, and six Schedules that must NOT be emitted.

Without this a parse failure is silent: a broken pattern returns fewer units,
or none, and the only complaint downstream is "No chunks were generated".

All counts below were measured against the 2022-12-31 consolidation.
"""

import re


REQUIRED_MARKERS = {
    "act title": "CHILDREN ACT",
    "section 1": "Short title",
    "section 2": "Interpretation",
    "section 8": "Best interests of the child",
    "section 11": "Right to parental care",
    "first part": "Part I",
    "custody part": "Part IX",
    "last part": "Part XVI",
}

EXPECTED_SECTIONS = list(range(1, 251))

EXPECTED_PARTS = 16

# This Act has none. Stated explicitly so that a later amendment inserting
# one is a deliberate edit here rather than a silent miss.
SAMPLE_LETTERED_SECTIONS: list[str] = []

SECTION_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")

LETTERED_SECTION_HEADING = re.compile(r"(?m)^(\d{1,3}[A-Z]{1,2})\.\s*$")

PART_HEADING = re.compile(r"(?m)^Part\s+(?:\d+|[IVXLCDM]+[A-Z]?)\s*[–—-]")

CHAPTER_HEADING = re.compile(r"(?m)^Chapter\s+[IVXLCDM]+[A-Z]?$")

# The six Schedules restart paragraph numbering, so counting section
# headings across the whole cleaned text double-counts badly. Everything
# from the First Schedule onward is out of scope for the section checks.
FIRST_SCHEDULE = re.compile(r"(?m)^FIRST\s+SCHEDULE\b", re.IGNORECASE)

# Measured: 250 section headings, of which one - section 250 - is spent and
# carries no operative text.
EXPECTED_LIVE_SECTIONS = 249


def _operative_body(text: str) -> str:
    """The text above the first Schedule - the only part holding sections."""

    match = FIRST_SCHEDULE.search(text)

    return text[: match.start()] if match else text


def validate_cleaned_text(text: str) -> list[str]:
    errors: list[str] = []

    errors.extend(_required_content(text))
    errors.extend(_section_sequence(_operative_body(text)))
    errors.extend(_divisions(_operative_body(text)))

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
            "Missing section numbers: " + ", ".join(map(str, missing[:20]))
        )

    if duplicates:
        errors.append(
            "Duplicate section numbers: " + ", ".join(map(str, duplicates[:20]))
        )

    lettered = set(LETTERED_SECTION_HEADING.findall(text))

    absent = [s for s in SAMPLE_LETTERED_SECTIONS if s not in lettered]

    if absent:
        errors.append("Lettered sections not detected: " + ", ".join(absent))

    return errors


def _divisions(text: str) -> list[str]:
    errors: list[str] = []

    parts = len(PART_HEADING.findall(text))
    chapters = len(CHAPTER_HEADING.findall(text))

    if parts != EXPECTED_PARTS:
        errors.append(f"Expected {EXPECTED_PARTS} Part headings, found {parts}.")

    if chapters:
        errors.append(f"Found {chapters} Chapter heading(s); this Act has none.")

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
            "Schedules were emitted; this Act's six Schedules are forms and "
            "tables that linear extraction shreds, and act.py stops at the "
            "first."
        )

    # Every citation path is Part -> Section. A one-level path would mean a
    # Part heading stopped matching; a three-level one would mean the Chapter
    # branch fired on an Act that has no Chapters.
    wrong_depth = [unit.path for unit in document.units if len(unit.path) != 2]

    if wrong_depth:
        errors.append(
            f"{len(wrong_depth)} section(s) have a citation path that is not "
            f"Part -> Section, e.g. {wrong_depth[0]}"
        )

    return errors
