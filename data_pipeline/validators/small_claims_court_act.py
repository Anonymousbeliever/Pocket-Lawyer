"""
Small Claims Court Act specific validation.

Generic text and chunk checks live in `generic`. What is specific here is
knowing that this Act runs from section 1 to 51 with no gaps, has six Parts,
one lettered section (20A), and no Schedules at all.

The smallest document in the corpus, and the only one with nothing unusual
about it - which is itself worth asserting, because "nothing unusual" is a
claim that stops being true the moment the Act is amended.

All counts below were measured against the 2022-12-31 consolidation.
"""

import re


REQUIRED_MARKERS = {
    "act title": "SMALL CLAIMS COURT ACT",
    "section 1": "Short title",
    "section 2": "Interpretation",
    "section 12": "Nature of claims and pecuniary jurisdiction",
    "section 38": "Appeals",
    "section 51": "Code of conduct for Adjudicators",
    "first part": "Part I",
    "jurisdiction part": "Part III",
    "last part": "Part VI",
}

EXPECTED_SECTIONS = list(range(1, 52))

EXPECTED_PARTS = 6

SAMPLE_LETTERED_SECTIONS = ["20A"]

SECTION_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")

LETTERED_SECTION_HEADING = re.compile(r"(?m)^(\d{1,3}[A-Z]{1,2})\.\s*$")

PART_HEADING = re.compile(r"(?m)^Part\s+(?:\d+|[IVXLCDM]+[A-Z]?)\s*[–—-]")

CHAPTER_HEADING = re.compile(r"(?m)^Chapter\s+[IVXLCDM]+[A-Z]?$")

SCHEDULE_HEADING = re.compile(
    r"(?m)^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+SCHEDULE\b", re.IGNORECASE
)

# Measured: 52 section headings (1-51 plus 20A), none repealed or spent.
EXPECTED_LIVE_SECTIONS = 52


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
    schedules = len(SCHEDULE_HEADING.findall(text))

    if parts != EXPECTED_PARTS:
        errors.append(f"Expected {EXPECTED_PARTS} Part headings, found {parts}.")

    if chapters:
        errors.append(f"Found {chapters} Chapter heading(s); this Act has none.")

    if schedules:
        errors.append(
            f"Found {schedules} Schedule heading(s); this Act has none, so "
            "either the source changed or the cleaner kept trailing matter."
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
        errors.append("Schedules were emitted; this Act has none.")

    wrong_depth = [unit.path for unit in document.units if len(unit.path) != 2]

    if wrong_depth:
        errors.append(
            f"{len(wrong_depth)} section(s) have a citation path that is not "
            f"Part -> Section, e.g. {wrong_depth[0]}"
        )

    return errors
