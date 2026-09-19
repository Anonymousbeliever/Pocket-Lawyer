"""
Penal Code specific validation.

Generic text and chunk checks live in `generic`. What is specific here is
knowing that this Act must contain sections 1 to 398 with no gaps, two Parts,
thirty-eight Chapters, and the lettered sections a digits-only pattern would
silently skip.

Without this a parse failure is silent: a broken pattern returns fewer units,
or none, and the only complaint downstream is "No chunks were generated."
"""

import re


REQUIRED_MARKERS = {
    "act title": "PENAL CODE",
    "section 1": "Short title",
    "section 203": "Murder",
    "section 398": "Punishment of accessories after the fact to misdemeanours",
    "first part": "Part I",
    "second part": "Part II",
}

EXPECTED_SECTIONS = list(range(1, 399))

EXPECTED_PARTS = 2

# The Chapter level is what distinguishes this Act from the Criminal
# Procedure Code, so its absence must be an error rather than a quiet
# flattening.
EXPECTED_CHAPTERS = 38

# Sections inserted by amendment carry a letter suffix and are the most
# likely thing to be lost by a too-narrow pattern.
SAMPLE_LETTERED_SECTIONS = ["26A", "122A", "122D", "242A", "278B"]

SECTION_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")

LETTERED_SECTION_HEADING = re.compile(r"(?m)^(\d{1,3}[A-Z]{1,2})\.\s*$")

PART_HEADING = re.compile(r"(?m)^Part\s+[IVXLCDM]+[A-Z]?\s*[–—-]")

CHAPTER_HEADING = re.compile(r"(?m)^Chapter\s+[IVXLCDM]+[A-Z]?$")

# Measured on the 2023-12-11 consolidation: 416 section headings, of which 48
# are repealed or deleted and carry no operative text.
EXPECTED_LIVE_SECTIONS = 368


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

    if chapters != EXPECTED_CHAPTERS:
        errors.append(
            f"Expected {EXPECTED_CHAPTERS} Chapter headings, found {chapters}."
        )

    return errors


def validate_document(document) -> list[str]:
    """Structural expectations for the parsed Code."""

    errors: list[str] = []

    sections = document.unit_count("section")

    if sections != EXPECTED_LIVE_SECTIONS:
        errors.append(
            f"Expected {EXPECTED_LIVE_SECTIONS} live sections, "
            f"parsed {sections}."
        )

    if document.unit_count("schedule"):
        errors.append(
            "Schedules were emitted; the Penal Code has none."
        )

    # The Chapter level is the reason this Act needed adapter work. If it
    # vanished, every citation would silently lose a level rather than fail.
    without_chapter = [
        unit.path
        for unit in document.units
        if len(unit.path) < 3
    ]

    if without_chapter:
        errors.append(
            f"{len(without_chapter)} section(s) carry no Chapter in their "
            f"citation path, e.g. {without_chapter[0]}"
        )

    return errors
