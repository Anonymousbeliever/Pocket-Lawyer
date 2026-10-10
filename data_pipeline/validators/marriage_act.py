"""
Marriage Act specific validation.

Generic text and chunk checks live in `generic`. What is specific here is
knowing that this Act runs from section 1 to 98 with no gaps, has fourteen
Parts, no lettered sections - and that the number 50 legitimately appears
twice in the text for a reason that is not a duplicate section.

All counts below were measured against the 2022-12-31 consolidation.
"""

import re


REQUIRED_MARKERS = {
    "act title": "MARRIAGE ACT",
    "section 1": "Short title",
    "section 3": "Meaning of marriage",
    "section 11": "Void marriages",
    "section 12": "Voidable marriages",
    "section 98": "Savings",
    "first part": "Part I",
    "divorce part": "Part X",
    "last part": "Part XIV",
}

EXPECTED_SECTIONS = list(range(1, 99))

EXPECTED_PARTS = 14

SAMPLE_LETTERED_SECTIONS: list[str] = []

SECTION_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")

LETTERED_SECTION_HEADING = re.compile(r"(?m)^(\d{1,3}[A-Z]{1,2})\.\s*$")

PART_HEADING = re.compile(r"(?m)^Part\s+(?:\d+|[IVXLCDM]+[A-Z]?)\s*[–—-]")

CHAPTER_HEADING = re.compile(r"(?m)^Chapter\s+[IVXLCDM]+[A-Z]?$")

# Section 18 ends "... appointed by the registrar under section 50." and the
# line wraps, leaving "50." alone on its own line - indistinguishable from a
# section heading to a pattern that matches a bare number.
#
# The parser creates a second section 50 there, whose body is empty because
# the next line opens section 19, so the empty-content drop discards it and
# no chunk id collides. The real section 50 (Appointment of Registrar and
# Marriage officers) is in Part VIII and survives intact.
#
# The only real cost is that section 18 loses the trailing "50." from its
# text. Recorded rather than papered over: if a future amendment makes this
# a genuine duplicate, the count below moves and the failure is loud.
KNOWN_TEXT_DUPLICATES = {50}

# Measured: 99 section headings for 98 sections (the wrapped cross-reference
# above accounts for the extra). Section 97 is spent, and the spurious 50
# carries no body, so both are dropped.
EXPECTED_LIVE_SECTIONS = 97


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

    duplicates = sorted(
        {n for n in numbers if numbers.count(n) > 1} - KNOWN_TEXT_DUPLICATES
    )

    if missing:
        errors.append(
            "Missing section numbers: " + ", ".join(map(str, missing[:20]))
        )

    if duplicates:
        errors.append(
            "Duplicate section numbers: " + ", ".join(map(str, duplicates[:20]))
        )

    # The known duplicate must still BE there. If it disappears, the text
    # changed under us and the exemption above is silently excusing nothing.
    for number in KNOWN_TEXT_DUPLICATES:
        if numbers.count(number) < 2:
            errors.append(
                f"Expected section {number} to appear twice (a wrapped "
                f"cross-reference in section 18); found "
                f"{numbers.count(number)}."
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
            "Schedules were emitted; the lone SCHEDULE is cut by the "
            "cleaner's end_pattern because it is not ordinal-prefixed and "
            "act.py therefore does not stop at it."
        )

    # The real section 50 must be the one that survived, in Part VIII.
    fifty = [u for u in document.units if u.number == "50"]

    if len(fifty) != 1:
        errors.append(
            f"Expected exactly one section 50 to survive parsing, got "
            f"{len(fifty)} - see KNOWN_TEXT_DUPLICATES."
        )
    elif fifty[0].path[0] != "Part VIII":
        errors.append(
            f"Section 50 is under {fifty[0].path[0]}, expected Part VIII - "
            "the wrapped cross-reference in section 18 was kept instead of "
            "the real section."
        )

    wrong_depth = [unit.path for unit in document.units if len(unit.path) != 2]

    if wrong_depth:
        errors.append(
            f"{len(wrong_depth)} section(s) have a citation path that is not "
            f"Part -> Section, e.g. {wrong_depth[0]}"
        )

    return errors
