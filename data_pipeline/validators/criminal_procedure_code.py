"""
Criminal Procedure Code specific validation.

Generic text and chunk checks live in `generic`. What is specific here
is knowing that this consolidation must contain sections 1 to 394 with
no gaps, thirteen Parts, and the thirty-three lettered sections that a
digits-only pattern would silently skip.

This matters more than it looks. A structure adapter that stops matching
does not raise — it returns fewer units, or none, and the only complaint
downstream is "No chunks were generated." These checks turn a silent
parse failure into a named one.
"""

import re


REQUIRED_MARKERS = {
    "act title": "CRIMINAL PROCEDURE CODE",
    "section 1": "Short title",
    "section 36A": "Remand by court",
    "section 394": "Expenses of assessors, witnesses, etc.",
    "first part": "Part I",
    "last part": "Part XII",
}

EXPECTED_SECTIONS = list(range(1, 395))

EXPECTED_PARTS = 13

# Sections inserted by amendment carry a letter suffix. They are the
# most likely thing to be lost by a too-narrow pattern, so a sample is
# checked by name.
SAMPLE_LETTERED_SECTIONS = ["36A", "137A", "137N", "379A"]

SECTION_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")

LETTERED_SECTION_HEADING = re.compile(r"(?m)^(\d{1,3}[A-Z]{1,2})\.\s*$")

PART_HEADING = re.compile(r"(?m)^Part\s+[IVXLCDM]+[A-Z]?\s*[–—-]")

# Parts VII and VIII were repealed in their entirety — Part VII is
# titled "Repealed", and committal proceedings went with Act No. 5 of
# 2003. Both parse correctly and then contribute no units, so the
# document is expected to carry eleven Parts, not thirteen.
EXPECTED_PART_LABELS = [
    "Part I",
    "Part II",
    "Part III",
    "Part IV",
    "Part V",
    "Part VI",
    "Part IX",
    "Part IXA",
    "Part X",
    "Part XI",
    "Part XII",
]

EXPECTED_LIVE_SECTIONS = 311


def validate_cleaned_text(text: str) -> list[str]:
    errors: list[str] = []

    errors.extend(_required_content(text))
    errors.extend(_section_sequence(text))
    errors.extend(_parts(text))

    return errors


def _required_content(text: str) -> list[str]:
    return [
        f"Missing required content: {name}"
        for name, marker in REQUIRED_MARKERS.items()
        if marker not in text
    ]


def _body(text: str) -> str:
    """Operative text only — schedules carry their own numbering."""

    return re.split(r"(?m)^FIRST\s+SCHEDULE\b", text)[0]


def _section_sequence(text: str) -> list[str]:
    errors: list[str] = []

    body = _body(text)

    numbers = [int(match) for match in SECTION_HEADING.findall(body)]

    if not numbers:
        return ["No section headings detected."]

    missing = [n for n in EXPECTED_SECTIONS if n not in set(numbers)]

    duplicates = sorted(
        {n for n in numbers if numbers.count(n) > 1}
    )

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

    lettered = set(LETTERED_SECTION_HEADING.findall(body))

    absent = [s for s in SAMPLE_LETTERED_SECTIONS if s not in lettered]

    if absent:
        errors.append(
            "Lettered sections not detected: " + ", ".join(absent)
        )

    return errors


def _parts(text: str) -> list[str]:
    found = len(PART_HEADING.findall(_body(text)))

    if found != EXPECTED_PARTS:
        return [f"Expected {EXPECTED_PARTS} Part headings, found {found}."]

    return []


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
            "Schedules were emitted; this adapter deliberately skips "
            "them because they are tables that line-based extraction "
            "cannot represent."
        )

    labels: list[str] = []

    for unit in document.units:
        if unit.path and unit.path[0] not in labels:
            labels.append(unit.path[0])

    if labels != EXPECTED_PART_LABELS:
        errors.append(
            "Parts represented in units do not match the expected set. "
            f"Got: {', '.join(labels)}"
        )

    return errors
