"""
Law of Succession Act specific validation.

Generic text and chunk checks live in `generic`. What is specific here is
knowing that this Act runs from section 1 to 101 with no gaps, has seven
Parts numbered II to VIII, three lettered sections, nine Schedules that must
NOT be emitted - and, the reason this file earns its place, that sections 1
to 4 belong to no Part at all.

Without this a parse failure is silent: a broken pattern returns fewer units,
or none, and the only complaint downstream is "No chunks were generated".

All counts below were measured against the 2022-12-31 consolidation, not
estimated.
"""

import re


REQUIRED_MARKERS = {
    "act title": "LAW OF SUCCESSION ACT",
    "section 1": "Short title",
    "section 3": "Interpretation",
    "section 5": "Persons capable of making wills",
    "section 34": "Meaning of intestacy",
    "section 45": "No intermeddling with property of deceased person",
    "section 96": "Sane murderer not to share in victim's estate",
    "first part": "Part II",
    "last part": "Part VIII",
}

EXPECTED_SECTIONS = list(range(1, 102))

# Seven, not eight: this Act has no Part I. Its divisions begin at Part II,
# and sections 1 to 4 sit above them ungrouped. An off-by-one "fix" here
# would mask exactly the parse failure this file exists to catch.
EXPECTED_PARTS = 7

SAMPLE_LETTERED_SECTIONS = ["49A", "50A", "75A"]

SECTION_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")

LETTERED_SECTION_HEADING = re.compile(r"(?m)^(\d{1,3}[A-Z]{1,2})\.\s*$")

PART_HEADING = re.compile(r"(?m)^Part\s+(?:\d+|[IVXLCDM]+[A-Z]?)\s*[–—-]")

CHAPTER_HEADING = re.compile(r"(?m)^Chapter\s+[IVXLCDM]+[A-Z]?$")

# The Schedules restart paragraph numbering, so counting section headings
# across the whole cleaned text double-counts. Everything from the First
# Schedule onward is out of scope for the section checks.
FIRST_SCHEDULE = re.compile(r"(?m)^FIRST\s+SCHEDULE\b", re.IGNORECASE)

# Measured: 104 section headings in the operative body (1-101 plus 49A, 50A
# and 75A), of which three carry no operative text - 25 and 89 repealed,
# 99 spent.
EXPECTED_LIVE_SECTIONS = 101

# The four sections this Act's missing Part I endangers. Section 3 is
# Interpretation; losing it costs every definition in the Act.
UNPARTED_SECTIONS = ["1", "2", "3", "4"]


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
        errors.append(
            f"Expected {EXPECTED_PARTS} Part headings (II to VIII - this Act "
            f"has no Part I), found {parts}."
        )

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
            "Schedules were emitted; this Act's nine Schedules are tables "
            "that linear extraction shreds, and act.py stops at the first."
        )

    parsed = {unit.number for unit in document.units}

    # The missing-Part-I regression, caught by name rather than by a count
    # that would merely look four short.
    missing_unparted = [s for s in UNPARTED_SECTIONS if s not in parsed]

    if missing_unparted:
        errors.append(
            "Sections above the first Part are missing: "
            + ", ".join(missing_unparted)
            + " - this Act has no Part I, so a parser that drops everything "
            "before the first Part heading loses them silently, section 3 "
            "(Interpretation) included."
        )

    # Those four, and only those four, carry a one-level citation path.
    # Anything else at one level means a Part heading stopped matching
    # partway through the Act.
    one_level = {
        unit.number for unit in document.units if len(unit.path) == 1
    }

    unexpected = sorted(one_level - set(UNPARTED_SECTIONS))

    if unexpected:
        errors.append(
            f"{len(unexpected)} section(s) carry no Part in their citation "
            f"path but should, e.g. section {unexpected[0]}"
        )

    return errors
