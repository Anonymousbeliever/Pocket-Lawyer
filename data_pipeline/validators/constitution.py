"""
Constitution-specific validation.

Generic text and chunk checks live in `generic`. What is specific here
is knowing that this document must contain articles 1 to 264 in order,
must open with the preamble, and must end with six schedules.
"""

import re


REQUIRED_MARKERS = {
    "preamble opening": "We, the people of Kenya",
    "preamble closing": "GOD BLESS KENYA",
    "article 1": "Sovereignty of the people",
    "article 49": "Rights of arrested persons",
    "article 264": "Repeal of previous Constitution",
    "first schedule": "FIRST SCHEDULE",
    "sixth schedule": "SIXTH SCHEDULE",
}

EXPECTED_ARTICLES = list(range(1, 265))

ARTICLE_HEADING = re.compile(r"(?m)^(\d{1,3})\.\s*$")


def validate_cleaned_text(text: str) -> list[str]:
    errors: list[str] = []

    errors.extend(_required_content(text))
    errors.extend(_article_sequence(text))
    errors.extend(_structure(text))

    return errors


def _required_content(text: str) -> list[str]:
    return [
        f"Missing required content: {name}"
        for name, marker in REQUIRED_MARKERS.items()
        if marker not in text
    ]


def _article_sequence(text: str) -> list[str]:
    errors: list[str] = []

    # Articles are only valid before the First Schedule.
    article_section = text.split("FIRST SCHEDULE", 1)[0]

    numbers = [
        int(match) for match in ARTICLE_HEADING.findall(article_section)
    ]

    if numbers == EXPECTED_ARTICLES:
        return errors

    if not numbers:
        return ["No article headings detected."]

    missing = [n for n in EXPECTED_ARTICLES if n not in numbers]

    duplicates = [
        n for n in sorted(set(numbers)) if numbers.count(n) > 1
    ]

    if missing:
        errors.append(
            "Missing article numbers: " + ", ".join(map(str, missing))
        )

    if duplicates:
        errors.append(
            "Duplicate article numbers: " + ", ".join(map(str, duplicates))
        )

    errors.append(
        "Article sequence does not match expected sequence 1-264."
    )

    return errors


def _structure(text: str) -> list[str]:
    errors: list[str] = []

    if not text.startswith("PREAMBLE"):
        errors.append("Document does not begin with PREAMBLE")

    for marker in ("Chapter ONE", "Chapter Two", "FIRST SCHEDULE", "SIXTH SCHEDULE"):
        if marker not in text:
            errors.append(f"{marker} not detected")

    return errors


def validate_document(document) -> list[str]:
    """Structural expectations for the parsed Constitution."""

    errors: list[str] = []

    articles = document.unit_count("article")
    schedules = document.unit_count("schedule")

    if articles != 264:
        errors.append(f"Expected 264 articles, parsed {articles}.")

    if schedules != 6:
        errors.append(f"Expected 6 schedules, parsed {schedules}.")

    return errors
