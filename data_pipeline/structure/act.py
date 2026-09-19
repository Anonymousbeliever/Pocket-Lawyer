"""
Structure adapter for a Kenyan Act of Parliament.

Parses cleaned text into the common IR:

    Act -> Parts -> Sections -> subsections -> paragraphs

This is deliberately written for Acts in general rather than for one
Act. Kenya Law publishes consolidated Acts in a consistent shape, so the
next Act should need only a `CleanerSpec` and a registry entry.

Two things are intentionally *not* emitted:

Repealed sections. A repealed section's whole body is a single bracket
line, e.g. "[Repealed by Act No. 33 of 1963, 1st Sch.]". That lands in
the title slot, leaves the content empty, and the unit is dropped — the
same rule the Constitution adapter applies. They carry no legal content
and would otherwise crowd the index with near-identical chunks.

Schedules. Kenyan Act schedules are typically multi-column tables, and
linear PDF text extraction shreds them into one line per cell. Indexing
that produces high-volume, low-signal chunks. Parsing them needs a
table-aware extractor, not a line-based one, so the adapter stops at the
first schedule heading.
"""

import re

from data_pipeline.ir import Document, Unit, document_from_entry
from data_pipeline.registry import DocumentEntry


# Part headings carry a Roman numeral that may take a letter suffix
# (Part IXA was inserted between IX and X), then a dash and a title:
#
#   Part III – GENERAL PROVISIONS ARREST, ESCAPE AND RETAKING
#
PART_PATTERN = re.compile(
    r"^Part\s+([IVXLCDM]+[A-Z]?)\s*[–—-]\s*(.*)$"
)

# Some Acts put a Chapter level between Part and Section. The Penal Code has
# two Parts and thirty-eight Chapters; the Criminal Procedure Code has none.
#
# Note the shape differs from a Part heading: where a Part is one line with a
# dash ("Part II – CRIMES"), a Chapter splits across two:
#
#   Chapter XV
#   OFFENCES AGAINST MORALITY
#
CHAPTER_PATTERN = re.compile(r"^Chapter\s+([IVXLCDM]+[A-Z]?)$")

# Sections appear as a bare number on its own line, followed by the
# section title on the next line. The letter suffix matters: 137A to
# 137N and 379A are real sections that a digits-only pattern misses.
#
#   36A.
#   Remand by court
#
SECTION_NUMBER_PATTERN = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s*$")

SUBSECTION_PATTERN = re.compile(r"^\((\d+)\)$")

# Paragraphs are (a), (b) ... and sub-paragraphs (i), (ii), (iii).
# Both are treated as one flat level: the hierarchy below a subsection
# is not separately addressable, matching the Constitution adapter's
# existing behaviour.
PARAGRAPH_PATTERN = re.compile(r"^\(([a-z]{1,4})\)$")

SCHEDULE_PATTERN = re.compile(
    r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)"
    r"\s+SCHEDULE\b",
    re.IGNORECASE,
)

# Amendment annotations: "[Act No. 5 of 2003, s. 61.]" after a section,
# and the "[Amended by ...]" log in the Act's opening block.
ANNOTATION_PREFIX = "["

# A line that ends a sentence or clause is running text, never a heading.
HEADING_FORBIDDEN_ENDINGS = (".", ";", ",", ":", "-", "–", "—")

HEADING_MAX_CHARS = 90


# ---------------------------------------------------------
# NESTED PARSE
# ---------------------------------------------------------

def _clean_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def _is_cross_heading(line: str, next_line: str) -> bool:
    """
    True for a division heading that sits between two sections.

    An Act groups sections under unnumbered headings — "SEARCH
    WARRANTS", "Arrest Generally", "PROVISIONS AS TO BAIL". They are not
    citable units, and without this they would be appended to the
    *previous* section's text, attributing one section's heading to
    another.

    The test is positional: a short line with no terminal punctuation,
    immediately followed by a section-number line. Measured against the
    Criminal Procedure Code it selects 45 lines, all of them genuine
    headings, and no wrapped body text.
    """

    if not SECTION_NUMBER_PATTERN.match(next_line):
        return False

    if len(line) >= HEADING_MAX_CHARS:
        return False

    # Amendment annotations also precede section numbers.
    if line.startswith(ANNOTATION_PREFIX):
        return False

    if line.endswith(HEADING_FORBIDDEN_ENDINGS):
        return False

    # A surviving page number is not a heading.
    if line.isdigit():
        return False

    return True


def _parse_tree(text: str) -> dict:
    lines = _clean_lines(text)

    tree: dict = {"parts": []}

    current_part = None
    current_chapter = None
    current_section = None
    current_subsection = None
    current_paragraph = None

    for index, line in enumerate(lines):

        next_line = lines[index + 1] if index + 1 < len(lines) else ""

        # SCHEDULES — everything from here on is out of scope.
        if SCHEDULE_PATTERN.match(line):
            break

        # PART
        part_match = PART_PATTERN.match(line)

        if part_match:
            current_part = {
                "number": part_match.group(1),
                "title": part_match.group(2).strip(),
                "sections": [],
            }

            tree["parts"].append(current_part)

            current_chapter = None
            current_section = None
            current_subsection = None
            current_paragraph = None

            continue

        # CHAPTER — optional level, present in the Penal Code, absent from
        # the Criminal Procedure Code. Its title is on the following line,
        # which then falls through with no section active and is dropped.
        chapter_match = CHAPTER_PATTERN.match(line)

        if chapter_match:
            current_chapter = {
                "number": chapter_match.group(1),
                "title": next_line,
            }

            current_section = None
            current_subsection = None
            current_paragraph = None

            continue

        # Everything before the first Part is the Act's opening block:
        # gazette details and the "[Amended by ...]" log. It is an
        # amendment history, not law.
        if current_part is None:
            continue

        # CROSS-HEADING
        if _is_cross_heading(line, next_line):
            continue

        # SECTION
        section_match = SECTION_NUMBER_PATTERN.match(line)

        if section_match:
            current_section = {
                "number": section_match.group(1),
                "title": next_line,
                "chapter": current_chapter,
                "subsections": [],
                "text": [],
            }

            current_part["sections"].append(current_section)

            current_subsection = None
            current_paragraph = None

            continue

        # SUBSECTION
        if current_section and SUBSECTION_PATTERN.match(line):
            current_subsection = {
                "number": SUBSECTION_PATTERN.match(line).group(1),
                "paragraphs": [],
                "text": [],
            }

            current_section["subsections"].append(current_subsection)

            current_paragraph = None

            continue

        # PARAGRAPH
        if current_subsection and PARAGRAPH_PATTERN.match(line):
            current_paragraph = {
                "number": PARAGRAPH_PATTERN.match(line).group(1),
                "text": [],
            }

            current_subsection["paragraphs"].append(current_paragraph)

            continue

        # CONTENT
        if current_subsection:
            if current_paragraph:
                current_paragraph["text"].append(line)
            else:
                current_subsection["text"].append(line)

        elif current_section:
            current_section["text"].append(line)

    return tree


# ---------------------------------------------------------
# TEXT RECONSTRUCTION
# ---------------------------------------------------------

def _section_text(section: dict) -> str:
    """Rebuild a section's readable legal text from the parse tree."""

    lines: list[str] = []

    for subsection in section["subsections"]:
        lines.append(
            f"({subsection['number']}) "
            + " ".join(subsection["text"])
        )

        for paragraph in subsection["paragraphs"]:
            lines.append(
                f"({paragraph['number']}) "
                + " ".join(paragraph["text"])
            )

    # Sections with no subsections carry their text directly. Skip the
    # line that merely repeats the section title.
    for text in section["text"]:
        if text.strip() != section["title"].strip():
            lines.append(text)

    return "\n".join(lines).strip()


def _path(part_label: str, section: dict) -> list[str]:
    """
    The citation breadcrumb, with the Chapter included when the Act has one.

    Labels only, never titles - the Constitution adapter does the same with
    "Chapter Four". It keeps the citation short and keeps the commas and en
    dashes that live in Act headings out of `Unit.slug()`, and therefore out
    of chunk ids.

    An Act without Chapters produces exactly what it did before:

        ["Part III", "Section 21"]                     Criminal Procedure Code
        ["Part II", "Chapter XV", "Section 203"]       Penal Code
    """

    path = [part_label]

    chapter = section.get("chapter")

    if chapter:
        path.append(f"Chapter {chapter['number']}")

    path.append(f"Section {section['number']}")

    return path


def _is_repealed(section: dict) -> bool:
    """
    A repealed section's title is its whole body, in brackets.

    Checked explicitly rather than relying on the empty-content drop, so
    that the count is reportable and a repealed section can never be
    indexed just because a stray line landed in its text.
    """

    title = section["title"].strip().lower().lstrip("[")

    return title.startswith(("repealed", "deleted"))


# ---------------------------------------------------------
# ADAPTER ENTRYPOINT
# ---------------------------------------------------------

def parse(text: str, entry: DocumentEntry) -> Document:
    tree = _parse_tree(text)

    units: list[Unit] = []

    for part in tree["parts"]:
        # The Part label, not the Part title, goes in the path — the
        # Constitution adapter does the same with "Chapter Four". It
        # keeps the citation short and the chunk id free of the commas
        # and en dashes that Part titles contain.
        part_label = f"Part {part['number']}"

        for section in part["sections"]:
            if _is_repealed(section):
                continue

            content = _section_text(section)

            if not content:
                continue

            units.append(
                Unit(
                    unit_type="section",
                    number=section["number"],
                    title=section["title"],
                    path=_path(part_label, section),
                    text=content,
                )
            )

    return document_from_entry(entry, units)
