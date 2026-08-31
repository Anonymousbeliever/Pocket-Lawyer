"""
Structure adapter for the Constitution of Kenya, 2010.

Parses cleaned text into the common IR:

    Constitution -> Chapters -> Articles -> subsections -> paragraphs
                 -> Schedules

The nested parse is kept as an internal step because it is what the
subsection/paragraph reconstruction needs; the adapter then flattens it
into `Unit` objects so that nothing downstream has to know what a
chapter is.
"""

import re

from data_pipeline.ir import Document, Unit
from data_pipeline.registry import DocumentEntry


CHAPTER_PATTERN = re.compile(
    r"^Chapter\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|"
    r"ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN)$",
    re.IGNORECASE,
)

# Articles in the cleaned Constitution appear as a bare number on its
# own line, followed by the article title on the next line:
#
#   1.
#   Sovereignty of the people
#
ARTICLE_NUMBER_PATTERN = re.compile(r"^(\d{1,3})\.\s*$")

SUBSECTION_PATTERN = re.compile(r"^\((\d+)\)$")

PARAGRAPH_PATTERN = re.compile(r"^\(([a-z])\)$")

SCHEDULE_PATTERN = re.compile(
    r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH) SCHEDULE$",
    re.IGNORECASE,
)


# ---------------------------------------------------------
# NESTED PARSE
# ---------------------------------------------------------

def _clean_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def _parse_tree(text: str) -> dict:
    lines = _clean_lines(text)

    tree: dict = {
        "preamble": "",
        "chapters": [],
        "schedules": [],
    }

    current_chapter = None
    current_article = None
    current_subsection = None
    current_paragraph = None

    preamble_lines: list[str] = []

    mode = "preamble"

    for index, line in enumerate(lines):

        # PREAMBLE
        if mode == "preamble":
            preamble_lines.append(line)

            if line == "GOD BLESS KENYA":
                mode = "before_chapter"

            continue

        # CHAPTER
        if CHAPTER_PATTERN.match(line):
            current_chapter = {
                "number": line.replace("Chapter", "").strip(),
                "title": "",
                "articles": [],
            }

            tree["chapters"].append(current_chapter)

            current_article = None
            current_subsection = None
            current_paragraph = None

            mode = "chapter"

            continue

        # CHAPTER TITLE
        if (
            current_chapter
            and not current_chapter["title"]
            and current_article is None
            and mode == "chapter"
        ):
            current_chapter["title"] = line
            continue

        # SCHEDULE
        if SCHEDULE_PATTERN.match(line):
            tree["schedules"].append(
                {
                    "title": line,
                    "content": [],
                }
            )

            current_chapter = None
            current_article = None
            current_subsection = None
            current_paragraph = None

            mode = "schedule"

            continue

        # ARTICLE
        if current_chapter and ARTICLE_NUMBER_PATTERN.match(line):
            article_number = int(
                ARTICLE_NUMBER_PATTERN.match(line).group(1)
            )

            next_line = (
                lines[index + 1]
                if index + 1 < len(lines)
                else ""
            )

            current_article = {
                "number": article_number,
                "title": next_line,
                "subsections": [],
                "text": [],
            }

            current_chapter["articles"].append(current_article)

            current_subsection = None
            current_paragraph = None

            continue

        # SUBSECTION
        if current_article and SUBSECTION_PATTERN.match(line):
            current_subsection = {
                "number": int(SUBSECTION_PATTERN.match(line).group(1)),
                "paragraphs": [],
                "text": [],
            }

            current_article["subsections"].append(current_subsection)

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

        elif current_article:
            current_article["text"].append(line)

        elif tree["schedules"]:
            tree["schedules"][-1]["content"].append(line)

    tree["preamble"] = "\n".join(preamble_lines)

    return tree


# ---------------------------------------------------------
# TEXT RECONSTRUCTION
# ---------------------------------------------------------

def _article_text(article: dict) -> str:
    """Rebuild an article's readable legal text from the parse tree."""

    lines: list[str] = []

    for subsection in article["subsections"]:
        lines.append(
            f"({subsection['number']}) "
            + " ".join(subsection["text"])
        )

        for paragraph in subsection["paragraphs"]:
            lines.append(
                f"({paragraph['number']}) "
                + " ".join(paragraph["text"])
            )

    # Articles with no subsections carry their text directly. Skip the
    # line that merely repeats the article title.
    for text in article["text"]:
        if text.strip() != article["title"].strip():
            lines.append(text)

    return "\n".join(lines).strip()


# ---------------------------------------------------------
# ADAPTER ENTRYPOINT
# ---------------------------------------------------------

def parse(text: str, entry: DocumentEntry) -> Document:
    tree = _parse_tree(text)

    units: list[Unit] = []

    for chapter in tree["chapters"]:
        chapter_label = f"Chapter {chapter['number']}"

        for article in chapter["articles"]:
            content = _article_text(article)

            if not content:
                continue

            units.append(
                Unit(
                    unit_type="article",
                    number=str(article["number"]),
                    title=article["title"],
                    path=[
                        chapter_label,
                        f"Article {article['number']}",
                    ],
                    text=content,
                )
            )

    for index, schedule in enumerate(tree["schedules"], start=1):
        content = "\n".join(schedule["content"]).strip()

        if not content:
            continue

        units.append(
            Unit(
                unit_type="schedule",
                number=str(index),
                title=schedule["title"],
                path=[schedule["title"]],
                text=content,
            )
        )

    return Document(
        document_id=entry.document_id,
        title=entry.title,
        document_type=entry.document_type,
        jurisdiction=entry.jurisdiction,
        language=entry.language,
        version=entry.version,
        effective_date=entry.effective_date,
        in_force=entry.in_force,
        source_name=entry.source_name,
        source_url=entry.source_url,
        units=units,
    )
