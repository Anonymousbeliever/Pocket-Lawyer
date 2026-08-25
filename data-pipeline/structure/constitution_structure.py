import json
import re
from pathlib import Path


INPUT_PATH = Path("data/cleaned/constitution_cleaned.txt")
OUTPUT_PATH = Path("data/processed/constitution_structure.json")


CHAPTER_PATTERN = re.compile(
    r"^Chapter\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|"
    r"ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN)$",
    re.IGNORECASE,
)

# IMPORTANT:
# Articles in the cleaned Constitution appear as:
#
# 1.
# Sovereignty of the people
#
ARTICLE_NUMBER_PATTERN = re.compile(r"^(\d{1,3})\.\s*$")

SUBSECTION_PATTERN = re.compile(r"^\((\d+)\)$")

PARAGRAPH_PATTERN = re.compile(r"^\(([a-z])\)$")

SCHEDULE_PATTERN = re.compile(
    r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH) SCHEDULE$",
    re.IGNORECASE,
)


def is_chapter(line: str) -> bool:
    return bool(CHAPTER_PATTERN.match(line.strip()))


def is_article_number(line: str) -> bool:
    return bool(ARTICLE_NUMBER_PATTERN.match(line.strip()))


def is_subsection(line: str) -> bool:
    return bool(SUBSECTION_PATTERN.match(line.strip()))


def is_paragraph(line: str) -> bool:
    return bool(PARAGRAPH_PATTERN.match(line.strip()))


def is_schedule(line: str) -> bool:
    return bool(SCHEDULE_PATTERN.match(line.strip()))


def clean_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def parse_constitution(text: str) -> dict:
    lines = clean_lines(text)

    document = {
        "document_id": "constitution-of-kenya-2010",
        "title": "Constitution of Kenya, 2010",
        "document_type": "constitution",
        "jurisdiction": "Kenya",
        "language": "English",
        "preamble": "",
        "chapters": [],
        "schedules": [],
    }

    current_chapter = None
    current_article = None
    current_subsection = None
    current_paragraph = None

    preamble_lines = []

    mode = "preamble"

    for index, line in enumerate(lines):

        # =========================================================
        # PREAMBLE
        # =========================================================

        if mode == "preamble":
            preamble_lines.append(line)

            if line == "GOD BLESS KENYA":
                mode = "before_chapter"

            continue

        # =========================================================
        # CHAPTER
        # =========================================================

        if is_chapter(line):
            current_chapter = {
                "number": line.replace("Chapter", "").strip(),
                "title": "",
                "articles": [],
            }

            document["chapters"].append(current_chapter)

            current_article = None
            current_subsection = None
            current_paragraph = None

            mode = "chapter"

            continue

        # =========================================================
        # CHAPTER TITLE
        # =========================================================

        if (
            current_chapter
            and not current_chapter["title"]
            and current_article is None
            and mode == "chapter"
        ):
            current_chapter["title"] = line
            continue

        # =========================================================
        # SCHEDULE
        # =========================================================

        if is_schedule(line):
            schedule = {
                "title": line,
                "content": [],
            }

            document["schedules"].append(schedule)

            current_chapter = None
            current_article = None
            current_subsection = None
            current_paragraph = None

            mode = "schedule"

            continue

        # =========================================================
        # ARTICLE
        # =========================================================

        if current_chapter and is_article_number(line):

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

        # =========================================================
        # SUBSECTION
        # =========================================================

        if current_article and is_subsection(line):

            match = SUBSECTION_PATTERN.match(line)

            current_subsection = {
                "number": int(match.group(1)),
                "paragraphs": [],
                "text": [],
            }

            current_article["subsections"].append(
                current_subsection
            )

            current_paragraph = None

            continue

        # =========================================================
        # PARAGRAPH
        # =========================================================

        if current_subsection and is_paragraph(line):

            match = PARAGRAPH_PATTERN.match(line)

            current_paragraph = {
                "number": match.group(1),
                "text": [],
            }

            current_subsection["paragraphs"].append(
                current_paragraph
            )

            continue

        # =========================================================
        # CONTENT
        # =========================================================

        if current_subsection:

            if current_paragraph:
                current_paragraph["text"].append(line)
            else:
                current_subsection["text"].append(line)

        elif current_article:

            current_article["text"].append(line)

        elif document["schedules"]:

            document["schedules"][-1]["content"].append(line)

    document["preamble"] = "\n".join(preamble_lines)

    return document


def main() -> None:

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input file not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = INPUT_PATH.read_text(
        encoding="utf-8"
    )

    document = parse_constitution(text)

    OUTPUT_PATH.write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("POCKET LAWYER — CONSTITUTION STRUCTURE")
    print("=" * 60)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    print(
        f"Chapters detected: "
        f"{len(document['chapters'])}"
    )

    print(
        f"Schedules detected: "
        f"{len(document['schedules'])}"
    )

    total_articles = sum(
        len(chapter["articles"])
        for chapter in document["chapters"]
    )

    print(
        f"Articles detected: "
        f"{total_articles}"
    )

    print()

    print("Structure extraction completed.")


if __name__ == "__main__":
    main()