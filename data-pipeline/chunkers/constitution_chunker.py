import json
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/constitution_structure.json"
)

OUTPUT_PATH = Path(
    "data/processed/constitution_chunks.json"
)


DOCUMENT_ID = "constitution-of-kenya-2010"


def build_article_content(article: dict) -> str:
    """
    Reconstruct the readable legal text of an article
    from the structured representation.
    """

    lines = []

    for subsection in article["subsections"]:

        number = subsection["number"]

        lines.append(
            f"({number}) "
            + " ".join(subsection["text"])
        )

        for paragraph in subsection["paragraphs"]:

            paragraph_number = paragraph["number"]

            text = " ".join(paragraph["text"])

            lines.append(
                f"({paragraph_number}) {text}"
            )

    # Some articles may have text that is not inside
    # a subsection.
    for text in article["text"]:

        if text.strip() != article["title"].strip():
            lines.append(text)

    return "\n".join(lines).strip()


def create_article_chunk(
    chapter: dict,
    article: dict,
) -> dict:

    content = build_article_content(article)

    return {
        "chunk_id": (
            f"{DOCUMENT_ID}"
            f"-chapter-{chapter['number'].lower()}"
            f"-article-{article['number']}"
        ),

        "document_id": DOCUMENT_ID,

        "document_type": "constitution",

        "title": "Constitution of Kenya, 2010",

        "jurisdiction": "Kenya",

        "language": "English",

        "chapter": {
            "number": chapter["number"],
            "title": chapter["title"],
        },

        "article": {
            "number": article["number"],
            "title": article["title"],
        },

        "content": content,

        "source": {
            "name": "Kenya Law",
            "document": "Constitution of Kenya, 2010",
        },
    }


def create_schedule_chunks(
    document: dict,
) -> list[dict]:

    chunks = []

    for index, schedule in enumerate(
        document["schedules"],
        start=1,
    ):

        content = "\n".join(
            schedule["content"]
        ).strip()

        if not content:
            continue

        chunks.append(
            {
                "chunk_id": (
                    f"{DOCUMENT_ID}"
                    f"-schedule-{index}"
                ),

                "document_id": DOCUMENT_ID,

                "document_type": "constitution",

                "title": "Constitution of Kenya, 2010",

                "jurisdiction": "Kenya",

                "language": "English",

                "schedule": {
                    "number": index,
                    "title": schedule["title"],
                },

                "content": content,

                "source": {
                    "name": "Kenya Law",
                    "document": "Constitution of Kenya, 2010",
                },
            }
        )

    return chunks


def create_chunks(document: dict) -> list[dict]:

    chunks = []

    # ---------------------------------------------------------
    # ARTICLE CHUNKS
    # ---------------------------------------------------------

    for chapter in document["chapters"]:

        for article in chapter["articles"]:

            chunk = create_article_chunk(
                chapter,
                article,
            )

            if chunk["content"]:
                chunks.append(chunk)

    # ---------------------------------------------------------
    # SCHEDULE CHUNKS
    # ---------------------------------------------------------

    chunks.extend(
        create_schedule_chunks(document)
    )

    return chunks


def validate_chunks(
    chunks: list[dict],
) -> list[str]:

    errors = []

    if not chunks:
        errors.append("No chunks were generated.")
        return errors

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append(
            "Duplicate chunk IDs detected."
        )

    for index, chunk in enumerate(chunks, start=1):

        required_fields = [
            "chunk_id",
            "document_id",
            "document_type",
            "title",
            "jurisdiction",
            "language",
            "content",
            "source",
        ]

        for field in required_fields:

            if field not in chunk:
                errors.append(
                    f"Chunk {index} missing field: {field}"
                )

        if not chunk.get("content", "").strip():

            errors.append(
                f"Chunk {index} has empty content."
            )

    return errors


def main() -> None:

    if not INPUT_PATH.exists():

        raise SystemExit(
            f"Input file not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    chunks = create_chunks(document)

    errors = validate_chunks(chunks)

    print("=" * 60)
    print("POCKET LAWYER — CONSTITUTION CHUNKING")
    print("=" * 60)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    print(
        f"Chunks generated: {len(chunks)}"
    )

    print()

    if errors:

        print("CHUNKING FAILED")
        print()

        for error in errors:
            print(f"[FAIL] {error}")

        print()
        print("=" * 60)

        raise SystemExit(1)

    OUTPUT_PATH.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    article_chunks = sum(
        1
        for chunk in chunks
        if "article" in chunk
    )

    schedule_chunks = sum(
        1
        for chunk in chunks
        if "schedule" in chunk
    )

    print("[PASS] Chunk IDs are unique")
    print("[PASS] Required metadata present")
    print("[PASS] No empty chunks")
    print("[PASS] Chunk validation passed")
    print()
    print(f"Article chunks:  {article_chunks}")
    print(f"Schedule chunks: {schedule_chunks}")
    print()
    print("Chunking completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()