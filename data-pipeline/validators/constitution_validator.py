import re
from pathlib import Path


EXPECTED_ARTICLES = range(1, 265)

REQUIRED_MARKERS = {
    "preamble opening": "We, the people of Kenya",
    "preamble closing": "GOD BLESS KENYA",
    "article 1": "Sovereignty of the people",
    "article 49": "Rights of arrested persons",
    "article 264": "Repeal of previous Constitution",
    "first schedule": "FIRST SCHEDULE",
    "sixth schedule": "SIXTH SCHEDULE",
}


def validate_required_content(text: str) -> list[str]:
    errors = []

    for name, marker in REQUIRED_MARKERS.items():
        if marker not in text:
            errors.append(f"Missing required content: {name}")

    return errors


def validate_article_sequence(text: str) -> list[str]:
    errors = []

    # Articles are only valid before the First Schedule.
    article_section = text.split("FIRST SCHEDULE", 1)[0]

    matches = re.findall(
        r"(?m)^(\d{1,3})\.\s*$",
        article_section,
    )

    article_numbers = [int(number) for number in matches]

    expected = list(range(1, 265))

    if article_numbers != expected:
        missing = [
            number
            for number in expected
            if number not in article_numbers
        ]

        duplicates = [
            number
            for number in sorted(set(article_numbers))
            if article_numbers.count(number) > 1
        ]

        if missing:
            errors.append(
                "Missing article numbers: "
                + ", ".join(map(str, missing))
            )

        if duplicates:
            errors.append(
                "Duplicate article numbers: "
                + ", ".join(map(str, duplicates))
            )

        if article_numbers:
            errors.append(
                "Article sequence does not match expected "
                "sequence 1–264."
            )
        else:
            errors.append(
                "No article headings detected."
            )

    return errors


def validate_forbidden_content(text: str) -> list[str]:
    errors = []

    forbidden_patterns = {
        "page markers": r"--- PAGE \d+ ---",
        "table of contents": r"(?m)^Contents$",
        "publisher blurb": r"Legislation as at",
        "fi ligature": "\ufb01",
        "fl ligature": "\ufb02",
        "ff ligature": "\ufb00",
        "ffi ligature": "\ufb03",
        "ffl ligature": "\ufb04",
    }

    for name, pattern in forbidden_patterns.items():
        if re.search(pattern, text):
            errors.append(
                f"Forbidden content still present: {name}"
            )

    return errors


def validate_structure(text: str) -> list[str]:
    errors = []

    if not text.startswith("PREAMBLE"):
        errors.append("Document does not begin with PREAMBLE")

    if "Chapter ONE" not in text:
        errors.append("Chapter ONE not detected")

    if "Chapter Two" not in text:
        errors.append("Chapter Two not detected")

    if "FIRST SCHEDULE" not in text:
        errors.append("FIRST SCHEDULE not detected")

    if "SIXTH SCHEDULE" not in text:
        errors.append("SIXTH SCHEDULE not detected")

    return errors


def validate_suspicious_characters(text: str) -> list[str]:
    errors = []

    suspicious = []

    for index, char in enumerate(text):
        codepoint = ord(char)

        # Allow normal ASCII plus common Unicode punctuation.
        if codepoint < 32 and char not in "\n\t":
            suspicious.append(
                f"U+{codepoint:04X}"
            )

    if suspicious:
        unique = sorted(set(suspicious))

        errors.append(
            "Suspicious control characters found: "
            + ", ".join(unique)
        )

    return errors


def validate_empty_lines(text: str) -> list[str]:
    errors = []

    lines = text.splitlines()

    if not lines:
        errors.append("Document is empty")
        return errors

    consecutive_blank_lines = re.search(
        r"\n\s*\n\s*\n",
        text,
    )

    if consecutive_blank_lines:
        errors.append(
            "Multiple consecutive blank lines detected"
        )

    return errors


def validate_document(path: Path) -> bool:
    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    text = path.read_text(encoding="utf-8")

    errors = []

    errors.extend(validate_required_content(text))
    errors.extend(validate_article_sequence(text))
    errors.extend(validate_forbidden_content(text))
    errors.extend(validate_structure(text))
    errors.extend(validate_suspicious_characters(text))
    errors.extend(validate_empty_lines(text))

    print()
    print("=" * 60)
    print("POCKET LAWYER — CONSTITUTION VALIDATION")
    print("=" * 60)
    print()

    print(f"File: {path}")
    print(f"Characters: {len(text):,}")
    print(f"Lines: {len(text.splitlines()):,}")

    article_section = text.split("FIRST SCHEDULE", 1)[0]

    article_matches = re.findall(
        r"(?m)^(\d{1,3})\.\s*$",
        article_section,
    )

    print(f"Article headings detected: {len(article_matches)}")

    print()
    print("-" * 60)

    if errors:
        print("VALIDATION FAILED")
        print()
        print(f"{len(errors)} issue(s) found:")
        print()

        for error in errors:
            print(f"[FAIL] {error}")

        print()
        print("=" * 60)

        return False

    print("VALIDATION PASSED")
    print()
    print("[PASS] Required content present")
    print("[PASS] Article sequence valid")
    print("[PASS] Forbidden content removed")
    print("[PASS] Document structure detected")
    print("[PASS] Character validation passed")
    print("[PASS] Formatting validation passed")

    print()
    print("=" * 60)

    return True


if __name__ == "__main__":
    cleaned_path = Path(
        "data/cleaned/constitution_cleaned.txt"
    )

    success = validate_document(cleaned_path)

    if not success:
        raise SystemExit(1)