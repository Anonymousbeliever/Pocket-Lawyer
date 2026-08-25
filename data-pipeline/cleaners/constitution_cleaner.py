import re
from pathlib import Path

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00a0": " ",
    "\u00ad": "",
}


def clean_constitution(text: str) -> str:
    """
    Clean the Kenya Law Constitution extract produced by PyMuPDF.

    Keeps the preamble and operative text. Drops publisher pages, the table
    of contents, running headers, page markers, and footer page numbers.
    """
    text = _cut_to_preamble(text)
    text = _remove_footer_page_numbers(text)
    text = re.sub(r"\n---\s*PAGE\s*\d+\s*---\n", "\n", text)
    text = re.sub(
        r"(?m)^Constitution of Kenya\s*\n\s*Kenya\s*\n",
        "",
        text,
    )
    text = _normalize_unicode(text)
    text = re.sub(r"([A-Za-z]+)-\n([a-z]+)", r"\1-\2", text)
    text = _normalize_lines(text)
    return text.strip() + "\n"


def _cut_to_preamble(text: str) -> str:
    match = re.search(
        r"(?m)^PREAMBLE\s*\nWe, the people of Kenya",
        text,
    )
    if not match:
        raise ValueError(
            "Could not find the Constitution preamble in the extract."
        )
    return text[match.start():]


def _remove_footer_page_numbers(text: str) -> str:
    # Page footers sit on their own line immediately before a page marker.
    text = re.sub(
        r"\n[ \t]*\d+[ \t]*\n+(?=--- PAGE \d+ ---)",
        "\n",
        text,
    )
    # Final page footer at end of file.
    text = re.sub(r"\n[ \t]*\d+[ \t]*\s*$", "\n", text)
    return text


def _normalize_unicode(text: str) -> str:
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)
    return text


def _normalize_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed = []
    blank = False
    prev = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if collapsed and not blank:
                collapsed.append("")
                blank = True
            continue
        if stripped == prev:
            continue
        collapsed.append(stripped)
        prev = stripped
        blank = False
    return "\n".join(collapsed)


def _verify(cleaned: str) -> None:
    checks = {
        "preamble opening": "We, the people of Kenya" in cleaned,
        "preamble close": "GOD BLESS KENYA" in cleaned,
        "article 1": "Sovereignty of the people" in cleaned,
        "article 49": "Rights of arrested persons" in cleaned,
        "article 264": "Repeal of previous Constitution" in cleaned,
        "first schedule": "FIRST SCHEDULE" in cleaned,
        "sixth schedule": "SIXTH SCHEDULE" in cleaned,
        "no page markers": "--- PAGE" not in cleaned,
        "no TOC page leaders": not re.search(r"\.{5,}\s+\d+", cleaned),
        "no fi ligature": "\ufb01" not in cleaned,
        "no fl ligature": "\ufb02" not in cleaned,
        "no publisher blurb": "Legislation as at" not in cleaned,
        "no contents heading": not re.search(r"(?m)^Contents$", cleaned),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("Cleaned constitution failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    raw_path = Path("data/extracted/constitution.txt")
    cleaned_path = Path("data/cleaned/constitution_cleaned.txt")

    if not raw_path.exists():
        raise SystemExit(f"Error: Input file '{raw_path}' not found.")

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_text = clean_constitution(raw_path.read_text(encoding="utf-8"))
    _verify(cleaned_text)
    cleaned_path.write_text(cleaned_text, encoding="utf-8")
    print(f"Successfully cleaned constitution saved to {cleaned_path}")
