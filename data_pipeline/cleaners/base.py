"""
Generic cleaning operations for extracted legal text.

Everything here applies to any PDF-extracted Kenyan legal document.
Document-specific knowledge (where the operative text begins, which
running header repeats on every page, which markers prove the clean
worked) lives in a `CleanerSpec` supplied by the type adapter.
"""

import re
from dataclasses import dataclass, field


LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    " ": " ",
    "­": "",
}


@dataclass
class CleanerSpec:
    """Per-document-type cleaning configuration."""

    # Text before the first match is publisher front matter and is cut.
    #
    # Optional: the Constitution opens with an unmistakable "PREAMBLE /
    # We, the people of Kenya", but an Act has no equivalent single
    # marker. When None the document is kept whole.
    start_pattern: str | None = None

    # Text from the first match onward is trailing matter and is cut.
    #
    # Added for the Penal Code, which ends with an alphabetical index the
    # document itself disclaims: "This index is not part of the Act, and is
    # inserted only for convenience." Indexing it would mean embedding
    # thousands of page-number fragments as though they were law.
    end_pattern: str | None = None

    # Running header repeated on every page, if any.
    running_header_pattern: str | None = None

    # Substrings that must survive cleaning. Cheap insurance against a
    # regex quietly eating half the document.
    verify_markers: dict[str, str] = field(default_factory=dict)


def clean(text: str, spec: CleanerSpec) -> str:
    """Clean extracted text according to a document type's spec."""

    if spec.start_pattern:
        text = cut_to_start(text, spec.start_pattern)

    if spec.end_pattern:
        text = cut_at_end(text, spec.end_pattern)

    text = remove_footer_page_numbers(text)
    text = remove_page_markers(text)

    if spec.running_header_pattern:
        text = re.sub(spec.running_header_pattern, "", text)

    text = normalize_unicode(text)
    text = join_hyphenated_linebreaks(text)
    text = normalize_lines(text)

    cleaned = text.strip() + "\n"

    verify(cleaned, spec)

    return cleaned


# ---------------------------------------------------------
# OPERATIONS
# ---------------------------------------------------------

def cut_to_start(text: str, start_pattern: str) -> str:
    match = re.search(start_pattern, text)

    if not match:
        raise ValueError(
            "Could not find the document start marker "
            f"({start_pattern!r}) in the extract."
        )

    return text[match.start():]


def cut_at_end(text: str, end_pattern: str) -> str:
    """
    Drop everything from the first match onward.

    Raises when the marker is absent, for the same reason `cut_to_start`
    does: a spec that names trailing matter and then silently fails to find
    it would ship the trailing matter as law.
    """

    match = re.search(end_pattern, text)

    if not match:
        raise ValueError(
            "Could not find the document end marker "
            f"({end_pattern!r}) in the extract."
        )

    return text[: match.start()]


def remove_page_markers(text: str) -> str:
    return re.sub(r"\n---\s*PAGE\s*\d+\s*---\n", "\n", text)


def remove_footer_page_numbers(text: str) -> str:
    # Page footers sit on their own line immediately before a page marker.
    text = re.sub(
        r"\n[ \t]*\d+[ \t]*\n+(?=--- PAGE \d+ ---)",
        "\n",
        text,
    )

    # Final page footer at end of file.
    text = re.sub(r"\n[ \t]*\d+[ \t]*\s*$", "\n", text)

    return text


def normalize_unicode(text: str) -> str:
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)

    return text


def join_hyphenated_linebreaks(text: str) -> str:
    return re.sub(r"([A-Za-z]+)-\n([a-z]+)", r"\1-\2", text)


def normalize_lines(text: str) -> str:
    """Strip trailing space, drop repeated lines, collapse blank runs."""

    lines = [line.rstrip() for line in text.splitlines()]

    collapsed: list[str] = []
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


def verify(cleaned: str, spec: CleanerSpec) -> None:
    missing = [
        name
        for name, marker in spec.verify_markers.items()
        if marker not in cleaned
    ]

    if missing:
        raise ValueError(
            "Cleaned document is missing expected content: "
            + ", ".join(missing)
        )
