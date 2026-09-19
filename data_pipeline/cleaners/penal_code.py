"""
Cleaning configuration for the Penal Code (Cap. 63).

Same Kenya Law shape as the Criminal Procedure Code - cover, licence page, a
fourteen-page table of contents, and a running header on every body page - but
the patterns name this document, so it needs its own spec. Everything else is
handled generically in `base`.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    # The operative text begins on page 17. Unique in the document: the cover
    # reads "THE PENAL CODE ACT", which does not match. Note the third line
    # differs from the Criminal Procedure Code's - this Act was assented to
    # rather than published in the Gazette first. Cutting here removes 23% of
    # the file: cover, licence blurb, and the whole table of contents.
    start_pattern=(
        r"(?m)^PENAL CODE\s*\nCAP\. 63\s*\nAssented to"
    ),
    # The Act ends at section 398, followed by an alphabetical index that the
    # document itself disclaims: "This index is not part of the Act, and is
    # inserted only for convenience." It is ~7,800 characters of entries like
    # "definition of ...... 256" - page-number fragments, not law.
    end_pattern=r"(?m)^INDEX\s*[–—-]\s*TO THE PENAL CODE",
    running_header_pattern=(
        r"(?m)^Penal Code \(Cap\. 63\)[ \t]*\n[ \t]*Kenya[ \t]*\n"
    ),
    verify_markers={
        "act title": "PENAL CODE",
        "first part": "Part I",
        "second part": "Part II",
        "first chapter": "Chapter I",
        "section 1": "Short title",
        "section 203": "Murder",
        "section 398": (
            "Punishment of accessories after the fact to misdemeanours"
        ),
    },
)
