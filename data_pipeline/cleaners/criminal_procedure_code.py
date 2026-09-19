"""
Cleaning configuration for the Criminal Procedure Code (Cap. 75).

The Kenya Law PDF opens with a cover, a publisher/licence page and a
thirteen-page table of contents, and repeats a
"Criminal Procedure Code (Cap. 75) / Kenya" running header on all 208
body pages. Everything else is handled generically in `base`.

This spec is per-document rather than per-type: both patterns embed the
document's own title, and `CleanerSpec` is static data with no access to
the registry entry, so it cannot be derived for Acts in general. A new
Act therefore needs a cleaner spec of its own even though it can reuse
`structure/act.py` unchanged.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    # The operative text begins on page 16. This anchor is unique in the
    # document: the cover page reads "THE CRIMINAL PROCEDURE CODE ACT",
    # which does not match. Cutting here removes 17% of the file —
    # cover, licence blurb, and the whole table of contents.
    start_pattern=(
        r"(?m)^CRIMINAL PROCEDURE CODE\s*\nCAP\. 75\s*\n"
        r"Published in Kenya Gazette"
    ),
    running_header_pattern=(
        r"(?m)^Criminal Procedure Code \(Cap\. 75\)[ \t]*\n[ \t]*Kenya[ \t]*\n"
    ),
    verify_markers={
        "act title": "CRIMINAL PROCEDURE CODE",
        "first part": "Part I",
        "last part": "Part XII",
        "section 1": "Short title",
        "section 36A": "Remand by court",
        "section 394": "Expenses of assessors, witnesses, etc.",
    },
)
