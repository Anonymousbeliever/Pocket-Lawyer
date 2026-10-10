"""
Cleaning configuration for the Law of Succession Act (Cap. 160).

The usual Kenya Law shape - cover, licence page, table of contents, and a
running header on every body page - with the patterns naming this document,
so it needs its own spec.

Like the Employment Act and unlike the Penal Code, there is no trailing
index, so `end_pattern` stays None. The Act's nine Schedules are left in the
cleaned text and excluded later: `structure/act.py` stops at the first
schedule heading, because linear PDF extraction shreds their tables into one
line per cell.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    # The operative text begins on page 7, after a six-page table of
    # contents. Two decoys sit above it and both are excluded by case:
    #
    #   "THE LAW OF SUCCESSION ACT"  the cover, where the leading "THE"
    #                                stops `^LAW OF SUCCESSION ACT` matching
    #   "Law of Succession Act"      the licence page and running header
    #
    start_pattern=(
        r"(?m)^LAW OF SUCCESSION ACT\s*\nCAP\. 160\s*\n"
        r"Published in Kenya Gazette"
    ),
    # No trailing index; the Act ends with its Schedules.
    end_pattern=None,
    running_header_pattern=(
        r"(?m)^Law of Succession Act \(Cap\. 160\)[ \t]*\n[ \t]*Kenya[ \t]*\n"
    ),
    verify_markers={
        "act title": "LAW OF SUCCESSION ACT",
        # Sections 1-4 belong to NO Part - this Act's divisions begin at
        # Part II. See `structure/act.py`.
        "section 1": "Short title",
        "section 3": "Interpretation",
        "first part": "Part II",
        "last part": "Part VIII",
        "section 5": "Persons capable of making wills",
        "section 34": "Meaning of intestacy",
        "section 45": "No intermeddling with property of deceased person",
        "section 96": "Sane murderer not to share in victim's estate",
    },
)
