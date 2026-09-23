"""
Cleaning configuration for the Employment Act (Cap. 226).

The same Kenya Law shape as the Criminal Procedure Code and the Penal Code -
cover, licence page, a table of contents, and a running header on every body
page - but the patterns name this document, so it needs its own spec.

One difference worth noting: this Act has no trailing index, so `end_pattern`
stays None. The operative text runs to section 92 and stops.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    # The operative text begins after a fourteen-page table of contents. Two
    # near-misses make this block the only safe anchor:
    #
    #   "THE EMPLOYMENT ACT"  the cover, which the leading "THE" excludes
    #   "Employment Act"      the licence page, which case excludes
    #
    # The third line differs from the Penal Code's, which was assented to
    # before publication; this Act was gazetted first.
    start_pattern=(
        r"(?m)^EMPLOYMENT ACT\s*\nCAP\. 226\s*\nPublished in Kenya Gazette"
    ),
    # No trailing index. Unlike the Penal Code, this Act ends at its last
    # section, so there is nothing to cut and naming a marker that does not
    # exist would raise.
    end_pattern=None,
    running_header_pattern=(
        r"(?m)^Employment Act \(Cap\. 226\)[ \t]*\n[ \t]*Kenya[ \t]*\n"
    ),
    verify_markers={
        "act title": "EMPLOYMENT ACT",
        # Part 1 is Arabic where every later Part is Roman - a quirk of the
        # source, not a typo here. See `structure/act.py`.
        "first part": "Part 1",
        "second part": "Part II",
        "last part": "Part XIII",
        "section 1": "Short title",
        "section 2": "Interpretation",
        "section 35": "Termination notice",
        "section 45": "Unfair termination",
        "section 90": "Rules",
    },
)
