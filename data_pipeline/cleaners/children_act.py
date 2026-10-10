"""
Cleaning configuration for the Children Act (Cap. 141).

The standard Kenya Law shape. At 122 pages this is the largest document in
the corpus, and its six Schedules are excluded by `structure/act.py`, which
stops at the first schedule heading.

Note the Act is the 2022 enactment (Act No. 29 of 2022) consolidated as at
31 December 2022, and Kenya Law cites it as Cap. 141.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    # The cover reads "THE CHILDREN ACT"; the leading "THE" is what stops
    # `^CHILDREN ACT` matching it, so this anchors on the body block only.
    start_pattern=(
        r"(?m)^CHILDREN ACT\s*\nCAP\. 141\s*\nPublished in Kenya Gazette"
    ),
    # No trailing index; the Act ends with its Schedules, which the
    # structure adapter drops.
    end_pattern=None,
    running_header_pattern=(
        r"(?m)^Children Act \(Cap\. 141\)[ \t]*\n[ \t]*Kenya[ \t]*\n"
    ),
    verify_markers={
        "act title": "CHILDREN ACT",
        "section 1": "Short title",
        "section 2": "Interpretation",
        "section 8": "Best interests of the child",
        "first part": "Part I",
        "custody part": "Part IX",
        "last part": "Part XVI",
    },
)
