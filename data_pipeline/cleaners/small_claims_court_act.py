"""
Cleaning configuration for the Small Claims Court Act (Cap. 10A).

The smallest document in the corpus at 18 pages, and the most
straightforward: six Parts, no Chapters, no Schedules, no trailing index.

Note the Chapter number carries a letter - Cap. 10A, inserted beside the
Magistrates' Courts Act at Cap. 10 - so the running header and start pattern
both have to allow for it.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    # The cover reads "THE SMALL CLAIMS COURT ACT"; the leading "THE"
    # excludes it.
    start_pattern=(
        r"(?m)^SMALL CLAIMS COURT ACT\s*\nCAP\. 10A\s*\n"
        r"Published in Kenya Gazette"
    ),
    # Nothing trails the last section.
    end_pattern=None,
    running_header_pattern=(
        r"(?m)^Small Claims Court Act \(Cap\. 10A\)[ \t]*\n[ \t]*Kenya[ \t]*\n"
    ),
    verify_markers={
        "act title": "SMALL CLAIMS COURT ACT",
        "section 1": "Short title",
        "section 2": "Interpretation",
        "section 12": "Nature of claims and pecuniary jurisdiction",
        "section 51": "Code of conduct for Adjudicators",
        "first part": "Part I",
        "jurisdiction part": "Part III",
        "last part": "Part VI",
    },
)
