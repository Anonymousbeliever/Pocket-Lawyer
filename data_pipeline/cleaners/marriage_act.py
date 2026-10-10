"""
Cleaning configuration for the Marriage Act (Cap. 150).

Standard Kenya Law shape with one thing that needs handling here rather than
in the structure adapter.

This Act's single schedule is headed **"SCHEDULE [s. 97]"**, not "FIRST
SCHEDULE". `structure/act.py` stops at the first `<ORDINAL> SCHEDULE`
heading, so a bare "SCHEDULE" does not stop it, and the schedule's contents
would be appended to whichever section was last open. The schedule is three
lines of repealed-act listing, so the pollution is small - but it is law
attributed to the wrong section, which is exactly what this project does not
do. `end_pattern` cuts it instead.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    # The cover reads "THE MARRIAGE ACT"; the leading "THE" excludes it.
    start_pattern=(
        r"(?m)^MARRIAGE ACT\s*\nCAP\. 150\s*\nPublished in Kenya Gazette"
    ),
    # The lone schedule, which act.py's ordinal-prefixed pattern misses.
    end_pattern=r"(?m)^SCHEDULE\s*\[s\.\s*97\]",
    running_header_pattern=(
        r"(?m)^Marriage Act \(Cap\. 150\)[ \t]*\n[ \t]*Kenya[ \t]*\n"
    ),
    verify_markers={
        "act title": "MARRIAGE ACT",
        "section 1": "Short title",
        "section 3": "Meaning of marriage",
        "section 11": "Void marriages",
        "section 12": "Voidable marriages",
        "first part": "Part I",
        "divorce part": "Part X",
        "last part": "Part XIV",
    },
)
