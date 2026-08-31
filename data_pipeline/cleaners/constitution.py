"""
Cleaning configuration for the Constitution of Kenya, 2010.

The Kenya Law PDF opens with publisher front matter and a table of
contents, and repeats a "Constitution of Kenya / Kenya" running header
on every page. Everything else is handled generically in `base`.
"""

from data_pipeline.cleaners.base import CleanerSpec


SPEC = CleanerSpec(
    start_pattern=r"(?m)^PREAMBLE\s*\nWe, the people of Kenya",
    running_header_pattern=(
        r"(?m)^Constitution of Kenya\s*\n\s*Kenya\s*\n"
    ),
    verify_markers={
        "preamble opening": "We, the people of Kenya",
        "preamble closing": "GOD BLESS KENYA",
        "article 1": "Sovereignty of the people",
        "article 49": "Rights of arrested persons",
        "article 264": "Repeal of previous Constitution",
        "first schedule": "FIRST SCHEDULE",
        "sixth schedule": "SIXTH SCHEDULE",
    },
)
