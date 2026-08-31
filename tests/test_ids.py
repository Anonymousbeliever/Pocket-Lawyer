"""
Point IDs must be a function of chunk identity, never of position.

Positional IDs were the bug that made ingesting a second document
overwrite the first.
"""

import pytest

from data_pipeline.chunking.ids import point_id


def test_point_id_is_deterministic():
    first = point_id("constitution-of-kenya-2010-chapter-four-article-49")
    second = point_id("constitution-of-kenya-2010-chapter-four-article-49")

    assert first == second


def test_different_chunks_get_different_ids():
    ids = {
        point_id(f"constitution-of-kenya-2010-chapter-four-article-{n}")
        for n in range(1, 300)
    }

    assert len(ids) == 299


def test_documents_do_not_collide():
    """The regression that motivated this module."""

    constitution = point_id("constitution-of-kenya-2010-article-1")
    employment = point_id("employment-act-2007-section-1")

    assert constitution != employment


def test_empty_chunk_id_is_rejected():
    with pytest.raises(ValueError):
        point_id("   ")
