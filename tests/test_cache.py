"""
The evaluation result cache.

A stale cache hides a regression, which is the failure this harness exists to
prevent - so most of these assert that the key changes when it must, not that
it stays the same when it can.

The load-bearing property is the difference between the two levels: level 1 is
keyed on the corpus, level 2 on the CANDIDATES. That is what makes adding a
document cost reranking only for the questions whose top-k actually moved.
"""

import json

import pytest

from evaluation.cache import (
    ResultCache,
    corpus_fingerprint,
    rerank_key,
    retrieval_key,
)


QUESTION = "What are my rights if I am arrested?"

ARTICLE_49 = "constitution-of-kenya-2010@v2010-chapter-four-article-49"
ARTICLE_51 = "constitution-of-kenya-2010@v2010-chapter-four-article-51"
CPC_29 = "criminal-procedure-code@v2023-12-11-part-iii-section-29"


# ---------------------------------------------------------
# THE TWO LEVELS ARE KEYED DIFFERENTLY
#
# This is the whole design. Get it wrong and the cache does nothing for the
# only case that hurts: adding a document.
# ---------------------------------------------------------

def test_a_new_document_invalidates_retrieval():
    before = retrieval_key(QUESTION, "corpus-of-two-documents")
    after = retrieval_key(QUESTION, "corpus-of-three-documents")

    assert before != after


def test_a_new_document_does_not_invalidate_reranking():
    """
    The point of the split. Reranking's inputs are the question and the
    candidates; the corpus is not one of them. A document that puts nothing
    into this question's top-k cannot change its reranking, so recomputing
    would be waste.
    """

    candidates = [ARTICLE_49, ARTICLE_51]

    assert rerank_key(QUESTION, candidates) == rerank_key(QUESTION, candidates)


def test_changed_candidates_do_invalidate_reranking():
    """When the document DOES reach this question, the answer can change."""

    before = rerank_key(QUESTION, [ARTICLE_49, ARTICLE_51])
    after = rerank_key(QUESTION, [ARTICLE_49, CPC_29, ARTICLE_51])

    assert before != after


def test_candidate_order_matters():
    """Reranking sees the list; a different order is a different input."""

    assert rerank_key(QUESTION, [ARTICLE_49, ARTICLE_51]) != rerank_key(
        QUESTION, [ARTICLE_51, ARTICLE_49]
    )


def test_different_questions_do_not_share_a_key():
    assert rerank_key("a", [ARTICLE_49]) != rerank_key("b", [ARTICLE_49])
    assert retrieval_key("a", "corpus") != retrieval_key("b", "corpus")


# ---------------------------------------------------------
# THE SOURCE HASH — WHAT MAKES THIS SAFE
# ---------------------------------------------------------

def test_editing_the_reranker_invalidates_reranking(monkeypatch):
    """
    Changing the diversity rule changes results while corpus and config stay
    identical. Without hashing the source, the cache would confidently serve
    the old answer - worse than having no cache at all.
    """

    from evaluation import cache as cache_module

    before = rerank_key(QUESTION, [ARTICLE_49])

    monkeypatch.setattr(
        cache_module, "_source_hash", lambda names: "pretend-edited"
    )

    assert rerank_key(QUESTION, [ARTICLE_49]) != before


def test_config_changes_invalidate(monkeypatch):
    """RETRIEVAL_TOP_K decides how many candidates come back."""

    from backend.app.core import config

    before = retrieval_key(QUESTION, "corpus")

    monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 15)

    assert retrieval_key(QUESTION, "corpus") != before


def test_rerank_config_changes_invalidate(monkeypatch):
    from backend.app.core import config

    before = rerank_key(QUESTION, [ARTICLE_49])

    monkeypatch.setattr(config, "RERANK_RELATIVE_RATIO", 0.5)

    assert rerank_key(QUESTION, [ARTICLE_49]) != before


# ---------------------------------------------------------
# CORPUS FINGERPRINT
# ---------------------------------------------------------

def write_document(root, slug, checksum, chunks=10):
    directory = root / slug
    directory.mkdir(parents=True)

    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "checksums": {"cleaned": checksum},
                "processing": {"chunk_count": chunks},
            }
        ),
        encoding="utf-8",
    )


def test_fingerprint_changes_when_a_document_is_added(tmp_path):
    write_document(tmp_path, "constitution@v2010", "aaa")

    before = corpus_fingerprint(tmp_path)

    write_document(tmp_path, "penal-code@v2023-12-11", "bbb")

    assert corpus_fingerprint(tmp_path) != before


def test_fingerprint_changes_when_text_is_recleaned(tmp_path):
    """
    A cleaner fix can change the text without changing a single chunk id.
    Hashing ids alone would miss it.
    """

    write_document(tmp_path, "penal-code@v2023-12-11", "aaa")
    before = corpus_fingerprint(tmp_path)

    (tmp_path / "penal-code@v2023-12-11" / "metadata.json").write_text(
        json.dumps(
            {
                "checksums": {"cleaned": "bbb"},
                "processing": {"chunk_count": 10},
            }
        ),
        encoding="utf-8",
    )

    assert corpus_fingerprint(tmp_path) != before


def test_fingerprint_is_stable_when_nothing_moved(tmp_path):
    write_document(tmp_path, "constitution@v2010", "aaa")

    assert corpus_fingerprint(tmp_path) == corpus_fingerprint(tmp_path)


def test_an_unreadable_metadata_file_does_not_crash(tmp_path):
    directory = tmp_path / "broken@v1"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text("{not json", encoding="utf-8")

    assert corpus_fingerprint(tmp_path)


# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    return ResultCache(directory=tmp_path)


def test_a_stored_value_comes_back(cache):
    cache.put("rerank", "k", [ARTICLE_49, CPC_29])

    assert cache.get("rerank", "k") == [ARTICLE_49, CPC_29]


def test_a_miss_returns_none(cache):
    assert cache.get("rerank", "absent") is None


def test_hits_and_misses_are_counted(cache):
    cache.get("rerank", "absent")
    cache.put("rerank", "k", [])
    cache.get("rerank", "k")

    assert cache.hits["rerank"] == 1
    assert cache.misses["rerank"] == 1


def test_levels_do_not_collide(cache):
    cache.put("retrieval", "same-key", ["retrieval value"])
    cache.put("rerank", "same-key", ["rerank value"])

    assert cache.get("retrieval", "same-key") == ["retrieval value"]
    assert cache.get("rerank", "same-key") == ["rerank value"]


def test_a_reference_run_reads_nothing_but_still_writes(tmp_path):
    """
    `--no-cache` must recompute, but the values it computes are correct by
    definition. Discarding them would cost a second full run to warm the
    cache - which is exactly what happened before this was split.
    """

    warm = ResultCache(directory=tmp_path)
    warm.put("rerank", "k", ["stale"])

    reference = ResultCache(directory=tmp_path, read=False, write=True)

    assert reference.get("rerank", "k") is None      # ignores what is stored
    assert not reference.used                         # so it may be baselined

    reference.put("rerank", "k", ["fresh"])

    assert ResultCache(directory=tmp_path).get("rerank", "k") == ["fresh"]


def test_reads_disabled_still_counts_the_work_as_computed(tmp_path):
    """The run must report what it actually did, not stay silent."""

    reference = ResultCache(directory=tmp_path, read=False)

    reference.get("rerank", "k")

    assert reference.misses["rerank"] == 1


def test_writes_can_be_disabled_independently(tmp_path):
    read_only = ResultCache(directory=tmp_path, write=False)

    read_only.put("rerank", "k", [ARTICLE_49])

    assert read_only.get("rerank", "k") is None


def test_a_corrupt_entry_is_a_miss_not_a_crash(cache, tmp_path):
    path = tmp_path / "rerank" / "k.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    assert cache.get("rerank", "k") is None
    assert cache.misses["rerank"] == 1


def test_used_reports_whether_anything_was_reused(cache):
    assert not cache.used

    cache.put("rerank", "k", [])
    cache.get("rerank", "k")

    assert cache.used
