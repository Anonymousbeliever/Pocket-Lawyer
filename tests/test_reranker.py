"""
The reranker trims the weak tail but never refuses on its own.

Absolute cross-encoder scores are not comparable between queries, so
they cannot gate sufficiency. Deciding whether the sources answer the
question belongs to the LLM's grounding prompt.

These tests stub the cross-encoder — loading the real model costs
~1 GB of RAM and is not what is under test.
"""

from backend.app.ai.reranker import LegalReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores[:len(pairs)]


def make_reranker(scores) -> LegalReranker:
    reranker = object.__new__(LegalReranker)
    reranker.model_name = "fake"
    reranker.model = FakeCrossEncoder(scores)
    return reranker


def documents(count: int) -> list[dict]:
    return [
        {"chunk_id": f"chunk-{n}", "content": f"legal text {n}"}
        for n in range(count)
    ]


# ---------------------------------------------------------
# TAIL TRIMMING
# ---------------------------------------------------------

def test_weak_tail_is_dropped():
    """The real spread for a well-phrased query."""

    reranker = make_reranker([0.9993, 0.9392, 0.2933])

    results = reranker.rerank(
        query="What are the rights of an arrested person?",
        documents=documents(3),
        relative_ratio=0.5,
    )

    assert len(results) == 2
    assert all(r["rerank_score"] >= 0.9993 * 0.5 for r in results)


def test_results_are_ordered_by_score():
    reranker = make_reranker([0.5, 0.99, 0.7])

    results = reranker.rerank(
        query="a question",
        documents=documents(3),
    )

    scores = [result["rerank_score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_caps_the_result_count():
    reranker = make_reranker([0.9, 0.8, 0.7, 0.6, 0.5])

    results = reranker.rerank(
        query="a question",
        documents=documents(5),
        top_k=2,
    )

    assert len(results) == 2


# ---------------------------------------------------------
# NEVER REFUSES ON ITS OWN
#
# Scores below were measured against the indexed Constitution. The
# gate that used to live here rejected a question the corpus answers,
# because a typo moved the correct source 15x while it still ranked
# first.
# ---------------------------------------------------------

def test_best_match_always_survives():
    """No score is low enough to lose the top result."""

    reranker = make_reranker([0.0086, 0.0011, 0.0004])

    results = reranker.rerank(
        query="Can police arrest me without teling me why?",
        documents=documents(3),
    )

    assert results
    assert results[0]["chunk_id"] == "chunk-0"


def test_typod_question_still_returns_its_source():
    """
    Regression: "Can police arrest me without teling me why?"

    Article 49 still ranks first, but scores 0.0086 — below the 0.0373
    that an unanswerable question scored. An absolute floor refused
    this; a relative cut cannot.
    """

    reranker = make_reranker([0.0086, 0.0011, 0.0004, 0.0002, 0.0001])

    results = reranker.rerank(
        query="Can police arrest me without teling me why?",
        documents=documents(5),
    )

    assert len(results) >= 1
    assert results[0]["rerank_score"] == 0.0086


def test_unanswerable_question_still_returns_candidates():
    """
    The corpus cannot answer this, and that judgement is the LLM's to
    make from the text — not the reranker's to make from a score.
    """

    reranker = make_reranker([0.0373, 0.0300, 0.0231, 0.0174, 0.0158])

    results = reranker.rerank(
        query="What is the legal process for filing for divorce in Kenya?",
        documents=documents(5),
    )

    assert results


def test_scores_near_the_top_are_all_kept():
    reranker = make_reranker([0.9993, 0.8538, 0.8085, 0.1635, 0.0709])

    results = reranker.rerank(
        query="What are the rights of an arrested person?",
        documents=documents(5),
    )

    # 0.0709 falls below 0.9993 * 0.10
    assert len(results) == 4
    assert min(r["rerank_score"] for r in results) == 0.1635


# ---------------------------------------------------------
# BASICS
# ---------------------------------------------------------

def test_empty_document_list_is_handled():
    reranker = make_reranker([])

    assert reranker.rerank(query="a question", documents=[]) == []


def test_original_documents_are_not_mutated():
    reranker = make_reranker([0.9])

    docs = documents(1)

    reranker.rerank(query="a question", documents=docs)

    assert "rerank_score" not in docs[0]
