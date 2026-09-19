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


def mixed(*spec: tuple[str, str]) -> list[dict]:
    """Chunks from named documents, in the order their scores are given."""

    return [
        {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "content": f"legal text {chunk_id}",
        }
        for document_id, chunk_id in spec
    ]


def ids(results: list[dict]) -> list[str]:
    return [result["chunk_id"] for result in results]


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
# ONE DOCUMENT MAY NOT TAKE EVERY SLOT
#
# Both scenarios below are real, measured after the Criminal Procedure
# Code was indexed alongside the Constitution. In each the correct
# authority was retrieved and then squeezed out of the top-k by a
# single document.
# ---------------------------------------------------------

def test_arrest_silence_regression():
    """
    "Do I have to answer police questions after being arrested?"

    s.36A (Remand by court) scored far above everything because its text
    contains "inquiries ... by the police". Article 49 was retrieved at
    #2 and then cut, leaving the LLM one irrelevant source — so it
    refused a question the corpus answers.

    The Act's best passage still wins. The Constitution's best is no
    longer discarded with it.
    """

    reranker = make_reranker([0.85, 0.02, 0.01, 0.008, 0.005])

    results = reranker.rerank(
        query="Do I have to answer police questions after being arrested?",
        documents=mixed(
            ("criminal-procedure-code", "cpc-36a"),
            ("constitution-of-kenya-2010", "article-49"),
            ("criminal-procedure-code", "cpc-35"),
            ("criminal-procedure-code", "cpc-36"),
            ("criminal-procedure-code", "cpc-25"),
        ),
    )

    assert ids(results) == ["cpc-36a", "article-49"]


def test_arrest_reason_regression():
    """
    "Can police arrest me without telling me why?"

    Five Criminal Procedure Code sections on arrest powers filled every
    slot and the system answered from them, citing real sections that
    do not address the duty to give reasons.
    """

    reranker = make_reranker([0.9, 0.8, 0.7, 0.6, 0.5, 0.05])

    results = reranker.rerank(
        query="Can police arrest me without telling me why?",
        documents=mixed(
            ("criminal-procedure-code", "cpc-29"),
            ("criminal-procedure-code", "cpc-31"),
            ("criminal-procedure-code", "cpc-64"),
            ("criminal-procedure-code", "cpc-2"),
            ("criminal-procedure-code", "cpc-345"),
            ("constitution-of-kenya-2010", "article-49"),
        ),
    )

    assert len(results) == 5
    assert "article-49" in ids(results)

    # The reserved slot costs the Act its weakest passage, not its best.
    assert "cpc-29" in ids(results)
    assert "cpc-345" not in ids(results)


def test_only_one_slot_is_reserved():
    """
    The shut-out document gets its best passage, not a foothold for its
    whole tail. Otherwise a below-cut document could crowd out real
    matches.
    """

    reranker = make_reranker([0.9, 0.8, 0.7, 0.6, 0.05, 0.04, 0.03])

    results = reranker.rerank(
        query="a question",
        documents=mixed(
            ("act", "act-1"),
            ("act", "act-2"),
            ("act", "act-3"),
            ("act", "act-4"),
            ("constitution", "con-1"),
            ("constitution", "con-2"),
            ("constitution", "con-3"),
        ),
    )

    from_constitution = [
        r for r in results if r["document_id"] == "constitution"
    ]

    assert len(from_constitution) == 1
    assert from_constitution[0]["chunk_id"] == "con-1"


def test_a_document_that_earns_its_slots_keeps_them():
    """Diversity caps one document; it does not force an even split."""

    reranker = make_reranker([0.9, 0.85, 0.8, 0.75, 0.7])

    results = reranker.rerank(
        query="a question",
        documents=mixed(
            ("act", "act-1"),
            ("constitution", "con-1"),
            ("act", "act-2"),
            ("constitution", "con-2"),
            ("act", "act-3"),
        ),
    )

    assert ids(results) == ["act-1", "con-1", "act-2", "con-2", "act-3"]


def test_results_stay_ordered_by_score_after_diversity():
    reranker = make_reranker([0.9, 0.8, 0.7, 0.6, 0.02])

    results = reranker.rerank(
        query="a question",
        documents=mixed(
            ("act", "act-1"),
            ("act", "act-2"),
            ("act", "act-3"),
            ("act", "act-4"),
            ("constitution", "con-1"),
        ),
    )

    scores = [result["rerank_score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_single_document_corpus_is_unaffected():
    """
    Behaviour before the Criminal Procedure Code was indexed must be
    bit-identical: with one document there is nothing to diversify, and
    capping slots would only discard good matches.
    """

    reranker = make_reranker([0.9, 0.8, 0.7, 0.6, 0.5])

    results = reranker.rerank(
        query="a question",
        documents=mixed(
            ("constitution", "con-1"),
            ("constitution", "con-2"),
            ("constitution", "con-3"),
            ("constitution", "con-4"),
            ("constitution", "con-5"),
        ),
    )

    assert len(results) == 5


def test_max_per_document_can_be_overridden():
    reranker = make_reranker([0.9, 0.8, 0.7, 0.6, 0.5])

    results = reranker.rerank(
        query="a question",
        documents=mixed(
            ("act", "act-1"),
            ("act", "act-2"),
            ("act", "act-3"),
            ("act", "act-4"),
            ("constitution", "con-1"),
        ),
        max_per_document=2,
    )

    assert ids(results) == ["act-1", "act-2", "con-1"]


# ---------------------------------------------------------
# BATCHING ACROSS QUESTIONS
#
# The evaluation harness scores every question in one cross-encoder call
# rather than one call each. It must return exactly what the sequential path
# returns - a faster harness that quietly changes results is worthless.
# ---------------------------------------------------------

def test_batched_reranking_equals_sequential():
    questions = [
        ("What are my rights if I am arrested?", mixed(
            ("constitution", "article-49"),
            ("criminal-procedure-code", "cpc-29"),
            ("constitution", "article-51"),
        )),
        ("Can I be released on bail?", mixed(
            ("criminal-procedure-code", "cpc-123"),
            ("constitution", "article-49"),
        )),
        ("What is robbery?", mixed(
            ("penal-code", "penal-295"),
            ("penal-code", "penal-296"),
            ("criminal-procedure-code", "cpc-29"),
        )),
    ]

    # One score per (query, document) pair, in the order a single batched
    # predict would see them: 3 + 2 + 3.
    per_query = [[0.9, 0.4, 0.2], [0.8, 0.05], [0.95, 0.7, 0.01]]

    sequential = [
        make_reranker(scores).rerank(query, documents)
        for (query, documents), scores in zip(questions, per_query)
    ]

    flat = [score for scores in per_query for score in scores]

    batched = make_reranker(flat).rerank_many(questions)

    assert [ids(r) for r in batched] == [ids(r) for r in sequential]


def test_batching_keeps_per_document_diversity():
    """The reserved slot is applied per query, not across the batch."""

    batched = make_reranker([0.9, 0.8, 0.02, 0.9, 0.01]).rerank_many(
        [
            ("first", mixed(
                ("act", "a-1"),
                ("act", "a-2"),
                ("constitution", "c-1"),
            )),
            ("second", mixed(
                ("act", "a-3"),
                ("constitution", "c-2"),
            )),
        ],
        top_k=2,
    )

    assert "c-1" in ids(batched[0])
    assert "c-2" in ids(batched[1])


def test_batching_handles_empty_and_blank_entries():
    batched = make_reranker([0.9]).rerank_many(
        [
            ("a real question", mixed(("act", "a-1"))),
            ("", mixed(("act", "a-2"))),
            ("no documents", []),
        ]
    )

    assert ids(batched[0]) == ["a-1"]
    assert batched[1] == []
    assert batched[2] == []


def test_batching_nothing_returns_nothing():
    assert make_reranker([]).rerank_many([]) == []


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
