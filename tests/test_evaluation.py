"""
The evaluation harness itself.

A harness nobody has watched fail is not known to work, so these tests
mostly assert that it *detects* things: regressions, false refusals,
false answers, and question sets that reference chunks which do not
exist.
"""

import pytest

from evaluation.report import compare, print_run, summarize
from evaluation.schema import (
    Question,
    QuestionResult,
    load_questions,
    validate_against_corpus,
    validate_questions,
)


def result(**overrides) -> QuestionResult:
    defaults = dict(
        id="q1",
        answerable=True,
        retrieval_hit=True,
        rerank_hit=True,
        rerank_top1=True,
    )
    defaults.update(overrides)
    return QuestionResult(**defaults)


# ---------------------------------------------------------
# PASS CRITERIA
# ---------------------------------------------------------

def test_tier1_passes_when_expected_authority_survives_reranking():
    assert result(rerank_hit=True).passed(tier=1)


def test_tier1_fails_when_reranking_drops_the_authority():
    """The regression class introduced by the score threshold."""

    assert not result(retrieval_hit=True, rerank_hit=False).passed(tier=1)


def test_tier1_does_not_judge_out_of_scope_questions():
    """Refusal cannot be checked without the LLM, so tier 1 abstains."""

    out_of_scope = result(
        answerable=False,
        retrieval_hit=None,
        rerank_hit=None,
        rerank_top1=None,
    )

    assert out_of_scope.passed(tier=1)


def test_tier2_requires_both_retrieval_and_an_answer():
    assert result(sufficient=True).passed(tier=2)
    assert not result(sufficient=False).passed(tier=2)


def tier2_result(**overrides) -> QuestionResult:
    """
    Tier 2 cannot see the raw retrieved set - LegalRAG.answer() returns
    only the reranked sources - so retrieval_hit is None there.
    """

    defaults = dict(
        id="q1",
        answerable=True,
        retrieval_hit=None,
        rerank_hit=True,
        rerank_top1=True,
        sufficient=True,
    )
    defaults.update(overrides)
    return QuestionResult(**defaults)


def test_tier2_judges_correctly_without_retrieval_hit():
    assert tier2_result().passed(tier=2)
    assert not tier2_result(rerank_hit=False).passed(tier=2)
    assert not tier2_result(sufficient=False).passed(tier=2)


def test_tier2_never_claims_retrieval_failed(capsys):
    """
    Regression: tier 2 derived retrieval_hit from the reranked list,
    so a question that WAS retrieved (tier 1 found it at #9) was
    reported as "NEVER RETRIEVED". That single mislabelling produced
    three wrong diagnoses, including a phantom Qdrant bug.
    """

    failing = tier2_result(rerank_hit=False, rerank_top1=False)

    question = Question(
        id="q1",
        question="What are county governments responsible for?",
        answerable=True,
        expect_any_of=["some-chunk"],
    )

    summary = summarize([failing], tier=2)
    print_run(summary, [failing], {"q1": question}, tier=2)

    output = capsys.readouterr().out

    assert "NEVER RETRIEVED" not in output
    assert "NOT AMONG THE SOURCES SENT TO THE LLM" in output


def test_unobservable_retrieval_is_not_scored():
    """A metric nothing measured must not be reported as a total."""

    summary = summarize([tier2_result(), tier2_result(id="q2")], tier=2)

    assert summary["metrics"]["retrieval_hit"]["total"] == 0
    assert summary["metrics"]["rerank_hit"]["total"] == 2


def test_tier2_out_of_scope_passes_only_when_refused():
    refused = result(answerable=False, retrieval_hit=None,
                     rerank_hit=None, rerank_top1=None, sufficient=False)
    answered = result(answerable=False, retrieval_hit=None,
                      rerank_hit=None, rerank_top1=None, sufficient=True)

    assert refused.passed(tier=2)
    assert not answered.passed(tier=2)


# ---------------------------------------------------------
# THE TWO FAILURE MODES, KEPT SEPARATE
# ---------------------------------------------------------

def test_false_refusal_is_detected():
    """Corpus answers it, system declined - what shipped this session."""

    assert result(sufficient=False).false_refusal
    assert not result(sufficient=False).false_answer


def test_false_answer_is_detected():
    """Corpus cannot answer, system answered anyway - the worst case."""

    answered = result(answerable=False, sufficient=True)

    assert answered.false_answer
    assert not answered.false_refusal


def test_wrong_authority_is_detected():
    """
    The blind spot the Criminal Procedure Code exposed. "Can police
    arrest me without telling me why?" was answered from s.29 and s.2
    while Article 49 was never cited — and because the question is
    answerable, `false_answer` could not see it.
    """

    answered_from_elsewhere = result(
        rerank_hit=True,
        sufficient=True,
        cited_expected=False,
    )

    assert answered_from_elsewhere.wrong_authority
    assert not answered_from_elsewhere.false_answer
    assert not answered_from_elsewhere.false_refusal


def test_wrong_authority_ignores_refusals_and_out_of_scope():
    assert not result(sufficient=False, cited_expected=False).wrong_authority

    assert not result(
        answerable=False,
        sufficient=True,
        cited_expected=False,
    ).wrong_authority


def test_wrong_authority_is_counted_but_does_not_fail_the_question():
    """
    Counted, not failed. `expect_any_of` lists the authorities we know
    of, not every authority that could be legitimate, so this flags an
    answer for reading rather than declaring it wrong.
    """

    answered_from_elsewhere = result(
        rerank_hit=True,
        sufficient=True,
        cited_expected=False,
    )

    summary = summarize([answered_from_elsewhere], tier=2)

    assert summary["counts"]["wrong_authority"] == 1
    assert answered_from_elsewhere.passed(tier=2)


def test_the_two_failure_modes_are_counted_separately():
    summary = summarize(
        [
            result(id="a", sufficient=False),
            result(id="b", answerable=False, retrieval_hit=None,
                   rerank_hit=None, rerank_top1=None, sufficient=True),
        ],
        tier=2,
    )

    assert summary["counts"]["false_refusal"] == 1
    assert summary["counts"]["false_answer"] == 1


# ---------------------------------------------------------
# SCORING
# ---------------------------------------------------------

def test_metrics_only_count_answerable_questions():
    summary = summarize(
        [
            result(id="a"),
            result(id="b", rerank_top1=False),
            result(id="c", answerable=False, retrieval_hit=None,
                   rerank_hit=None, rerank_top1=None),
        ],
        tier=1,
    )

    assert summary["metrics"]["rerank_hit"] == {"passed": 2, "total": 2}
    assert summary["metrics"]["rerank_top1"] == {"passed": 1, "total": 2}
    assert summary["counts"]["out_of_scope"] == 1


def test_retrieval_and_rerank_are_scored_separately():
    """
    Separating them is what says *which stage* broke: retrieval found
    it, reranking threw it away.
    """

    summary = summarize(
        [result(retrieval_hit=True, rerank_hit=False, rerank_top1=False)],
        tier=1,
    )

    assert summary["metrics"]["retrieval_hit"]["passed"] == 1
    assert summary["metrics"]["rerank_hit"]["passed"] == 0


# ---------------------------------------------------------
# BASELINE COMPARISON
# ---------------------------------------------------------

def test_identical_run_shows_no_regression():
    summary = summarize([result(id="a"), result(id="b")], tier=1)

    assert compare(summary, summary)["regressed"] == []


def test_regression_is_named_not_just_counted():
    before = summarize([result(id="a"), result(id="b")], tier=1)
    after = summarize(
        [result(id="a"), result(id="b", rerank_hit=False)],
        tier=1,
    )

    comparison = compare(after, before)

    assert comparison["regressed"] == ["b"]
    assert comparison["deltas"]["rerank_hit"]["delta"] == -1


def test_improvement_is_reported():
    before = summarize([result(id="a", rerank_hit=False)], tier=1)
    after = summarize([result(id="a")], tier=1)

    assert compare(after, before)["improved"] == ["a"]


def test_added_and_removed_questions_are_tracked():
    before = summarize([result(id="a"), result(id="old")], tier=1)
    after = summarize([result(id="a"), result(id="new")], tier=1)

    comparison = compare(after, before)

    assert comparison["new"] == ["new"]
    assert comparison["removed"] == ["old"]
    assert comparison["regressed"] == []


def test_changed_embedding_model_invalidates_the_comparison():
    """A baseline from a different model is not a valid reference."""

    before = summarize([result(id="a")], tier=1,
                       metadata={"embedding_model": "BAAI/bge-m3"})
    after = summarize([result(id="a")], tier=1,
                      metadata={"embedding_model": "something-else"})

    assert compare(after, before)["incomparable"]


# ---------------------------------------------------------
# QUESTION SET VALIDATION
# ---------------------------------------------------------

def test_answerable_question_needs_an_expected_chunk():
    errors = validate_questions(
        [Question(id="q", question="Why?", answerable=True)]
    )

    assert any("expected chunk id" in e for e in errors)


def test_out_of_scope_question_must_not_expect_chunks():
    errors = validate_questions(
        [
            Question(
                id="q",
                question="Why?",
                answerable=False,
                expect_any_of=["some-chunk"],
            )
        ]
    )

    assert any("must not list expected chunks" in e for e in errors)


def test_duplicate_ids_are_rejected():
    errors = validate_questions(
        [
            Question(id="q", question="A?", answerable=False),
            Question(id="q", question="B?", answerable=False),
        ]
    )

    assert any("Duplicate" in e for e in errors)


def test_unknown_chunk_ids_are_caught():
    """
    A typo in an expected id would otherwise look like a retrieval
    bug and send you hunting in the wrong place.
    """

    errors = validate_against_corpus(
        [
            Question(
                id="q",
                question="A?",
                answerable=True,
                expect_any_of=["constitution-of-kenya-2010-article-999"],
            )
        ],
        known_ids={"constitution-of-kenya-2010-chapter-four-article-49"},
    )

    assert any("not in the corpus" in e for e in errors)


# ---------------------------------------------------------
# THE REAL QUESTION SET
# ---------------------------------------------------------

def test_shipped_question_set_is_valid():
    questions = load_questions()

    assert len(questions) >= 40

    answerable = [q for q in questions if q.answerable]
    out_of_scope = [q for q in questions if not q.answerable]

    assert answerable and out_of_scope


def test_shipped_questions_reference_real_chunks():
    """Guards against the corpus being re-chunked under new ids."""

    from evaluation.schema import corpus_chunk_ids

    known = corpus_chunk_ids()

    if not known:
        pytest.skip("no ingested corpus on disk to validate against")

    assert validate_against_corpus(load_questions(), known) == []
