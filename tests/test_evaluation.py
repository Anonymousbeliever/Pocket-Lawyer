"""
The evaluation harness itself.

A harness nobody has watched fail is not known to work, so these tests
mostly assert that it *detects* things: regressions, false refusals,
false answers, and question sets that reference chunks which do not
exist.
"""

import pytest

from evaluation.report import (
    compare,
    load_baseline,
    print_run,
    promotion_refusal,
    save_baseline,
    suite_summary,
    summarize,
)
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

def test_tier0_judges_retrieval_only():
    """
    No reranker runs, so `rerank_hit` is unknown and must not be consulted.
    The Criminal Procedure Code's worst regression - Article 49 falling
    outside the top 30 for `arrest-bail` - is visible at this tier.
    """

    found = result(retrieval_hit=True, rerank_hit=None, rerank_top1=None)
    missed = result(retrieval_hit=False, rerank_hit=None, rerank_top1=None)

    assert found.passed(tier=0)
    assert not missed.passed(tier=0)


def test_tier0_does_not_judge_out_of_scope_questions():
    out_of_scope = result(
        answerable=False,
        retrieval_hit=None,
        rerank_hit=None,
        rerank_top1=None,
    )

    assert out_of_scope.passed(tier=0)


def test_tier0_and_tier1_disagree_on_purpose():
    """
    Retrieved but reranked away: tier 0 passes it, tier 1 fails it. That is
    the difference between the two, and why their baselines are not
    interchangeable.
    """

    dropped = result(retrieval_hit=True, rerank_hit=False, rerank_top1=False)

    assert dropped.passed(tier=0)
    assert not dropped.passed(tier=1)


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
    assert comparison["deltas"]["statutory"]["rerank_hit"]["delta"] == -1


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


def test_comparing_across_tiers_is_refused():
    """
    `passed()` asks a different question at each tier, so a cross-tier diff
    reports regressions that are only a change of criteria. This had been
    happening silently on every tier 2 run against the tier 1 baseline.
    """

    tier1 = summarize([result(id="a")], tier=1, metadata={"tier": 1})
    tier2 = summarize([result(id="a", sufficient=True)], tier=2,
                      metadata={"tier": 2})

    reasons = compare(tier2, tier1)["incomparable"]

    assert any("tier changed" in reason for reason in reasons)


def test_a_no_rerank_run_is_flagged_not_hidden():
    """
    With --no-rerank the delta IS the measurement, so the numbers must still
    print - this only stops them being read as regressions.
    """

    baseline = summarize([result(id="a")], tier=1,
                         metadata={"tier": 1, "reranked": True})
    measured = summarize([result(id="a", rerank_hit=False)], tier=1,
                         metadata={"tier": 1, "reranked": False})

    comparison = compare(measured, baseline)

    assert any("reranking disabled" in r for r in comparison["incomparable"])
    assert comparison["deltas"]["statutory"]["rerank_hit"]["delta"] == -1


def test_a_reranked_run_is_not_flagged():
    baseline = summarize([result(id="a")], tier=1,
                         metadata={"tier": 1, "reranked": True})
    current = summarize([result(id="a")], tier=1,
                        metadata={"tier": 1, "reranked": True})

    assert compare(current, baseline)["incomparable"] == []


def test_same_tier_comparison_is_allowed():
    before = summarize([result(id="a")], tier=1, metadata={"tier": 1})
    after = summarize([result(id="a")], tier=1, metadata={"tier": 1})

    assert compare(after, before)["incomparable"] == []


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


# ---------------------------------------------------------
# SUITES
#
# Two question sets that measure different things, reported separately.
# The statutory suite is worded like the statute; the product suite like a
# person. Combined into one figure they hid a 62% product number behind a
# 98% retrieval one, which is the failure these tests exist to prevent.
# ---------------------------------------------------------

def test_both_suites_load_and_are_tagged_by_file():
    questions = load_questions()

    suites = {q.suite for q in questions}

    assert suites == {"statutory", "product"}

    product = [q for q in questions if q.suite == "product"]

    assert product and all(q.id.startswith("product-") for q in product)


def test_a_single_suite_can_be_loaded_alone():
    product = load_questions(suites=("product",))

    assert product and {q.suite for q in product} == {"product"}


def test_advice_questions_are_out_of_scope_and_marked():
    advice = [q for q in load_questions() if q.kind == "advice"]

    assert advice, "the product suite should mark its advice questions"
    assert not any(q.answerable for q in advice)


def test_suites_are_measured_separately():
    summary = summarize(
        [
            result(id="s1", suite="statutory"),
            result(id="p1", suite="product", rerank_hit=False),
        ],
        tier=1,
    )

    assert summary["suites"]["statutory"]["metrics"]["rerank_hit"] == {
        "passed": 1, "total": 1,
    }
    assert summary["suites"]["product"]["metrics"]["rerank_hit"] == {
        "passed": 0, "total": 1,
    }


def test_the_report_never_prints_a_combined_headline(capsys):
    summary = summarize(
        [
            result(id="s1", suite="statutory"),
            result(id="p1", suite="product"),
        ],
        tier=1,
    )

    print_run(summary, [], {}, tier=1)

    out = capsys.readouterr().out

    assert "STATUTORY SUITE" in out
    assert "PRODUCT SUITE" in out
    assert "2/2" not in out, "a combined total leaked into the report"


def test_running_one_suite_does_not_report_the_other_as_removed():
    """
    `--suite product` must not list every statutory question as removed -
    the --id comparison bug, one level up.
    """

    baseline = summarize(
        [result(id="s1", suite="statutory"), result(id="p1", suite="product")],
        tier=1,
    )
    product_only = summarize([result(id="p1", suite="product")], tier=1)

    assert compare(product_only, baseline)["removed"] == []


def test_a_baseline_from_before_suites_is_read_as_statutory():
    """Older baselines hold only statutory questions, at the top level."""

    old = summarize([result(id="s1", rerank_hit=True)], tier=1)
    del old["suites"]

    current = summarize([result(id="s1", rerank_hit=False)], tier=1)

    delta = compare(current, old)["deltas"]["statutory"]["rerank_hit"]

    assert delta["delta"] == -1


# ---------------------------------------------------------
# WHO REFUSED
# ---------------------------------------------------------

def test_grounding_refusals_are_split_by_what_they_did():
    """
    On an answerable question the check caused a false refusal; on an
    out-of-scope one it stopped a false answer. The rule is worth the
    second against the first, so they are counted apart.
    """

    summary = summarize(
        [
            result(id="caused", sufficient=False, refused_by_grounding=True),
            result(id="model", sufficient=False),
            result(
                id="stopped",
                answerable=False,
                rerank_hit=None,
                rerank_top1=None,
                sufficient=False,
                refused_by_grounding=True,
            ),
        ],
        tier=2,
    )

    counts = summary["suites"]["statutory"]["counts"]

    assert counts["false_refusal"] == 2
    assert counts["grounding_refused_answerable"] == 1
    assert counts["grounding_refused_out_of_scope"] == 1


def test_a_grounding_refusal_prints_its_evidence(capsys):
    refused = result(
        id="q",
        sufficient=False,
        refused_by_grounding=True,
        statements_kept=["What survived."],
        statements_dropped=1,
        dropped=[
            {
                "text": "What was dropped.",
                "chunk_id": "some-chunk",
                "quote": "the quote it gave",
                "failure": "quote_not_found",
            }
        ],
    )

    print_run(summarize([refused], tier=2), [refused], {}, tier=2)

    out = capsys.readouterr().out

    assert "REFUSED BY THE GROUNDING CHECK" in out
    assert "(1 kept, 1 dropped)" in out
    assert "[quote_not_found] What was dropped." in out
    assert "the quote it gave" in out


def test_a_passing_answer_the_judge_rejected_is_named(capsys):
    """
    A run once reported "1 non_sequitur" among three answers and nothing
    said which one - the question passed, so it never reached FAILURES.
    """

    answered = result(
        id="q",
        sufficient=True,
        cited_expected=True,
        answer_quality=False,
        judge_defect="non_sequitur",
        judge_reason="Answers a different question.",
    )

    assert answered.passed(tier=2)

    print_run(summarize([answered], tier=2), [answered], {}, tier=2)

    out = capsys.readouterr().out

    assert "ANSWERS THE JUDGE REJECTED" in out
    assert "non_sequitur" in out
    assert "Answers a different question." in out


def test_a_rejected_answer_names_the_failing_claim_and_the_law(capsys):
    rejected = result(
        id="q",
        sufficient=True,
        cited_expected=True,
        answer_quality=False,
        judge_defect="wrong_actor",
        judge_claims=[
            {"claim": "Fine.", "section_says": "x", "finding": "stated",
             "reason": ""},
            {"claim": "The police must return it.",
             "section_says": "the court shall direct the thing to be restored",
             "finding": "wrong_actor",
             "reason": "The section gives the power to the court."},
        ],
    )

    print_run(summarize([rejected], tier=2), [rejected], {}, tier=2)

    out = capsys.readouterr().out

    assert "claim 2  : [wrong_actor] The police must return it." in out
    assert "the court shall direct the thing to be restored" in out
    assert "claim 1" not in out, "a stated claim should not be listed"


def test_a_served_answer_that_lost_claims_shows_them(capsys):
    trimmed = result(
        id="q",
        sufficient=True,
        cited_expected=True,
        answer_quality=True,
        statements_kept=["What survived."],
        dropped=[
            {
                "text": "What was dropped.",
                "chunk_id": "some-chunk",
                "quote": "the quote it gave",
                "failure": "quote_not_found",
            }
        ],
    )

    print_run(summarize([trimmed], tier=2), [trimmed], {}, tier=2)

    out = capsys.readouterr().out

    assert "ANSWERS GIVEN WITH CLAIMS DROPPED" in out
    assert "[quote_not_found] What was dropped." in out


def test_a_false_answer_prints_what_the_user_read_and_what_it_cited(capsys):
    """
    The "cited" line for a false answer used to print the top three
    RERANKED sources. What the LLM was shown and what it relied on are
    different things, and the label must say which is which.
    """

    answered = result(
        id="q",
        answerable=False,
        rerank_hit=None,
        rerank_top1=None,
        sufficient=True,
        answer_text="Yes.\nSection 9 says so.",
        cited_top=["relied-on"],
        reranked_top=["shown-first", "shown-second", "relied-on"],
    )

    print_run(summarize([answered], tier=2), [answered], {}, tier=2)

    out = capsys.readouterr().out

    assert "answer   : Yes." in out
    assert "Section 9 says so." in out
    assert "cited    : ['relied-on']" in out
    assert "reranked : ['shown-first', 'shown-second', 'relied-on']" in out


def test_an_answer_from_unexpected_law_prints_the_answer(capsys):
    unexpected = result(
        id="q",
        sufficient=True,
        cited_expected=False,
        answer_text="No. Section 49 applies.",
        cited_top=["section-49"],
    )

    assert unexpected.wrong_authority and unexpected.passed(tier=2)

    print_run(summarize([unexpected], tier=2), [unexpected], {}, tier=2)

    out = capsys.readouterr().out

    assert "ANSWERED FROM SOURCES WE DID NOT EXPECT" in out
    assert "answer   : No. Section 49 applies." in out


def test_a_model_refusal_is_named_as_the_models(capsys):
    refused = result(id="q", sufficient=False)

    print_run(summarize([refused], tier=2), [refused], {}, tier=2)

    out = capsys.readouterr().out

    assert "REFUSED BY THE MODEL" in out
    assert "GROUNDING CHECK -" not in out


# ---------------------------------------------------------
# BASELINES - one per suite per tier, and only complete runs
# ---------------------------------------------------------

def complete(tier: int = 1, **scope_overrides) -> dict:
    """Metadata for a run that may legitimately become a baseline."""

    scope = {"suites": ["statutory", "product"], "document": None, "ids": []}
    scope.update(scope_overrides)

    return {
        "tier": tier,
        "scope": scope,
        "reranked": True,
        "cache": {"hits": {"retrieval": 0, "rerank": 0}},
    }


def run(tier: int = 1, metadata: dict | None = None) -> dict:
    return summarize(
        [result(id="s1", suite="statutory"), result(id="p1", suite="product")],
        tier=tier,
        metadata=metadata if metadata is not None else complete(tier),
    )


def test_a_complete_uncached_run_may_be_promoted():
    assert promotion_refusal(run()) is None


@pytest.mark.parametrize(
    "metadata, reason",
    [
        ({"tier": 1}, "scope"),
        (complete(ids=["arrest-bail"]), "--id"),
        (complete(document="penal-code"), "--document"),
        ({**complete(), "reranked": False}, "--no-rerank"),
        ({**complete(), "cache": {"hits": {"rerank": 3}}}, "cache"),
    ],
)
def test_a_run_that_measured_less_is_refused(metadata, reason):
    assert reason in promotion_refusal(run(metadata=metadata))


def test_each_suite_gets_its_own_baseline(tmp_path):
    written = save_baseline(run(), directory=tmp_path)

    assert set(written) == {"statutory", "product"}

    statutory = load_baseline("statutory", 1, directory=tmp_path)

    assert set(statutory["questions"]) == {"s1"}
    assert set(statutory["suites"]) == {"statutory"}


def test_a_tier0_run_cannot_replace_the_tier1_baseline(tmp_path):
    """
    The gap this layout closes: a cold-cache tier-0 run used to be
    promotable over the single baseline, silently swapping a reranking
    record for a retrieval-only one.
    """

    save_baseline(run(tier=1), directory=tmp_path)
    before = load_baseline("statutory", 1, directory=tmp_path)

    save_baseline(run(tier=0), directory=tmp_path)

    assert load_baseline("statutory", 1, directory=tmp_path) == before
    assert load_baseline("statutory", 0, directory=tmp_path) is not None


def test_one_suite_run_leaves_the_other_baseline_alone(tmp_path):
    save_baseline(run(), directory=tmp_path)
    product_before = load_baseline("product", 1, directory=tmp_path)

    statutory_only = summarize(
        [result(id="s1", suite="statutory", rerank_hit=False)],
        tier=1,
        metadata=complete(suites=["statutory"]),
    )

    save_baseline(statutory_only, directory=tmp_path)

    assert load_baseline("product", 1, directory=tmp_path) == product_before


def test_a_suite_is_compared_only_with_its_own_baseline(tmp_path):
    save_baseline(run(), directory=tmp_path)

    current = summarize(
        [result(id="s1", suite="statutory", rerank_hit=False),
         result(id="p1", suite="product")],
        tier=1,
        metadata=complete(),
    )

    statutory = compare(
        suite_summary(current, "statutory"),
        load_baseline("statutory", 1, directory=tmp_path),
    )

    assert statutory["regressed"] == ["s1"]
    assert statutory["removed"] == []
    assert statutory["new"] == []
