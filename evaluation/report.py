"""
Scoring, console output, and baseline comparison.

The comparison is the point of the harness. A single run tells you a
number; a run compared against a recorded baseline tells you what
changed, which is the only thing that makes corpus expansion or a
retrieval tweak safe to do.
"""

import json
from datetime import datetime
from pathlib import Path

from evaluation.schema import Question, QuestionResult


# One baseline per suite PER TIER: `statutory-tier1.json`, `product-tier2.json`.
#
# There used to be a single `baseline.json`, and any complete-looking run
# could be promoted over it. A tier-0 run, or one scoped to a single suite,
# would then silently replace the tier-1 record with a narrower one. Keying
# the file on suite and tier makes that impossible by construction: a run can
# only ever replace the baseline it is actually comparable to.
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

# Every run is written here so a result you have already looked at can
# be promoted to the baseline without paying for the whole run again.
LAST_RUN_PATH = Path(__file__).resolve().parent / "last_run.json"


# ---------------------------------------------------------
# SCORING
# ---------------------------------------------------------

SUITE_ORDER = ("statutory", "product")


def summarize(
    results: list[QuestionResult],
    tier: int,
    metadata: dict | None = None,
) -> dict:
    """
    Reduce raw results to metrics plus a per-question record.

    Measured PER SUITE, and the suites are what gets reported. The two
    measure different things: the statutory suite is worded like the
    statute and tests retrieval regressions; the product suite is worded
    like a person and tests whether people are served. Combined into one
    figure they hid a 62% product number behind a 98% retrieval one - so
    the top-level `metrics`/`counts` below are kept only so older
    baselines still load, and are never printed as a headline.
    """

    metrics, counts = _measure(results, tier)

    suites = {
        name: dict(zip(("metrics", "counts"), _measure(members, tier)))
        for name in SUITE_ORDER
        if (members := [r for r in results if r.suite == name])
    }

    return {
        "metadata": metadata or {},
        "metrics": metrics,
        "counts": counts,
        "suites": suites,
        "questions": {
            r.id: {
                "suite": r.suite,
                "passed": r.passed(tier),
                "answerable": r.answerable,
                "retrieval_hit": r.retrieval_hit,
                "rerank_hit": r.rerank_hit,
                "rerank_top1": r.rerank_top1,
                "sufficient": r.sufficient,
                "refused_by_grounding": r.refused_by_grounding,
                "cited_expected": r.cited_expected,
                "answer_quality": r.answer_quality,
            }
            for r in results
        },
    }


def _measure(
    results: list[QuestionResult],
    tier: int,
) -> tuple[dict, dict]:
    """Metrics and counts for one group of results."""

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    metrics = {
        "retrieval_hit": _ratio(answerable, "retrieval_hit"),
        "rerank_hit": _ratio(answerable, "rerank_hit"),
        "rerank_top1": _ratio(answerable, "rerank_top1"),
    }

    counts = {
        "questions": len(results),
        "answerable": len(answerable),
        "out_of_scope": len(unanswerable),
        "passed": sum(1 for r in results if r.passed(tier)),
    }

    if tier >= 2:
        scored = [r for r in results if r.sufficient is not None]

        metrics["refusal_correct"] = {
            "passed": sum(1 for r in scored if r.refusal_correct),
            "total": len(scored),
        }

        # Kept as separate counts, never averaged into one accuracy
        # number. Declining an answerable question is bad; answering
        # one the corpus cannot support is much worse.
        counts["false_refusal"] = sum(1 for r in results if r.false_refusal)
        counts["false_answer"] = sum(1 for r in results if r.false_answer)

        # Answered, was shown the right law, and cited something else.
        # Invisible until now: `false_answer` only looks at questions
        # the corpus cannot answer, so an in-scope question answered
        # from the wrong sections scored as a clean pass.
        counts["wrong_authority"] = sum(
            1 for r in results if r.wrong_authority
        )

        # Assertions the grounding check dropped. Not a pass/fail metric -
        # it is the model straining against its sources, and it is the
        # leading indicator for both of the counts above.
        counts["ungrounded_claims"] = sum(r.ungrounded for r in results)

        counts["answers_with_ungrounded"] = sum(
            1 for r in results if r.ungrounded
        )

        # The grounding check's refusals, split by what they did. On an
        # answerable question it is a false refusal the check CAUSED; on an
        # out-of-scope one it is a false answer the check STOPPED. The rule
        # is only worth what the second number buys against the first, so
        # they are never reported as one.
        counts["grounding_refused_answerable"] = sum(
            1 for r in results if r.refused_by_grounding and r.answerable
        )

        counts["grounding_refused_out_of_scope"] = sum(
            1 for r in results if r.refused_by_grounding and not r.answerable
        )

        # The only metric in this harness that reads the answer. Kept
        # OUT of `metrics` and out of the headline pass count on purpose:
        # retrieval scores and answer quality measure different things,
        # and averaging them into one number is how the project spent
        # weeks improving a figure that did not track the product.
        judged = [r for r in results if r.answer_quality is not None]

        counts["answer_quality"] = {
            "passed": sum(1 for r in judged if r.answer_quality),
            "total": len(judged),
        }

        defects: dict[str, int] = {}

        for r in judged:
            if not r.answer_quality and r.judge_defect:
                defects[r.judge_defect] = defects.get(r.judge_defect, 0) + 1

        counts["answer_defects"] = defects

        counts["not_judged"] = sum(
            1
            for r in results
            if r.sufficient is True and r.answer_quality is None
        )

    return metrics, counts


def _ratio(results: list[QuestionResult], attribute: str) -> dict:
    scored = [r for r in results if getattr(r, attribute) is not None]

    return {
        "passed": sum(1 for r in scored if getattr(r, attribute)),
        "total": len(scored),
    }


def build_metadata(tier: int, extra: dict) -> dict:
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "tier": tier,
        **extra,
    }


# ---------------------------------------------------------
# CONSOLE OUTPUT
# ---------------------------------------------------------

def print_run(
    summary: dict,
    results: list[QuestionResult],
    questions: dict[str, Question],
    tier: int,
) -> None:
    print()
    print("=" * 64)
    print(f"EVALUATION RESULTS  (tier {tier})")
    print("=" * 64)

    judged = summary.get("metadata", {}).get("judged", True)

    # One block per suite, deliberately with no combined total beneath
    # them. See `summarize`.
    for name, suite in (summary.get("suites") or {}).items():
        _print_suite(name, suite["metrics"], suite["counts"], tier, judged)

    failures = [r for r in results if not r.passed(tier)]

    if failures:
        print()
        print("-" * 64)
        print("FAILURES")
        print("-" * 64)

        for result in failures:
            _print_failure(result, questions.get(result.id))

    # Not failures — the expected authority did reach the LLM. But the
    # answer rests on something else, which is worth a human reading
    # either the answer or the question's expected set.
    flagged = [
        r for r in results if r.wrong_authority and r.passed(tier)
    ]

    if flagged:
        print()
        print("-" * 64)
        print("ANSWERED FROM SOURCES WE DID NOT EXPECT")
        print("-" * 64)

        for result in flagged:
            question = questions.get(result.id)

            print()
            print(f"[READ] {result.id}")

            if question:
                print(f"       {question.question}")
                print(f"       expected : {sorted(question.expect_any_of)}")

            # The whole point of this section is that someone reads the
            # answer - so print it, not just the ids it cited.
            _print_answer(result)
            print(f"       cited    : {result.cited_top}")
            _print_dropped(result)

    # A question can PASS - right law reached the LLM, and it answered -
    # while the judge says the answer is a non-sequitur. Until this was
    # printed, the judge's count was all anyone saw: a run reported "1
    # non_sequitur" among three answers and nothing said which one.
    rejected = [r for r in results if r.answer_quality is False]

    if rejected:
        print()
        print("-" * 64)
        print("ANSWERS THE JUDGE REJECTED")
        print("-" * 64)

        for result in rejected:
            question = questions.get(result.id)

            print()
            print(f"[READ] {result.id}")

            if question:
                print(f"       {question.question}")

            print(f"       defect   : {result.judge_defect}")
            print(f"       judge    : {result.judge_reason}")
            _print_answer(result)
            print(f"       cited    : {result.cited_top}")

            # Which claim failed, and the section's own words beside it -
            # enough to check the judge without opening the corpus.
            for number, check in enumerate(result.judge_claims, start=1):
                if check.get("finding") == "stated":
                    continue

                print(f"       claim {number}  : [{check.get('finding')}] "
                      f"{check.get('claim')}")
                print(f"                  law says  \"{check.get('section_says')}\"")
                print(f"                  judge     {check.get('reason')}")

            _print_dropped(result)

    # Served answers that lost claims on the way. Failures already print
    # their drops; these passed, so nothing else would show them.
    trimmed = [
        r
        for r in results
        if r.dropped and r.passed(tier) and r.answer_quality is not False
    ]

    if trimmed:
        print()
        print("-" * 64)
        print("ANSWERS GIVEN WITH CLAIMS DROPPED")
        print("-" * 64)

        for result in trimmed:
            question = questions.get(result.id)

            print()
            print(f"[READ] {result.id}")

            if question:
                print(f"       {question.question}")

            _print_evidence(result)

    print()


SUITE_TITLES = {
    "statutory": "STATUTORY SUITE - worded like the statute (retrieval regression)",
    "product": "PRODUCT SUITE - worded like a person (is anyone served?)",
}


def _print_suite(
    name: str,
    metrics: dict,
    counts: dict,
    tier: int,
    judged: bool,
) -> None:
    print()
    print(f"  {SUITE_TITLES.get(name, name.upper())}")
    print()

    for metric, value in metrics.items():
        total = value["total"]

        if not total:
            continue

        passed = value["passed"]

        print(
            f"    {metric:<16} {passed:>3}/{total:<3}  "
            f"{100 * passed / total:5.1f}%"
        )

    if tier >= 2:
        false_refusals = counts.get("false_refusal", 0)
        by_grounding = counts.get("grounding_refused_answerable", 0)

        print()
        print(
            f"    false refusals   {false_refusals}  "
            f"({false_refusals - by_grounding} by the model, "
            f"{by_grounding} by the grounding check)"
        )
        print(f"    false answers    {counts.get('false_answer', 0)}")
        print(
            f"    stopped answers  "
            f"{counts.get('grounding_refused_out_of_scope', 0)}  "
            f"(out of scope, answered by the model, refused by the "
            f"grounding check)"
        )
        print(f"    wrong authority  {counts.get('wrong_authority', 0)}")
        print(
            f"    ungrounded       {counts.get('ungrounded_claims', 0)} "
            f"claim(s) dropped across "
            f"{counts.get('answers_with_ungrounded', 0)} answer(s)"
        )

        _print_answer_quality(counts, judged=judged)

    print()
    print(
        f"    {counts['passed']}/{counts['questions']} passing "
        f"({counts['answerable']} answerable, "
        f"{counts['out_of_scope']} out of scope)"
    )


def _print_answer_quality(counts: dict, judged: bool = True) -> None:
    """
    Reported apart from the retrieval metrics, and deliberately so.

    Every other number here answers "did the right passage reach the
    LLM". This one answers "was the answer any good". They moved in
    opposite directions once already - 121/133 on retrieval while a
    question about leaving property to one's children was answered with
    a point about adopted children - and a single combined figure would
    have hidden it.
    """

    quality = counts.get("answer_quality") or {}
    total = quality.get("total", 0)

    print()
    print("  ANSWER QUALITY (judged, tier 2 only)")

    if not total:
        # Two very different reasons, and saying the wrong one sends
        # someone hunting a flag that was never set.
        if not judged:
            print("    not measured - --no-judge was set")
        else:
            print(
                "    nothing to judge - every answered question was a "
                "refusal, and refusals are scored by refusal_correct"
            )
        return

    passed = quality.get("passed", 0)

    print(
        f"    responsive + supported + no defect   {passed:>3}/{total:<3}"
        f"  {100 * passed / total:5.1f}%"
    )

    defects = counts.get("answer_defects") or {}

    for name, number in sorted(defects.items(), key=lambda item: -item[1]):
        print(f"      {name:<16} {number}")

    if counts.get("not_judged"):
        print(
            f"    {counts['not_judged']} answered question(s) could not be "
            f"judged (judge error)"
        )


def _print_failure(result: QuestionResult, question: Question | None) -> None:
    print()
    print(f"[FAIL] {result.id}")

    if question:
        print(f"       {question.question}")

    if not result.answerable:
        # This line used to be labelled "cited" while printing the top three
        # RERANKED sources - what the LLM was shown, not what it relied on.
        print("       cause    : expected a refusal, but the system answered")
        _print_answer(result)
        print(f"       cited    : {result.cited_top or '(none)'}")
        print(f"       reranked : {result.reranked_top or '(none)'}")
        _print_dropped(result)
        return

    expected = set(question.expect_any_of) if question else set()

    # Naming the stage is the point of scoring retrieval and reranking
    # separately. "Retrieved then dropped" and "never retrieved" need
    # completely different fixes.
    if result.retrieval_hit is None:
        # Tier 2 cannot see the raw retrieved set, so it must not claim
        # retrieval failed. Run tier 1 to find out which stage broke.
        if not result.rerank_hit:
            print(
                "       cause    : NOT AMONG THE SOURCES SENT TO THE LLM "
                "(run tier 1 to see whether retrieval or reranking lost it)"
            )

    elif not result.retrieval_hit:
        print("       cause    : NEVER RETRIEVED - vector search missed it")

    elif not result.rerank_hit:
        rank = _first_rank(result.retrieved_top, expected)
        print(
            f"       cause    : RETRIEVED THEN DROPPED - reranking "
            f"discarded it (was #{rank} of {len(result.retrieved_top)})"
        )

    if result.refused_by_grounding:
        print(
            "       cause    : REFUSED BY THE GROUNDING CHECK - the model "
            "answered, but too few statements survived verification "
            f"({len(result.statements_kept)} kept, "
            f"{result.statements_dropped} dropped)"
        )

    elif result.sufficient is False:
        print("       cause    : REFUSED BY THE MODEL - it declined itself")

    # The refusal as the user read it, including the model's own account of
    # what the sources lack. Without it, a refusal with the right law among
    # the sources could be counted but not judged: the first Luna run refused
    # will-on-phone with s.11, s.8 and s.9 ranked first to third, and the
    # report could not say whether that was caution or a correct reading of
    # an Act that never mentions phones.
    if result.sufficient is False:
        _print_answer(result)

    print(f"       expected : {sorted(expected)}")

    if result.rewrites:
        print(f"       rewrites : {result.rewrites}")

    print(f"       reranked : {result.reranked_top or '(none)'}")

    if result.retrieved_top:
        print(f"       retrieved: {result.retrieved_top[:5]}")

    if result.cited_top:
        print(f"       cited    : {result.cited_top}")

    # Printed for EVERY failure with drops, not only grounding refusals: a
    # model refusal after a dropped claim is worth reading too.
    _print_evidence(result)


def _print_answer(result: QuestionResult) -> None:
    """The rendered answer, as the user would have read it."""

    lines = result.answer_text.splitlines()

    if not lines:
        return

    print(f"       answer   : {lines[0]}")

    for line in lines[1:]:
        print(f"                  {line}")


def _print_evidence(result: QuestionResult) -> None:
    """What the answer kept, and each claim verification dropped and why."""

    for kept in result.statements_kept:
        print(f"       kept     : {kept}")

    _print_dropped(result)


def _print_dropped(result: QuestionResult) -> None:
    for item in result.dropped:
        print(f"       dropped  : [{item['failure']}] {item['text']}")
        print(f"                  cited  {item['chunk_id']}")
        print(f"                  quote  \"{item['quote']}\"")


def _first_rank(retrieved: list[str], expected: set[str]) -> str:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return str(index)

    return "?"


# ---------------------------------------------------------
# BASELINE
# ---------------------------------------------------------

def baseline_path(
    suite: str,
    tier: int,
    directory: Path = BASELINE_DIR,
) -> Path:
    return directory / f"{suite}-tier{tier}.json"


def suite_summary(summary: dict, suite: str) -> dict:
    """
    One suite's share of a run, in the same shape as a whole run, so that
    `compare` reads it unchanged.
    """

    block = summary["suites"][suite]

    return {
        "metadata": summary.get("metadata", {}),
        "metrics": block["metrics"],
        "counts": block["counts"],
        "suites": {suite: block},
        "questions": {
            qid: record
            for qid, record in summary.get("questions", {}).items()
            if record.get("suite", "statutory") == suite
        },
    }


def promotion_refusal(summary: dict) -> str | None:
    """
    Why this run may not become a baseline - or None if it may.

    One rule for both `--save-baseline` and `--accept-last`, so they cannot
    drift apart. Every condition here is a way a run can LOOK complete while
    measuring less than the baseline it would replace.
    """

    metadata = summary.get("metadata") or {}
    scope = metadata.get("scope")

    # Checked first: without a recorded scope none of the checks below can be
    # made, and guessing that a run was complete is exactly the old gap.
    if scope is None or metadata.get("tier") is None:
        return (
            "the run does not record its scope - it predates scope "
            "tracking. Re-run it."
        )

    if scope.get("ids"):
        return "it ran only selected questions (--id)."

    if scope.get("document"):
        return (
            f"it ran only one document's questions "
            f"(--document {scope['document']}), and out-of-scope "
            f"questions are excluded from those."
        )

    if metadata.get("reranked") is False:
        return "reranking was disabled - --no-rerank is a measurement, not the pipeline."

    # A run rewrote while rewriting is not the pipeline, or did not while it
    # is. Either way the run measured something other than what a baseline
    # records - and the "before" baselines Phase B is measured against are
    # exactly what an accidental --accept-last after a --rewrite run would
    # have overwritten.
    rewrote = bool(metadata.get("rewrite"))

    if rewrote != bool(metadata.get("pipeline_rewrite")):
        return (
            f"query rewriting was {'on' if rewrote else 'off'} in this run "
            f"but QUERY_REWRITE is {'off' if rewrote else 'on'} - a "
            "measurement of rewriting, not the pipeline."
        )

    # QUERY_REWRITE means both phases: the pipeline's reranker reads the
    # rewrites. A run that rewrote for retrieval only, at a tier with a
    # reranker, measured something the pipeline does not do. Tier 0 has no
    # reranker, so for it rewriting retrieval IS the whole pipeline.
    if (
        rewrote
        and metadata.get("tier", 0) >= 1
        and not metadata.get("rerank_rewrites")
    ):
        return (
            "the reranker did not read the rewrites, but with QUERY_REWRITE "
            "on the pipeline's does - a measurement of retrieval-only "
            "rewriting, not the pipeline."
        )

    # A reference result must be computed, not recalled.
    if any(((metadata.get("cache") or {}).get("hits") or {}).values()):
        return "it used the cache. Re-run with --no-cache, then promote that."

    return None


def load_baseline(
    suite: str,
    tier: int,
    directory: Path = BASELINE_DIR,
) -> dict | None:
    path = baseline_path(suite, tier, directory)

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(
    summary: dict,
    directory: Path = BASELINE_DIR,
) -> dict[str, Path]:
    """
    Write one baseline per suite the run contains, at the run's own tier.
    Suites the run did not contain keep whatever baseline they had.

    Callers must check `promotion_refusal` first; this only writes.
    """

    tier = summary["metadata"]["tier"]

    directory.mkdir(parents=True, exist_ok=True)

    written = {}

    for suite in summary.get("suites") or {}:
        path = baseline_path(suite, tier, directory)

        path.write_text(
            json.dumps(
                suite_summary(summary, suite),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        written[suite] = path

    return written


def save_last_run(summary: dict, path: Path = LAST_RUN_PATH) -> None:
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_last_run(path: Path = LAST_RUN_PATH) -> dict | None:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def compare(current: dict, baseline: dict) -> dict:
    """
    Diff a run against a baseline.

    Returns the aggregate movements, the questions whose status
    changed, and whether anything regressed.
    """

    incomparable = _incomparable_reasons(current, baseline)

    # Deltas are computed per suite and never across them. A baseline
    # recorded before suites existed holds only the statutory questions at
    # its top level, so it is read as that suite rather than rejected.
    before_suites = baseline.get("suites") or {
        "statutory": {"metrics": baseline.get("metrics", {})}
    }

    deltas: dict[str, dict] = {}

    for suite, block in (current.get("suites") or {}).items():
        before_metrics = (before_suites.get(suite) or {}).get("metrics", {})

        for name, value in block["metrics"].items():
            before = before_metrics.get(name)

            if not before or not value["total"]:
                continue

            deltas.setdefault(suite, {})[name] = {
                "before": before["passed"],
                "after": value["passed"],
                "total": value["total"],
                "delta": value["passed"] - before["passed"],
            }

    old_questions = baseline.get("questions", {})

    # Only questions from suites that actually ran can have been removed.
    # Without this, `--suite product` would report every statutory
    # question as removed - the --id comparison bug again, one level up.
    ran = {r.get("suite", "statutory") for r in current["questions"].values()}

    regressed = []
    improved = []

    for qid, record in current["questions"].items():
        previous = old_questions.get(qid)

        if previous is None:
            continue

        if previous["passed"] and not record["passed"]:
            regressed.append(qid)

        elif not previous["passed"] and record["passed"]:
            improved.append(qid)

    removed = {
        qid
        for qid, record in old_questions.items()
        if record.get("suite", "statutory") in ran
    } - set(current["questions"])

    return {
        "incomparable": incomparable,
        "deltas": deltas,
        "regressed": sorted(regressed),
        "improved": sorted(improved),
        "new": sorted(set(current["questions"]) - set(old_questions)),
        "removed": sorted(removed),
    }


def _incomparable_reasons(current: dict, baseline: dict) -> list[str]:
    """
    A baseline taken under a different configuration is not a valid
    reference. Say so rather than report a meaningless delta.
    """

    reasons = []

    for key in ("embedding_model", "reranker_model", "collection"):
        before = baseline.get("metadata", {}).get(key)
        after = current.get("metadata", {}).get(key)

        if before and after and before != after:
            reasons.append(f"{key} changed: {before} -> {after}")

    # `passed()` means something different at each tier - tier 0 asks whether
    # the authority was retrieved, tier 1 whether it survived reranking, tier
    # 2 whether the answer was also given. Comparing across them reports
    # regressions that are only a change of question, which had been
    # happening quietly on every tier 2 run against the tier 1 baseline.
    before_tier = baseline.get("metadata", {}).get("tier")
    after_tier = current.get("metadata", {}).get("tier")

    if before_tier is not None and after_tier is not None:
        if before_tier != after_tier:
            reasons.append(
                f"tier changed: {before_tier} -> {after_tier} "
                "(pass criteria differ, so the comparison is not meaningful)"
            )

    # Flagged rather than hidden: with --no-rerank the delta IS the
    # measurement, so the numbers still print - this only says what changed
    # so nobody reads it as a regression.
    if current.get("metadata", {}).get("reranked") is False:
        reasons.append(
            "reranking disabled in this run - the deltas below measure what "
            "the cross-encoder contributes, they are not regressions"
        )

    # The same idea for rewriting: the delta IS the measurement.
    before_rewrite = bool(baseline.get("metadata", {}).get("rewrite"))
    after_rewrite = bool(current.get("metadata", {}).get("rewrite"))

    if before_rewrite != after_rewrite:
        reasons.append(
            f"query rewriting {'on' if after_rewrite else 'off'} in this "
            f"run, {'on' if before_rewrite else 'off'} in the baseline - the "
            "deltas below measure what rewriting changes"
        )

    # Phase C: whether the reranker read the rewrites as well.
    before_c = bool(baseline.get("metadata", {}).get("rerank_rewrites"))
    after_c = bool(current.get("metadata", {}).get("rerank_rewrites"))

    if before_c != after_c:
        reasons.append(
            f"the reranker {'read' if after_c else 'did not read'} the "
            f"rewritten terms in this run and "
            f"{'did' if before_c else 'did not'} in the baseline - the "
            "deltas below measure what that changes"
        )

    return reasons


def print_comparison(comparison: dict, label: str = "") -> None:
    print("=" * 64)
    print(f"COMPARED TO BASELINE{f'  ({label})' if label else ''}")
    print("=" * 64)

    if comparison["incomparable"]:
        print()
        print("[WARN] baseline was recorded under a different setup;")
        print("       these numbers are not directly comparable:")

        for reason in comparison["incomparable"]:
            print(f"       - {reason}")

    for suite, metrics in comparison["deltas"].items():
        print()
        print(f"  {suite}")

        for name, delta in metrics.items():
            arrow = ""

            if delta["delta"] > 0:
                arrow = f"  (+{delta['delta']})"
            elif delta["delta"] < 0:
                arrow = f"  ({delta['delta']})"

            print(
                f"    {name:<16} {delta['before']:>3} -> "
                f"{delta['after']:>3} / {delta['total']:<3}{arrow}"
            )

    for label, key in (
        ("REGRESSED", "regressed"),
        ("improved", "improved"),
        ("new", "new"),
        ("removed from set", "removed"),
    ):
        if comparison[key]:
            print()
            print(f"  {label}: {', '.join(comparison[key])}")

    print()

    if comparison["regressed"]:
        print(
            f"[FAIL] {len(comparison['regressed'])} question(s) "
            "regressed since the baseline"
        )
    else:
        print("[PASS] no regressions")

    print()
