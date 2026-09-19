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


BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"

# Every run is written here so a result you have already looked at can
# be promoted to the baseline without paying for the whole run again.
LAST_RUN_PATH = Path(__file__).resolve().parent / "last_run.json"


# ---------------------------------------------------------
# SCORING
# ---------------------------------------------------------

def summarize(
    results: list[QuestionResult],
    tier: int,
    metadata: dict | None = None,
) -> dict:
    """Reduce raw results to metrics plus a per-question record."""

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

    return {
        "metadata": metadata or {},
        "metrics": metrics,
        "counts": counts,
        "questions": {
            r.id: {
                "passed": r.passed(tier),
                "answerable": r.answerable,
                "retrieval_hit": r.retrieval_hit,
                "rerank_hit": r.rerank_hit,
                "rerank_top1": r.rerank_top1,
                "sufficient": r.sufficient,
                "cited_expected": r.cited_expected,
            }
            for r in results
        },
    }


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
    counts = summary["counts"]

    print()
    print("=" * 64)
    print(f"EVALUATION RESULTS  (tier {tier})")
    print("=" * 64)
    print()

    for name, value in summary["metrics"].items():
        total = value["total"]

        if not total:
            continue

        passed = value["passed"]
        pct = 100 * passed / total

        print(f"  {name:<16} {passed:>3}/{total:<3}  {pct:5.1f}%")

    if tier >= 2:
        print()
        print(f"  false refusals   {counts.get('false_refusal', 0)}")
        print(f"  false answers    {counts.get('false_answer', 0)}")
        print(f"  wrong authority  {counts.get('wrong_authority', 0)}")

    print()
    print(
        f"  {counts['passed']}/{counts['questions']} questions passing "
        f"({counts['answerable']} answerable, "
        f"{counts['out_of_scope']} out of scope)"
    )

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

            print(f"       cited    : {result.cited_top}")

    print()


def _print_failure(result: QuestionResult, question: Question | None) -> None:
    print()
    print(f"[FAIL] {result.id}")

    if question:
        print(f"       {question.question}")

    if not result.answerable:
        print("       cause    : expected a refusal, but the system answered")
        print(f"       cited    : {result.reranked_top[:3]}")
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

    if result.sufficient is False:
        print("       cause    : system refused a question the corpus answers")

    print(f"       expected : {sorted(expected)}")
    print(f"       reranked : {result.reranked_top or '(none)'}")

    if result.retrieved_top:
        print(f"       retrieved: {result.retrieved_top[:5]}")

    if result.cited_top:
        print(f"       cited    : {result.cited_top}")


def _first_rank(retrieved: list[str], expected: set[str]) -> str:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return str(index)

    return "?"


# ---------------------------------------------------------
# BASELINE
# ---------------------------------------------------------

def load_baseline(path: Path = BASELINE_PATH) -> dict | None:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(summary: dict, path: Path = BASELINE_PATH) -> None:
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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

    deltas = {}

    for name, value in current["metrics"].items():
        before = baseline.get("metrics", {}).get(name)

        if not before or not value["total"]:
            continue

        deltas[name] = {
            "before": before["passed"],
            "after": value["passed"],
            "total": value["total"],
            "delta": value["passed"] - before["passed"],
        }

    old_questions = baseline.get("questions", {})

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

    return {
        "incomparable": incomparable,
        "deltas": deltas,
        "regressed": sorted(regressed),
        "improved": sorted(improved),
        "new": sorted(set(current["questions"]) - set(old_questions)),
        "removed": sorted(set(old_questions) - set(current["questions"])),
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

    return reasons


def print_comparison(comparison: dict) -> None:
    print("=" * 64)
    print("COMPARED TO BASELINE")
    print("=" * 64)

    if comparison["incomparable"]:
        print()
        print("[WARN] baseline was recorded under a different setup;")
        print("       these numbers are not directly comparable:")

        for reason in comparison["incomparable"]:
            print(f"       - {reason}")

    print()

    for name, delta in comparison["deltas"].items():
        arrow = ""

        if delta["delta"] > 0:
            arrow = f"  (+{delta['delta']})"
        elif delta["delta"] < 0:
            arrow = f"  ({delta['delta']})"

        print(
            f"  {name:<16} {delta['before']:>3} -> "
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
