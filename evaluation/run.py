"""
Pocket Lawyer evaluation runner.

    python -m evaluation.run                     tier 1, compare to baseline
    python -m evaluation.run --tier 2            full pipeline
    python -m evaluation.run --save-baseline     accept this run as reference
    python -m evaluation.run --id arrest-reason  one question, for debugging

Tier 1 runs retrieval and reranking only. It needs no OpenAI key and
costs nothing, so it can be run after any change to chunking,
embeddings, retrieval or reranking. Tier 2 adds the LLM, which is the
only way to check whether the system answered or refused.

Baselines live in evaluation/baselines/, one per suite per tier, and a run
is only ever compared with - or promoted over - its own suite and tier.

Models are loaded once for the whole run, never per question.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from backend.app.ai.answer import render
from backend.app.ai.rewrite import rerank_query
from backend.app.core.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QUERY_REWRITE,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    REWRITE_COUNT,
    REWRITE_TOP_K,
)
from evaluation.cache import (
    ResultCache,
    corpus_fingerprint,
    rerank_key,
    retrieval_key,
)
from evaluation.judge import (
    AnswerJudge,
    headline as judge_headline,
    quality as judge_quality,
)
from evaluation.report import (
    build_metadata,
    compare,
    load_baseline,
    load_last_run,
    print_comparison,
    print_run,
    promotion_refusal,
    save_baseline,
    save_last_run,
    suite_summary,
    summarize,
)
from evaluation.schema import (
    Question,
    QuestionResult,
    corpus_chunk_ids,
    load_questions,
    validate_against_corpus,
)


# The rewrites every rewrite run used, one reviewable JSON per question.
# Committed: unlike evaluation/.cache, deleting it CHANGES results, because
# the rewriter's model gives different phrases on different calls.
REWRITES_DIR = Path(__file__).resolve().parent / "rewrites"


def _retrieve_all(
    questions: list[Question],
    retriever,
    cache: ResultCache,
    fingerprint: str,
    rewriter=None,
) -> tuple[list[list[dict]], list[list[str]]]:
    """
    Retrieve for every question, reusing level 1 where the corpus has not
    moved. Cheap either way - an embedding and a Qdrant query.

    With a rewriter, each question is first restated in legal register and
    retrieval searches with the rewrites as well. Returns the retrieved
    chunks and, per question, the rewrites used.
    """

    retrieved: list[list[dict]] = []
    rewrites: list[list[str]] = []

    for index, question in enumerate(questions, start=1):
        print(
            f"  [{index:>2}/{len(questions)}] {question.id}",
            flush=True,
        )

        extra = rewriter.rewrite(question.question) if rewriter else []

        key = retrieval_key(question.question, fingerprint, extra)

        chunks = cache.get("retrieval", key)

        if chunks is None:
            chunks = retriever.retrieve(
                query=question.question,
                top_k=RETRIEVAL_TOP_K,
                extra_queries=extra or None,
            )

            cache.put("retrieval", key, chunks)

        retrieved.append(chunks)
        rewrites.append(extra)

    return retrieved, rewrites


def _score_question(
    question: Question,
    retrieved_ids: list[str],
    reranked_ids: list[str] | None,
) -> QuestionResult:
    """
    `reranked_ids=None` means no reranker ran - tier 0. The rerank metrics
    then stay None so `_ratio` reports them as unmeasured rather than as
    zero; a metric nothing measured must not be published as a failure.
    """

    result = QuestionResult(
        id=question.id,
        answerable=question.answerable,
        retrieved_top=retrieved_ids,
        reranked_top=reranked_ids or [],
    )

    # Only an answerable question has an authority to find.
    if question.answerable:
        expected = set(question.expect_any_of)

        result.retrieval_hit = bool(expected & set(retrieved_ids))

        if reranked_ids is not None:
            result.rerank_hit = bool(expected & set(reranked_ids))
            result.rerank_top1 = bool(
                reranked_ids and reranked_ids[0] in expected
            )

    return result


def evaluate_tier0(
    questions: list[Question],
    retriever,
    cache: ResultCache,
    fingerprint: str,
    rewriter=None,
) -> list[QuestionResult]:
    """
    Retrieval only. No cross-encoder is constructed, so ~1.1 GB never loads
    and hundreds of questions run in seconds.

    Narrower than tier 1, not lesser: the Criminal Procedure Code's worst
    regression was Article 49 falling outside the top 30 for `arrest-bail`,
    which needed no reranking to detect.
    """

    retrieved, rewrites = _retrieve_all(
        questions, retriever, cache, fingerprint, rewriter
    )

    results = [
        _score_question(
            question,
            [c.get("chunk_id") for c in chunks],
            None,
        )
        for question, chunks in zip(questions, retrieved)
    ]

    for result, used in zip(results, rewrites):
        result.rewrites = used

    return results


def evaluate_tier1(
    questions: list[Question],
    retriever,
    reranker,
    cache: ResultCache,
    fingerprint: str,
    rewriter=None,
    rerank_rewrites: bool = False,
) -> list[QuestionResult]:
    """
    `reranker=None` skips the cross-encoder and takes retrieval's own top-k.

    `rerank_rewrites` is Phase C: the reranker reads the question followed
    by its rewrites. Without it, rewrites affect retrieval only - which is
    what keeps the two phases separately measurable.

    That is a measurement, not a mode to ship. Reranking was added when
    retrieval was dense-only and noisy; retrieval is now hybrid and at
    67/67, and the cross-encoder is discarding correct passages from ranks
    8 and 11. Comparing the two answers whether it still earns its place -
    and "delete it" is a legitimate outcome, worth ~1.1 GB and the dominant
    request latency.
    """

    retrieved, rewrites = _retrieve_all(
        questions, retriever, cache, fingerprint, rewriter
    )

    retrieved_ids = [
        [c.get("chunk_id") for c in chunks] for chunks in retrieved
    ]

    if reranker is None:
        results = [
            _score_question(question, ids, ids[:RERANK_TOP_K])
            for question, ids in zip(questions, retrieved_ids)
        ]

        for result, used in zip(results, rewrites):
            result.rewrites = used

        return results

    # What the cross-encoder actually reads for each question.
    queries = [
        rerank_query(question.question, used)
        if rerank_rewrites
        else question.question
        for question, used in zip(questions, rewrites)
    ]

    # Level 2 is keyed on the CANDIDATES, not the corpus, so a question whose
    # top-k did not move survives the arrival of a whole new document. And on
    # the text the reranker READ, not the question's id - so a Phase C result
    # can never be served to a run without Phase C, or the other way round.
    keys = [
        rerank_key(query, ids)
        for query, ids in zip(queries, retrieved_ids)
    ]

    reranked_ids: list[list[str] | None] = [
        cache.get("rerank", key) for key in keys
    ]

    pending = [i for i, value in enumerate(reranked_ids) if value is None]

    if pending:
        print()
        print(
            f"Reranking {len(pending)} of {len(questions)} "
            f"({len(questions) - len(pending)} reused)...",
            flush=True,
        )

        # One cross-encoder call for every question that needs one, rather
        # than one call per question with the batch mostly empty.
        batched = reranker.rerank_many(
            [(queries[i], retrieved[i]) for i in pending],
            top_k=RERANK_TOP_K,
        )

        for i, chunks in zip(pending, batched):
            ids = [c.get("chunk_id") for c in chunks]

            reranked_ids[i] = ids
            cache.put("rerank", keys[i], ids)

    results = [
        _score_question(question, ids, ranked)
        for question, ids, ranked in zip(
            questions, retrieved_ids, reranked_ids
        )
    ]

    for result, used in zip(results, rewrites):
        result.rewrites = used

    return results


def evaluate_tier2(
    questions: list[Question],
    rag,
    judge=None,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []

    for index, question in enumerate(questions, start=1):
        print(
            f"  [{index:>2}/{len(questions)}] {question.id}",
            flush=True,
        )

        outcome = rag.answer(question.question)

        # `considered` is what survived reranking and reached the LLM.
        # `cited` is what the answer actually relied on. Neither is the
        # raw retrieved set, which answer() does not expose.
        considered = [c.get("chunk_id") for c in outcome["considered"]]
        cited = [c.get("chunk_id") for c in outcome["sources"]]

        structured = outcome["structured"]

        result = QuestionResult(
            id=question.id,
            answerable=question.answerable,
            reranked_top=considered,
            cited_top=cited,
            sufficient=structured.sufficient,
            # Assertions the model made and could not support with a
            # quote from the source it cited. Dropped before the user
            # sees them; counted here because a rising number means the
            # model is straining against its sources.
            ungrounded=len(structured.ungrounded),
            refused_by_grounding=structured.refused_by_grounding,
            statements_kept=[s.text for s in structured.statements],
            statements_dropped=structured.statements_dropped,
            dropped=[
                {
                    "text": item.text,
                    "chunk_id": item.chunk_id,
                    "quote": item.quote,
                    "failure": item.failure,
                }
                for item in structured.ungrounded
            ],
            answer_text=outcome["answer"],
        )

        if question.answerable:
            expected = set(question.expect_any_of)

            # retrieval_hit stays None on purpose. Deriving it from the
            # reranked set would just duplicate rerank_hit and report a
            # retrieval failure that never happened.
            result.rerank_hit = bool(expected & set(considered))
            result.rerank_top1 = bool(
                considered and considered[0] in expected
            )

            # Seeing the right law and relying on it are different
            # things, and only this distinguishes them.
            result.cited_expected = bool(expected & set(cited))

        # Judge only what the user would actually read as an answer. A
        # refusal is scored by refusal_correct; asking the judge whether
        # it "responds to the question" would mark every correct refusal
        # as a failure.
        if judge is not None and structured.sufficient:
            # Every claim the user reads, qualifications included - an
            # overstated qualification misleads as much as a statement.
            claims = [
                item.text
                for item in structured.statements + structured.qualifications
            ]

            verdict = judge.judge(
                question=question.question,
                # Without the fixed advocate note: the judge checks what the
                # model wrote against the law, and asking it to find that
                # note in a section would invite a false rejection on every
                # answer.
                answer=render(structured, advocate_note=False),
                claims=claims,
                sources=outcome["sources"],
            )

            result.answer_quality = judge_quality(verdict)
            result.judge_defect, result.judge_reason = judge_headline(verdict)
            result.judge_claims = [
                {"claim": text, **check}
                for text, check in zip(claims, verdict.get("claims") or [])
            ]

            if verdict.get("error"):
                print(f"        [WARN] judge failed: {verdict['error']}")

        results.append(result)

    return results


def write_review(
    questions: list[Question],
    results: list[QuestionResult],
    limit: int,
) -> Path:
    """
    Write a sample of judged answers for a human to mark.

    The judge is a model, and this project has already been burned once by
    trusting an unchecked metric. Marking a sample by hand is what turns
    "the judge says 82%" into "the judge agrees with a person N times out
    of M".

    Flagged answers come first, and are capped at half the sample. A
    reviewer learns most from the ones the judge rejected, but a sample of
    nothing but rejections cannot show FALSE rejections - which is the
    failure mode that would quietly make the system look worse than it is.
    """

    judged = [r for r in results if r.answer_quality is not None]

    failed = [r for r in judged if not r.answer_quality]
    passed = [r for r in judged if r.answer_quality]

    chosen = failed[: max(1, limit // 2)]
    chosen += passed[: limit - len(chosen)]
    chosen = chosen[:limit]

    asked = {q.id: q.question for q in questions}

    directory = Path(__file__).resolve().parent / "review"
    directory.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    path = directory / f"{stamp}.md"

    lines = [
        f"# Answer review — {stamp}",
        "",
        "Mark each one `AGREE` or `DISAGREE` on the verdict line, and say",
        "why in a word. The number that matters is how often a person and",
        "the judge reach the same conclusion.",
        "",
    ]

    for result in chosen:
        lines += [
            "---",
            "",
            f"## {result.id}",
            "",
            f"**Question:** {asked.get(result.id, '?')}",
            "",
            f"**Judge:** {'PASS' if result.answer_quality else 'FAIL'}"
            f" — defect: `{result.judge_defect}`",
            "",
            f"> {result.judge_reason}",
            "",
            f"**Cited:** {', '.join(result.cited_top) or '(none)'}",
            "",
            "**Your verdict:** AGREE / DISAGREE — ",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Pocket Lawyer evaluation set."
    )

    parser.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help=(
            "0 = retrieval only, seconds, no reranker loaded; "
            "1 = retrieval + reranking (free); "
            "2 = full pipeline (uses the LLM)"
        ),
    )

    parser.add_argument(
        "--document",
        help=(
            "only questions expecting chunks from this document, by id "
            "prefix, e.g. penal-code. Out-of-scope questions reference no "
            "document and are therefore excluded"
        ),
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help=(
            "tier 1 without the cross-encoder - take retrieval's own top-k. "
            "A measurement of whether reranking still earns its place now "
            "that retrieval is hybrid, not a mode to ship"
        ),
    )

    parser.add_argument(
        "--rewrite",
        action="store_true",
        help=(
            "tiers 0 and 1: restate each question in legal register and "
            "retrieve with the rewrites too (Phase B). A measurement until "
            "QUERY_REWRITE is turned on - such a run is compared against "
            "the baselines but cannot replace them"
        ),
    )

    parser.add_argument(
        "--rerank-rewrites",
        action="store_true",
        help=(
            "tier 1 with --rewrite: the reranker reads the question followed "
            "by its rewrites as legal terms (Phase C). Without it, rewrites "
            "affect retrieval only, so the two phases stay separately "
            "measurable"
        ),
    )

    parser.add_argument(
        "--suite",
        choices=["statutory", "product", "all"],
        default="all",
        help=(
            "which question suite to run. statutory: worded like the "
            "statute, a retrieval regression test. product: worded like a "
            "person, the test of whether anyone is served. Always reported "
            "separately, never as one combined figure"
        ),
    )

    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "tier 2 without the answer-quality judge. Every other metric "
            "measures retrieval; this is the only one that reads the "
            "answer, so skip it only to save the API cost"
        ),
    )

    parser.add_argument(
        "--review",
        type=int,
        default=0,
        metavar="N",
        help=(
            "write N judged answers to evaluation/review/ for a human to "
            "mark. The judge is not trusted on its own word - this is how "
            "its agreement with a person gets measured"
        ),
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "recompute everything. Use after changing anything the cache "
            "key cannot see, such as a library upgrade"
        ),
    )

    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="record this run as the new reference",
    )

    parser.add_argument(
        "--id",
        action="append",
        help="run only this question id (repeatable)",
    )

    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="skip the baseline comparison",
    )

    parser.add_argument(
        "--accept-last",
        action="store_true",
        help=(
            "promote the previous run to the baseline without running "
            "again - use when you have already seen a result and want "
            "to record it"
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # PROMOTE A RUN YOU HAVE ALREADY SEEN
    # -----------------------------------------------------

    if args.accept_last:
        previous = load_last_run()

        if previous is None:
            print("[FAIL] no previous run recorded to promote.")
            print("       Run the evaluation first.")
            sys.exit(1)

        refusal = promotion_refusal(previous)

        if refusal:
            print(f"[FAIL] refusing to baseline the last run: {refusal}")
            sys.exit(1)

        _promote(previous)
        sys.exit(0)

    # -----------------------------------------------------
    # LOAD AND VALIDATE THE QUESTION SET
    # -----------------------------------------------------

    suites = ("statutory", "product") if args.suite == "all" else (args.suite,)

    questions = load_questions(suites=suites)

    corpus_errors = validate_against_corpus(questions, corpus_chunk_ids())

    if corpus_errors:
        print("[FAIL] question set references chunks that do not exist:")
        for error in corpus_errors:
            print(f"       {error}")
        sys.exit(1)

    if args.document:
        # The expected chunk ids already name their document, so no tag on
        # Question is needed: "penal-code@v2023-12-11-part-ii-..." .
        prefix = f"{args.document}@"

        questions = [
            q
            for q in questions
            if any(cid.startswith(prefix) for cid in q.expect_any_of)
        ]

        if not questions:
            print(f"[FAIL] no questions expect chunks from {args.document!r}")
            sys.exit(1)

    if args.id:
        wanted = set(args.id)
        questions = [q for q in questions if q.id in wanted]

        missing = wanted - {q.id for q in questions}

        if missing:
            print(f"[FAIL] unknown question id(s): {', '.join(sorted(missing))}")
            sys.exit(1)

    print("=" * 64)
    print(f"POCKET LAWYER — EVALUATION (tier {args.tier})")
    print("=" * 64)
    print()
    print(f"Questions  : {len(questions)}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Embedding  : {EMBEDDING_MODEL}")
    print(f"Reranker   : {RERANKER_MODEL}")
    print()

    # -----------------------------------------------------
    # RUN
    # -----------------------------------------------------

    # A baseline must be computed, never recalled - so a reference run reads
    # nothing. It still WRITES, because the values it computes are correct by
    # definition and discarding them would cost a second full run to warm the
    # cache.
    cache = ResultCache(
        read=not (args.no_cache or args.save_baseline),
        write=True,
    )

    fingerprint = corpus_fingerprint()

    rewriter = None

    if args.tier < 2 and (args.rewrite or QUERY_REWRITE):
        from backend.app.ai.rewrite import QueryRewriter

        # Every rewrite used is kept in a committed folder and reused: the
        # rewriter's model varies between calls, so this record is what
        # makes a rewrite run repeatable - and free once it exists.
        rewriter = QueryRewriter(cache_dir=REWRITES_DIR)

        print(f"Rewriting  : {rewriter.model_name}, up to "
              f"{rewriter.count} phrases per question")
        print()

    elif args.rewrite:
        # Tier 2 calls LegalRAG.answer(), which does not rewrite yet. Saying
        # so beats a run labelled "rewrite" that measured nothing of it.
        print("[FAIL] --rewrite applies to tiers 0 and 1 until rewriting "
              "is wired into the answering pipeline")
        sys.exit(1)

    if args.rerank_rewrites:
        # Checked before any model loads: each of these would otherwise
        # produce a run labelled Phase C that measured nothing of it.
        if rewriter is None:
            print("[FAIL] --rerank-rewrites needs --rewrite: without "
                  "rewrites there is nothing to give the reranker")
            sys.exit(1)

        if args.tier != 1 or args.no_rerank:
            print("[FAIL] --rerank-rewrites applies to tier 1 with the "
                  "reranker on - it changes what the reranker reads")
            sys.exit(1)

        print("Reranking  : the question followed by its rewritten terms "
              "(Phase C)")
        print()

    if args.tier < 2:
        # Deliberately not LegalRAG: it constructs LegalLLM, which
        # requires an API key. Tiers 0 and 1 must run without one.
        from backend.app.ai.retriever import LegalRetriever

        retriever = LegalRetriever()

        if args.tier == 0:
            print()
            print("Running (retrieval only)...")

            results = evaluate_tier0(
                questions, retriever, cache, fingerprint, rewriter
            )

        elif args.no_rerank:
            print()
            print("Running (NO reranking - retrieval top-k only)...")

            results = evaluate_tier1(
                questions, retriever, None, cache, fingerprint, rewriter
            )

        else:
            from backend.app.ai.reranker import LegalReranker

            print()
            reranker = LegalReranker()

            print()
            print("Running...")

            results = evaluate_tier1(
                questions,
                retriever,
                reranker,
                cache,
                fingerprint,
                rewriter,
                rerank_rewrites=args.rerank_rewrites,
            )

    else:
        from backend.app.ai.rag import LegalRAG

        rag = LegalRAG()

        judge = None

        if not args.no_judge:
            judge = AnswerJudge()
            print("[PASS] Answer-quality judge ready")

        print()
        print("Running...")

        results = evaluate_tier2(questions, rag, judge)

        if args.review:
            path = write_review(questions, results, args.review)
            print()
            print(f"[PASS] {args.review} answers written for review -> {path}")

    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    metadata = build_metadata(
        tier=args.tier,
        extra={
            # What this run covered. Promotion reads it: a run scoped to
            # one document or a handful of ids can look complete from its
            # results alone, and must never become the record.
            "scope": {
                "suites": list(suites),
                "document": args.document,
                "ids": sorted(args.id or []),
            },
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "rerank_top_k": RERANK_TOP_K,
            "reranked": not args.no_rerank,
            "judged": args.tier >= 2 and not args.no_judge,
            # Whether this run rewrote, and whether rewriting is the
            # pipeline. Promotion refuses a run where the two differ.
            "rewrite": rewriter is not None,
            "pipeline_rewrite": QUERY_REWRITE,
            "rewrite_model": rewriter.model_name if rewriter else None,
            "rewrite_count": REWRITE_COUNT if rewriter else None,
            "rewrite_top_k": REWRITE_TOP_K if rewriter else None,
            # Phase C. Always False without rewriting, which promotion
            # already refuses while QUERY_REWRITE is off.
            "rerank_rewrites": args.rerank_rewrites,
            "cache": cache.summary(),
        },
    )

    if rewriter is not None:
        rewritten = sum(1 for r in results if r.rewrites)

        print()
        print(
            f"Rewrites: {rewritten}/{len(results)} questions rewritten "
            f"({rewriter.hits} reused, {rewriter.calls} new, "
            f"{rewriter.failures} failed)"
        )

        if rewriter.failures:
            print(f"          last error: {rewriter.last_error}")

        if not rewritten:
            # Without this, a run with no key or a dead API would report
            # numbers identical to the baseline under a "rewrite" label.
            print(
                "[WARN] no question was rewritten, so this run measured "
                "nothing about rewriting. Check OPENAI_API_KEY."
            )

    print()
    print(
        f"Cache: retrieval {cache.hits['retrieval']} reused / "
        f"{cache.misses['retrieval']} computed"
    )

    if args.tier == 1:
        # The number that matters when a document is added: how many
        # questions had candidates that actually moved.
        print(
            f"       rerank    {cache.hits['rerank']} reused / "
            f"{cache.misses['rerank']} computed"
        )

    if not cache.read:
        print("       (reading disabled; results still written)")

    # Every evaluate_* function builds results from ids alone; the suite is
    # copied across here once rather than threaded through all of them.
    suite_of = {q.id: q.suite for q in questions}

    for result in results:
        result.suite = suite_of.get(result.id, "statutory")

    summary = summarize(results, tier=args.tier, metadata=metadata)

    by_id = {q.id: q for q in questions}

    print_run(summary, results, by_id, tier=args.tier)

    # Record every real run so it can be promoted later with --accept-last
    # instead of being regenerated.
    #
    # Measurements are excluded. A --no-rerank run is not the pipeline, and
    # letting it overwrite the record means a later --accept-last is looking
    # at the wrong thing - which is exactly what happened the first time this
    # flag was used.
    if not args.id and not args.no_rerank:
        save_last_run(summary)

    # -----------------------------------------------------
    # COMPARE / BASELINE
    # -----------------------------------------------------

    regressed = False

    # An --id run is excluded for the same reason it cannot be baselined: it
    # is one question against a whole-set reference. Comparing them reports
    # every other question as "removed from set" and calls the one question a
    # regression, then exits 1 - noise, during the debugging --id exists for.
    if not args.no_compare and not args.save_baseline and not args.id:
        # Each suite against its OWN baseline at this tier - so the
        # "comparison is not meaningful" warning a tier-2 run used to get
        # against the tier-1 record can no longer arise.
        for suite in summary["suites"]:
            label = f"{suite} suite, tier {args.tier}"

            baseline = load_baseline(suite, args.tier)

            if baseline is None:
                print(f"No baseline recorded yet for the {label}.")
                print(
                    "Promote a complete, uncached run of it with "
                    "--accept-last."
                )
                print()
                continue

            comparison = compare(suite_summary(summary, suite), baseline)
            print_comparison(comparison, label=label)
            regressed = regressed or bool(comparison["regressed"])

    if args.save_baseline:
        refusal = promotion_refusal(summary)

        if refusal:
            print(f"[FAIL] refusing to baseline this run: {refusal}")
            sys.exit(1)

        _promote(summary)
        print()

    sys.exit(1 if regressed else 0)


def _promote(summary: dict) -> None:
    """Write the run's baselines and say exactly which ones moved."""

    created = summary.get("metadata", {}).get("created", "unknown")
    tier = summary["metadata"]["tier"]

    for suite, path in save_baseline(summary).items():
        counts = summary["suites"][suite]["counts"]

        print(
            f"[PASS] {suite} tier {tier} baseline set from the run of "
            f"{created} ({counts['passed']}/{counts['questions']} passing)"
            f" -> {path.name}"
        )


if __name__ == "__main__":
    main()
