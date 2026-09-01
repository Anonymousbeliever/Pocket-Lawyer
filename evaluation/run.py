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

Models are loaded once for the whole run, never per question.
"""

import argparse
import sys

from backend.app.core.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)
from evaluation.report import (
    build_metadata,
    compare,
    load_baseline,
    load_last_run,
    print_comparison,
    print_run,
    save_baseline,
    save_last_run,
    summarize,
)
from evaluation.schema import (
    Question,
    QuestionResult,
    corpus_chunk_ids,
    load_questions,
    validate_against_corpus,
)


def evaluate_tier1(
    questions: list[Question],
    retriever,
    reranker,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []

    for index, question in enumerate(questions, start=1):
        print(
            f"  [{index:>2}/{len(questions)}] {question.id}",
            flush=True,
        )

        retrieved = retriever.retrieve(
            query=question.question,
            top_k=RETRIEVAL_TOP_K,
        )

        reranked = reranker.rerank(
            query=question.question,
            documents=retrieved,
            top_k=RERANK_TOP_K,
        )

        retrieved_ids = [c.get("chunk_id") for c in retrieved]
        reranked_ids = [c.get("chunk_id") for c in reranked]

        result = QuestionResult(
            id=question.id,
            answerable=question.answerable,
            retrieved_top=retrieved_ids,
            reranked_top=reranked_ids,
        )

        # Only an answerable question has an authority to find.
        if question.answerable:
            expected = set(question.expect_any_of)

            result.retrieval_hit = bool(expected & set(retrieved_ids))
            result.rerank_hit = bool(expected & set(reranked_ids))
            result.rerank_top1 = bool(
                reranked_ids and reranked_ids[0] in expected
            )

        results.append(result)

    return results


def evaluate_tier2(
    questions: list[Question],
    rag,
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

        result = QuestionResult(
            id=question.id,
            answerable=question.answerable,
            reranked_top=considered,
            cited_top=cited,
            sufficient=outcome["structured"].sufficient,
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

        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Pocket Lawyer evaluation set."
    )

    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=1,
        help="1 = retrieval only (free), 2 = full pipeline (uses the LLM)",
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

        save_baseline(previous)

        created = previous.get("metadata", {}).get("created", "unknown")
        counts = previous.get("counts", {})

        print(
            f"[PASS] baseline set from the run of {created} "
            f"({counts.get('passed')}/{counts.get('questions')} passing)"
        )
        sys.exit(0)

    # -----------------------------------------------------
    # LOAD AND VALIDATE THE QUESTION SET
    # -----------------------------------------------------

    questions = load_questions()

    corpus_errors = validate_against_corpus(questions, corpus_chunk_ids())

    if corpus_errors:
        print("[FAIL] question set references chunks that do not exist:")
        for error in corpus_errors:
            print(f"       {error}")
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

    if args.tier == 1:
        # Deliberately not LegalRAG: it constructs LegalLLM, which
        # requires an API key. Tier 1 must run without one.
        from backend.app.ai.reranker import LegalReranker
        from backend.app.ai.retriever import LegalRetriever

        retriever = LegalRetriever()
        print()
        reranker = LegalReranker()

        print()
        print("Running...")

        results = evaluate_tier1(questions, retriever, reranker)

    else:
        from backend.app.ai.rag import LegalRAG

        rag = LegalRAG()

        print()
        print("Running...")

        results = evaluate_tier2(questions, rag)

    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    metadata = build_metadata(
        tier=args.tier,
        extra={
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "rerank_top_k": RERANK_TOP_K,
        },
    )

    summary = summarize(results, tier=args.tier, metadata=metadata)

    by_id = {q.id: q for q in questions}

    print_run(summary, results, by_id, tier=args.tier)

    # Record every run so it can be promoted later with --accept-last
    # instead of being regenerated.
    if not args.id:
        save_last_run(summary)

    # -----------------------------------------------------
    # COMPARE / BASELINE
    # -----------------------------------------------------

    regressed = False

    if not args.no_compare and not args.save_baseline:
        baseline = load_baseline()

        if baseline is None:
            print("No baseline recorded yet.")
            print("Save one with: python -m evaluation.run --save-baseline")
            print()
        else:
            comparison = compare(summary, baseline)
            print_comparison(comparison)
            regressed = bool(comparison["regressed"])

    if args.save_baseline:
        if args.id:
            print("[FAIL] refusing to baseline a partial run (--id given)")
            sys.exit(1)

        save_baseline(summary)
        print("[PASS] baseline saved")
        print()

    sys.exit(1 if regressed else 0)


if __name__ == "__main__":
    main()
