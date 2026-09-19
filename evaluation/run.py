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
from evaluation.cache import (
    ResultCache,
    corpus_fingerprint,
    rerank_key,
    retrieval_key,
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


def _retrieve_all(
    questions: list[Question],
    retriever,
    cache: ResultCache,
    fingerprint: str,
) -> list[list[dict]]:
    """
    Retrieve for every question, reusing level 1 where the corpus has not
    moved. Cheap either way - an embedding and a Qdrant query.
    """

    retrieved: list[list[dict]] = []

    for index, question in enumerate(questions, start=1):
        print(
            f"  [{index:>2}/{len(questions)}] {question.id}",
            flush=True,
        )

        key = retrieval_key(question.question, fingerprint)

        chunks = cache.get("retrieval", key)

        if chunks is None:
            chunks = retriever.retrieve(
                query=question.question,
                top_k=RETRIEVAL_TOP_K,
            )

            cache.put("retrieval", key, chunks)

        retrieved.append(chunks)

    return retrieved


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
) -> list[QuestionResult]:
    """
    Retrieval only. No cross-encoder is constructed, so ~1.1 GB never loads
    and hundreds of questions run in seconds.

    Narrower than tier 1, not lesser: the Criminal Procedure Code's worst
    regression was Article 49 falling outside the top 30 for `arrest-bail`,
    which needed no reranking to detect.
    """

    retrieved = _retrieve_all(questions, retriever, cache, fingerprint)

    return [
        _score_question(
            question,
            [c.get("chunk_id") for c in chunks],
            None,
        )
        for question, chunks in zip(questions, retrieved)
    ]


def evaluate_tier1(
    questions: list[Question],
    retriever,
    reranker,
    cache: ResultCache,
    fingerprint: str,
) -> list[QuestionResult]:
    retrieved = _retrieve_all(questions, retriever, cache, fingerprint)

    retrieved_ids = [
        [c.get("chunk_id") for c in chunks] for chunks in retrieved
    ]

    # Level 2 is keyed on the CANDIDATES, not the corpus, so a question whose
    # top-k did not move survives the arrival of a whole new document.
    keys = [
        rerank_key(question.question, ids)
        for question, ids in zip(questions, retrieved_ids)
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
            [(questions[i].question, retrieved[i]) for i in pending],
            top_k=RERANK_TOP_K,
        )

        for i, chunks in zip(pending, batched):
            ids = [c.get("chunk_id") for c in chunks]

            reranked_ids[i] = ids
            cache.put("rerank", keys[i], ids)

    return [
        _score_question(question, ids, ranked)
        for question, ids, ranked in zip(
            questions, retrieved_ids, reranked_ids
        )
    ]


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

            # Seeing the right law and relying on it are different
            # things, and only this distinguishes them.
            result.cited_expected = bool(expected & set(cited))

        results.append(result)

    return results


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

        # A reference result must be computed, not recalled.
        cache_meta = (previous.get("metadata") or {}).get("cache") or {}

        if any((cache_meta.get("hits") or {}).values()):
            print("[FAIL] refusing to baseline a run that used the cache.")
            print("       Re-run with --no-cache, then promote that.")
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

    if args.tier < 2:
        # Deliberately not LegalRAG: it constructs LegalLLM, which
        # requires an API key. Tiers 0 and 1 must run without one.
        from backend.app.ai.retriever import LegalRetriever

        retriever = LegalRetriever()

        if args.tier == 0:
            print()
            print("Running (retrieval only)...")

            results = evaluate_tier0(
                questions, retriever, cache, fingerprint
            )

        else:
            from backend.app.ai.reranker import LegalReranker

            print()
            reranker = LegalReranker()

            print()
            print("Running...")

            results = evaluate_tier1(
                questions, retriever, reranker, cache, fingerprint
            )

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
            "cache": cache.summary(),
        },
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
