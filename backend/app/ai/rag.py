import sys

from backend.app.ai.answer import (
    LegalAnswer,
    cited_sources,
    render,
)
from backend.app.ai.retriever import LegalRetriever
from backend.app.ai.reranker import LegalReranker
from backend.app.ai.llm import LegalLLM
from backend.app.ai.rewrite import QueryRewriter, rerank_query
from backend.app.core.config import QUERY_REWRITE, RERANK_TOP_K, RETRIEVAL_TOP_K


def resolve_rewriter(rewriter=None):
    """
    The rewriter the pipeline uses: the one given, else a fresh one when
    QUERY_REWRITE is on, else none.

    The serving rewriter is built with NO store, deliberately: storing
    rewrites would write users' questions to disk. The evaluation passes
    its own rewriter, backed by the committed store, so tier 2 answers from
    exactly the rewrites tier 1 measured.
    """

    if rewriter is not None:
        return rewriter

    if QUERY_REWRITE:
        return QueryRewriter(cache_dir=None)

    return None


def _no_sources_answer(reason: str) -> LegalAnswer:
    """The result when retrieval produced nothing to reason over."""

    return LegalAnswer(
        sufficient=False,
        question_type="open",
        verdict="not_applicable",
        missing=reason,
    )


class LegalRAG:
    """
    Orchestrates the Pocket Lawyer RAG pipeline.

    Flow:

        User Question
              ↓
        Rewriting          (when QUERY_REWRITE is on)
              ↓
        Semantic Retrieval (the question + its rewrites)
              ↓
        Qdrant
              ↓
        Reranking          (reads the question + its rewrites)
              ↓
        LLM                (reads the question alone)
              ↓
        Grounded Answer

    The rewrites only decide which law is FOUND. The answering model never
    sees them - it answers the person's own words from real retrieved text,
    so a poor rewrite can cost a retrieval, never a citation.
    """

    def __init__(self, rewriter=None):
        print("Initializing Pocket Lawyer RAG...")
        print()

        # -------------------------------------------------
        # RETRIEVER
        # -------------------------------------------------

        print("Loading retriever...")

        self.retriever = LegalRetriever()

        print()

        # -------------------------------------------------
        # RERANKER
        # -------------------------------------------------

        print("Loading reranker...")

        self.reranker = LegalReranker()

        print()

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        print("Loading LLM...")

        self.llm = LegalLLM()

        print()

        # -------------------------------------------------
        # REWRITER
        # -------------------------------------------------

        self.rewriter = resolve_rewriter(rewriter)

        if self.rewriter is not None:
            print(
                f"[PASS] Query rewriting on "
                f"({self.rewriter.model_name})"
            )
            print()

        print("[PASS] RAG components initialized")

    # -----------------------------------------------------
    # RAG PIPELINE
    # -----------------------------------------------------

    def answer(
        self,
        question: str,
    ) -> dict:
        """
        Run the complete RAG pipeline.

        Returns:
            answer      rendered, user-facing text
            structured  the LegalAnswer object
            sources     only the sources the answer actually cited
            considered  every source that reached the LLM
            rewrites    the search phrases used besides the question -
                        empty when rewriting is off or produced nothing
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # -------------------------------------------------
        # STEP 0 — REWRITING
        # -------------------------------------------------

        # Empty when rewriting is off, and empty when it fails - the
        # rewriter degrades to [], and with [] every step below is exactly
        # what it was before rewriting existed.
        rewrites: list[str] = []

        if self.rewriter is not None:
            print()
            print("Rewriting the question into legal terms...")

            rewrites = self.rewriter.rewrite(question)

            if rewrites:
                print(f"[PASS] {len(rewrites)} search phrase(s)")
            else:
                print("[WARN] no rewrites - searching with the question alone")

        # -------------------------------------------------
        # STEP 1 — RETRIEVAL
        # -------------------------------------------------

        print()
        print("Retrieving legal sources...")

        retrieved = self.retriever.retrieve(
            query=question,
            top_k=RETRIEVAL_TOP_K,
            extra_queries=rewrites or None,
        )

        print(
            f"[PASS] Retrieved {len(retrieved)} "
            f"candidate sources"
        )

        if not retrieved:
            answer = _no_sources_answer(
                "No legal sources were retrieved for this question."
            )

            return {
                "answer": render(answer),
                "structured": answer,
                "sources": [],
                "considered": [],
                "rewrites": rewrites,
            }

        # -------------------------------------------------
        # STEP 2 — RERANKING
        # -------------------------------------------------

        print("Reranking legal sources...")

        # Phase C: the reranker reads the question followed by its
        # rewrites. With no rewrites this is the question itself, exactly as
        # before. Measured at tier 1 before being wired in: product 15 ->
        # 20/22 and statutory 106 -> 115/118 reach the LLM, no regressions.
        reranked = self.reranker.rerank(
            query=rerank_query(question, rewrites),
            documents=retrieved,
            top_k=RERANK_TOP_K,
        )

        print(
            f"[PASS] Reranked to {len(reranked)} "
            f"final sources"
        )

        if not reranked:
            answer = _no_sources_answer(
                "No retrieved source was relevant to this question."
            )

            return {
                "answer": render(answer),
                "structured": answer,
                "sources": [],
                "considered": [],
                "rewrites": rewrites,
            }

        # -------------------------------------------------
        # STEP 3 — LLM GENERATION
        # -------------------------------------------------

        print("Generating grounded answer...")

        answer = self.llm.generate(
            question=question,
            sources=reranked,
        )

        print("[PASS] Answer generated")

        if answer.ungrounded:
            print(
                f"[WARN] dropped {len(answer.ungrounded)} assertion(s) "
                f"whose quote was not found in the source they cited"
            )

        if answer.refused_by_grounding:
            print(
                "[WARN] the model answered, but too few of its claims "
                "survived verification - refused by the grounding check"
            )

        return {
            "answer": render(answer),
            "structured": answer,
            "sources": cited_sources(answer, reranked),
            "considered": reranked,
            "rewrites": rewrites,
        }


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------


def main():
    print("=" * 60)
    print("POCKET LAWYER — RAG PIPELINE")
    print("=" * 60)
    print()

    try:
        rag = LegalRAG()

    except Exception as error:
        print()
        print("[ERROR] Failed to initialize RAG pipeline")
        print()
        print(error)
        sys.exit(1)

    print()
    print("Models are loaded. Ask as many questions as you like;")
    print("blank line or Ctrl-C to quit.")

    while True:
        print()

        try:
            question = input(
                "Enter your legal question: "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            break

        answer_one(rag, question)

    print()
    print("=" * 60)
    print("RAG pipeline completed.")
    print("=" * 60)


def answer_one(rag: LegalRAG, question: str) -> None:
    """
    Answer one question and print it.

    Split out of main() so the models load once and then serve any
    number of questions. Reloading BGE-M3 and the cross-encoder per
    question costs ~2.5 GB and most of a minute, which made the CLI
    unusable for anything but a single lookup.
    """

    print()
    print("=" * 60)
    print("POCKET LAWYER — RAG")
    print("=" * 60)
    print()

    print(f"Question: {question}")
    print()

    try:
        result = rag.answer(question)

    except Exception as error:
        print()
        print("[ERROR] RAG pipeline failed")
        print()
        print(error)
        return

    # -----------------------------------------------------
    # FINAL ANSWER
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print()

    print(result["answer"])

    # -----------------------------------------------------
    # STRUCTURED FIELDS
    # -----------------------------------------------------

    structured = result["structured"]

    print()
    print("-" * 60)
    print(
        f"sufficient={structured.sufficient}  "
        f"type={structured.question_type}  "
        f"verdict={structured.verdict}"
    )
    print(
        f"cited {len(result['sources'])} of "
        f"{len(result['considered'])} sources considered"
    )

    if result.get("rewrites"):
        print(f"searched with: {result['rewrites']}")

    print("-" * 60)

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("SOURCES CITED")
    print("=" * 60)
    print()

    if not result["sources"]:
        print("(none)")
        print()

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"SOURCE {index}"
        )

        print(
            f"Document: "
            f"{source.get('title')}"
        )

        print(
            f"Citation: "
            f"{source.get('citation')} — "
            f"{source.get('unit_title')}"
        )

        print(
            f"Rerank score: "
            f"{source.get('rerank_score', 0):.4f}"
        )

        print(
            f"Published by: "
            f"{source.get('source_name')}"
        )

        print(
            f"Current as at: "
            f"{source.get('as_at')}"
        )

        print(
            f"Chunk ID: "
            f"{source.get('chunk_id')}"
        )

        print()


if __name__ == "__main__":
    main()