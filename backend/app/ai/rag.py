import sys

from backend.app.ai.answer import (
    LegalAnswer,
    cited_sources,
    render,
)
from backend.app.ai.retriever import LegalRetriever
from backend.app.ai.reranker import LegalReranker
from backend.app.ai.llm import LegalLLM
from backend.app.core.config import RERANK_TOP_K, RETRIEVAL_TOP_K


def _no_sources_answer(reason: str) -> LegalAnswer:
    """The result when retrieval produced nothing to reason over."""

    return LegalAnswer(
        sufficient=False,
        question_type="open",
        verdict="not_applicable",
        explanation=reason,
    )


class LegalRAG:
    """
    Orchestrates the Pocket Lawyer RAG pipeline.

    Flow:

        User Question
              ↓
        Semantic Retrieval
              ↓
        Qdrant
              ↓
        Reranking
              ↓
        LLM
              ↓
        Grounded Answer
    """

    def __init__(self):
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
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # -------------------------------------------------
        # STEP 1 — RETRIEVAL
        # -------------------------------------------------

        print()
        print("Retrieving legal sources...")

        retrieved = self.retriever.retrieve(
            query=question,
            top_k=RETRIEVAL_TOP_K,
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
            }

        # -------------------------------------------------
        # STEP 2 — RERANKING
        # -------------------------------------------------

        print("Reranking legal sources...")

        reranked = self.reranker.rerank(
            query=question,
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

        if answer.unverified_citations:
            print(
                f"[WARN] removed {len(answer.unverified_citations)} "
                f"citation(s) not present in the retrieved sources: "
                f"{answer.unverified_citations}"
            )

        return {
            "answer": render(answer),
            "structured": answer,
            "sources": cited_sources(answer, reranked),
            "considered": reranked,
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

    question = input(
        "Enter your legal question: "
    ).strip()

    if not question:
        print("No question provided.")
        sys.exit(1)

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
        sys.exit(1)

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

    print("=" * 60)
    print("RAG pipeline completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()