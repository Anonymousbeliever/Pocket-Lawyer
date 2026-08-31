import sys

from backend.app.ai.retriever import LegalRetriever
from backend.app.ai.reranker import LegalReranker
from backend.app.ai.llm import LegalLLM
from backend.app.core.config import RERANK_TOP_K, RETRIEVAL_TOP_K


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

        Returns both the generated answer and the
        sources used to generate it.
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
            return {
                "answer": (
                    "I don't have enough reliable "
                    "information to answer this confidently. "
                    "Please consult a qualified advocate."
                ),
                "sources": [],
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
            return {
                "answer": (
                    "I don't have enough reliable "
                    "information to answer this confidently. "
                    "Please consult a qualified advocate."
                ),
                "sources": [],
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

        return {
            "answer": answer,
            "sources": reranked,
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
    # SOURCES
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("SOURCES USED")
    print("=" * 60)
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