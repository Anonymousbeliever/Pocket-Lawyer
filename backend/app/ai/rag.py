from pathlib import Path
import sys

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Allow imports from backend/app
APP_DIR = PROJECT_ROOT / "backend" / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from ai.retriever import LegalRetriever
from ai.reranker import LegalReranker


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RETRIEVAL_TOP_K = 15
RERANK_TOP_K = 5


class LegalRAG:
    """
    Orchestrates the retrieval and reranking stages
    of the Pocket Lawyer RAG system.

    Current pipeline:

        Question
            ↓
        BGE-M3 embedding
            ↓
        Qdrant retrieval
            ↓
        Top 15 candidates
            ↓
        BGE reranker
            ↓
        Top 5 legal sources

    The LLM answer-generation stage will be added later.
    """

    def __init__(
        self,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
    ):
        if retrieval_top_k < 1:
            raise ValueError(
                "retrieval_top_k must be at least 1."
            )

        if rerank_top_k < 1:
            raise ValueError(
                "rerank_top_k must be at least 1."
            )

        if rerank_top_k > retrieval_top_k:
            raise ValueError(
                "rerank_top_k cannot be greater than "
                "retrieval_top_k."
            )

        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

        print("Initializing Pocket Lawyer RAG...")
        print()

        print("Loading retriever...")
        self.retriever = LegalRetriever()

        print()

        print("Loading reranker...")
        self.reranker = LegalReranker()

        print()
        print("[PASS] RAG components initialized")

    # -----------------------------------------------------
    # RAG RETRIEVAL PIPELINE
    # -----------------------------------------------------

    def retrieve_context(
        self,
        query: str,
    ) -> list[dict]:
        """
        Retrieve and rerank legal sources for a question.

        Returns the highest-ranked legal chunks that
        will eventually be supplied to the LLM.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # -------------------------------------------------
        # STEP 1 — SEMANTIC RETRIEVAL
        # -------------------------------------------------

        retrieved_documents = self.retriever.retrieve(
            query=query,
            top_k=self.retrieval_top_k,
        )

        if not retrieved_documents:
            return []

        # -------------------------------------------------
        # STEP 2 — RERANK
        # -------------------------------------------------

        reranked_documents = self.reranker.rerank(
            query=query,
            documents=retrieved_documents,
            top_k=self.rerank_top_k,
        )

        return reranked_documents

    # -----------------------------------------------------
    # BUILD LLM CONTEXT
    # -----------------------------------------------------

    def build_context(
        self,
        documents: list[dict],
    ) -> str:
        """
        Convert retrieved legal documents into a context
        string that can later be supplied to an LLM.
        """

        if not documents:
            return ""

        context_parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            chapter = document.get("chapter") or {}
            article = document.get("article") or {}
            source = document.get("source") or {}

            chapter_number = chapter.get(
                "number",
                "Unknown",
            )

            chapter_title = chapter.get(
                "title",
                "Unknown",
            )

            article_number = article.get(
                "number",
                "Unknown",
            )

            article_title = article.get(
                "title",
                "Unknown",
            )

            source_name = source.get(
                "name",
                "Unknown",
            )

            content = document.get(
                "content",
                "",
            ).strip()

            context_parts.append(
                f"""SOURCE {index}
Document: {document.get("title", "Unknown")}
Chapter: {chapter_number} — {chapter_title}
Article: {article_number} — {article_title}
Source: {source_name}
Chunk ID: {document.get("chunk_id", "Unknown")}

Content:
{content}
"""
            )

        return "\n\n".join(context_parts)

    # -----------------------------------------------------
    # COMPLETE CURRENT PIPELINE
    # -----------------------------------------------------

    def run(
        self,
        query: str,
    ) -> dict:
        """
        Execute the current RAG pipeline.

        At this stage this method retrieves and reranks
        sources but does not generate an answer yet.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        documents = self.retrieve_context(
            query=query,
        )

        context = self.build_context(
            documents,
        )

        return {
            "query": query,
            "retrieved_documents": documents,
            "context": context,
        }


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("POCKET LAWYER — RAG PIPELINE")
    print("=" * 60)
    print()

    rag = LegalRAG()

    print()
    query = input(
        "Enter your legal question: "
    ).strip()

    if not query:
        print("No question provided.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("RAG RETRIEVAL PIPELINE")
    print("=" * 60)
    print()

    print(f"Question: {query}")
    print()

    result = rag.run(query)

    documents = result["retrieved_documents"]

    print(
        f"Final sources: {len(documents)}"
    )

    print()

    for index, document in enumerate(
        documents,
        start=1,
    ):
        chapter = document.get("chapter") or {}
        article = document.get("article") or {}

        print("-" * 60)
        print(f"FINAL SOURCE #{index}")

        print(
            f"Rerank score: "
            f"{document.get('rerank_score', 0):.4f}"
        )

        print(
            f"Article: "
            f"{article.get('number')} — "
            f"{article.get('title')}"
        )

        print(
            f"Chunk ID: "
            f"{document.get('chunk_id')}"
        )

        print()
        print("Content:")
        print(
            document.get(
                "content",
                "",
            )
        )

    print()
    print("=" * 60)
    print("LLM CONTEXT")
    print("=" * 60)
    print()

    print(result["context"])

    print("=" * 60)
    print("RAG retrieval pipeline completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()