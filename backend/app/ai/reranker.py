from sentence_transformers import CrossEncoder


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

DEFAULT_TOP_K = 5


class LegalReranker:
    """
    Reranks legal chunks retrieved from Qdrant.

    The reranker examines the user's question together
    with each legal passage and assigns a relevance score.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
    ):
        self.model_name = model_name

        print(f"Loading reranker model: {self.model_name}")

        self.model = CrossEncoder(
            self.model_name
        )

        print("[PASS] Reranker model loaded")

    # -----------------------------------------------------
    # RERANK
    # -----------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Rerank retrieved legal documents.

        Args:
            query:
                User's legal question.

            documents:
                Legal chunks returned by the retriever.

            top_k:
                Number of results to return after reranking.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Create question/document pairs.
        pairs = [
            [
                query,
                document.get("content", ""),
            ]
            for document in documents
        ]

        # Score every question/document pair.
        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(documents, scores):
            result = document.copy()

            result["rerank_score"] = float(score)

            reranked.append(result)

        # Highest relevance first.
        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("POCKET LAWYER — LEGAL RERANKER")
    print("=" * 60)
    print()

    reranker = LegalReranker()

    # -----------------------------------------------------
    # Test documents
    # -----------------------------------------------------

    documents = [
        {
            "chunk_id": "constitution-article-49",
            "article": {
                "number": 49,
                "title": "Rights of arrested persons",
            },
            "content": (
                "(1) An arrested person has the right— "
                "(a) to be informed promptly, in a language "
                "that the person understands, of the reason "
                "for the arrest; "
                "(b) to remain silent; "
                "(c) to communicate with an advocate."
            ),
        },
        {
            "chunk_id": "constitution-article-51",
            "article": {
                "number": 51,
                "title": (
                    "Rights of persons detained, held in "
                    "custody or imprisoned"
                ),
            },
            "content": (
                "A person who is detained, held in custody "
                "or imprisoned under the law retains all "
                "rights and fundamental freedoms in the "
                "Bill of Rights."
            ),
        },
        {
            "chunk_id": "constitution-article-29",
            "article": {
                "number": 29,
                "title": "Freedom and security of the person",
            },
            "content": (
                "Every person has the right to freedom and "
                "security of the person, including the right "
                "not to be deprived of freedom arbitrarily."
            ),
        },
    ]

    query = input(
        "Enter your legal question: "
    ).strip()

    if not query:
        print("No question provided.")
        return

    print()
    print("=" * 60)
    print("LEGAL RERANKING")
    print("=" * 60)
    print()

    print(f"Query: {query}")
    print()

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
    )

    print(f"Results: {len(results)}")
    print()

    for index, result in enumerate(
        results,
        start=1,
    ):
        article = result.get("article") or {}

        print("-" * 60)
        print(f"Result #{index}")

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Article: "
            f"{article.get('number')} — "
            f"{article.get('title')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print()
        print("Content:")
        print(result.get("content", ""))

    print()
    print("=" * 60)
    print("Reranking completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()