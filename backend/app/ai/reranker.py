from sentence_transformers import CrossEncoder

from backend.app.core.config import (
    RERANK_RELATIVE_RATIO,
    RERANK_TOP_K,
    RERANKER_MODEL,
)
from backend.app.core.passage import contextual_text


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
        top_k: int = RERANK_TOP_K,
        relative_ratio: float = RERANK_RELATIVE_RATIO,
    ) -> list[dict]:
        """
        Rerank retrieved legal documents and trim the weak tail.

        Retrieval always returns its full limit, so the bottom of the
        list is usually unrelated to the question. Those are dropped
        relative to the best match for this same query:

            cut = top_score * relative_ratio

        The top result always survives, so reranking never refuses on
        its own. That is deliberate. Absolute cross-encoder scores are
        not comparable between queries — a typo dropped Article 49
        from 0.1344 to 0.0086 while still ranking it first, below the
        0.0373 that an unanswerable question scored. Whether the
        sources actually answer the question is decided by the LLM
        under its grounding prompt, not here.

        Args:
            query:
                User's legal question.

            documents:
                Legal chunks returned by the retriever.

            top_k:
                Maximum number of results to return after reranking.

            relative_ratio:
                Fraction of the top score a result must reach to be
                kept.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Score against the same contextual form used at ingest time:
        # the citation and title give the cross-encoder something to
        # judge, which bare list-shaped schedule text does not.
        pairs = [
            [
                query,
                contextual_text(document),
            ]
            for document in documents
        ]

        # Score every question/document pair.
        scores = [float(score) for score in self.model.predict(pairs)]

        # Relative to this query's own best match, so the cut adapts to
        # however this phrasing happened to score.
        cut = max(scores) * relative_ratio

        reranked = []

        for document, score in zip(documents, scores):
            if score < cut:
                continue

            result = document.copy()

            result["rerank_score"] = score

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
            "citation": "Chapter Four — Article 49",
            "unit_title": "Rights of arrested persons",
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
            "citation": "Chapter Four — Article 51",
            "unit_title": (
                "Rights of persons detained, held in "
                "custody or imprisoned"
            ),
            "content": (
                "A person who is detained, held in custody "
                "or imprisoned under the law retains all "
                "rights and fundamental freedoms in the "
                "Bill of Rights."
            ),
        },
        {
            "chunk_id": "constitution-article-29",
            "citation": "Chapter Four — Article 29",
            "unit_title": "Freedom and security of the person",
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
    print(f"Relative ratio: {RERANK_RELATIVE_RATIO}")
    print()

    results = reranker.rerank(
        query=query,
        documents=documents,
    )

    print(f"Results kept: {len(results)} of {len(documents)}")
    print()

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("-" * 60)
        print(f"Result #{index}")

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print(f"Citation: {result.get('citation')}")
        print(f"Title: {result.get('unit_title')}")

        print()
        print("Content:")
        print(result.get("content", ""))

    print()
    print("=" * 60)
    print("Reranking completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
