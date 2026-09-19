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
        max_per_document: int | None = None,
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

        One document may not take every slot. A citizen's question
        usually has its *right* in the Constitution and its *procedure*
        in an Act, and until the Criminal Procedure Code was indexed
        that distinction could not arise — there was only one document.
        With two, "Do I have to answer police questions after being
        arrested?" put five Criminal Procedure Code sections in front of
        the LLM and none of Article 49, because s.36A (Remand by court)
        contains "inquiries ... by the police" while the constitutional
        right is worded "the right to remain silent". The system then
        refused a question the corpus answers.

        So the best-scoring passage from a document that would otherwise
        be shut out keeps a slot, even below the tail cut. This is a
        structural rule rather than another threshold, deliberately:
        cross-encoder scores are not trustworthy enough to tune against.

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

            max_per_document:
                Most slots any one document may occupy. Defaults to
                top_k - 1, which reserves exactly one slot for the
                strongest passage from somewhere else.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        scores = self._score(query, documents)

        return self._select(
            documents,
            scores,
            top_k=top_k,
            relative_ratio=relative_ratio,
            max_per_document=max_per_document,
        )

    def rerank_many(
        self,
        batches: list[tuple[str, list[dict]]],
        top_k: int = RERANK_TOP_K,
        relative_ratio: float = RERANK_RELATIVE_RATIO,
        max_per_document: int | None = None,
    ) -> list[list[dict]]:
        """
        Rerank several queries in ONE cross-encoder call.

        Identical output to calling `rerank` on each batch in turn - the
        selection below is the same function - but a single `predict` over
        every pair uses the CPU far better than one call per question.

        This is the evaluation harness's path. Scoring 82 questions
        separately means 82 round trips through the model with the batch
        mostly empty; scoring them together fills it.
        """

        eligible = [
            (index, query, documents)
            for index, (query, documents) in enumerate(batches)
            if query.strip() and documents
        ]

        if not eligible:
            return [[] for _ in batches]

        pairs: list[list[str]] = []

        for _, query, documents in eligible:
            pairs.extend(self._pairs(query, documents))

        flat = [float(score) for score in self.model.predict(pairs)]

        results: list[list[dict]] = [[] for _ in batches]

        offset = 0

        for index, _, documents in eligible:
            scores = flat[offset : offset + len(documents)]
            offset += len(documents)

            results[index] = self._select(
                documents,
                scores,
                top_k=top_k,
                relative_ratio=relative_ratio,
                max_per_document=max_per_document,
            )

        return results

    @staticmethod
    def _pairs(query: str, documents: list[dict]) -> list[list[str]]:
        """
        Score against the same contextual form used at ingest time: the
        citation and title give the cross-encoder something to judge, which
        bare list-shaped schedule text does not.
        """

        return [[query, contextual_text(document)] for document in documents]

    def _score(self, query: str, documents: list[dict]) -> list[float]:
        return [
            float(score)
            for score in self.model.predict(self._pairs(query, documents))
        ]

    def _select(
        self,
        documents: list[dict],
        scores: list[float],
        top_k: int,
        relative_ratio: float,
        max_per_document: int | None,
    ) -> list[dict]:
        """
        Tail trim, then diversity, then top_k.

        Extracted unchanged from `rerank` so that a batched scoring pass can
        reuse it. Everything here was already the behaviour.
        """

        scored = []

        for document, score in zip(documents, scores):
            result = document.copy()

            result["rerank_score"] = score

            scored.append(result)

        # Highest relevance first.
        scored.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        # Relative to this query's own best match, so the cut adapts to
        # however this phrasing happened to score.
        cut = max(scores) * relative_ratio

        sources = {item.get("document_id") for item in scored}

        # With one document in play there is nothing to diversify, and
        # capping slots would only discard good matches. This is also
        # what keeps behaviour identical to before the corpus grew.
        if len(sources) < 2:
            return [
                item
                for item in scored
                if item["rerank_score"] >= cut
            ][:top_k]

        return self._select_across_documents(
            scored,
            cut=cut,
            top_k=top_k,
            max_per_document=(
                max(1, top_k - 1)
                if max_per_document is None
                else max_per_document
            ),
        )

    @staticmethod
    def _select_across_documents(
        scored: list[dict],
        cut: float,
        top_k: int,
        max_per_document: int,
    ) -> list[dict]:
        """
        Fill top_k without letting one document take every slot.

        Two passes over the same score-ordered list. The first is
        ordinary selection — anything clearing the tail cut, capped per
        document. The second spends whatever slots are left on the best
        passage from a document that got nothing, and is the only place
        a below-cut passage can be kept.
        """

        selected: list[dict] = []
        taken: dict[str, int] = {}

        for item in scored:
            if len(selected) >= top_k:
                break

            if item["rerank_score"] < cut:
                continue

            source = item.get("document_id")

            if taken.get(source, 0) >= max_per_document:
                continue

            selected.append(item)
            taken[source] = taken.get(source, 0) + 1

        for item in scored:
            if len(selected) >= top_k:
                break

            source = item.get("document_id")

            if source in taken:
                continue

            selected.append(item)
            taken[source] = 1

        selected.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return selected


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
