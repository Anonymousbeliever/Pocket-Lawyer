import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector as QdrantSparseVector,
)
from sentence_transformers import SentenceTransformer

from backend.app.core.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    EMBEDDING_MODEL,
    HYBRID_RETRIEVAL,
    QDRANT_URL,
    RETRIEVAL_TOP_K,
    SPARSE_TOP_K,
    SPARSE_VECTOR_NAME,
)
from backend.app.core.sparse import LexicalEncoder


class LegalRetriever:
    """
    Retrieves relevant legal chunks from Qdrant
    using BGE-M3 semantic embeddings.

    The embedding model here must match the one used at ingest time.
    Both read it from backend.app.core.config for exactly that reason.
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        print("Connecting to Qdrant...")
        self.client = QdrantClient(url=self.qdrant_url)

        # Verify connection
        self.client.get_collections()

        print("[PASS] Connected to Qdrant")

        print()
        print(f"Loading embedding model: {self.embedding_model_name}")

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        print("[PASS] Embedding model loaded")

        # Lexical weights from the same model - only a Linear(1024, 1)
        # head on top, so no second model and no extra memory.
        self.lexical = None

        if HYBRID_RETRIEVAL:
            self.lexical = LexicalEncoder(
                self.embedding_model,
                model_name=self.embedding_model_name,
            )

            print("[PASS] Sparse head loaded (hybrid retrieval)")

    # -----------------------------------------------------
    # QUERY EMBEDDING
    # -----------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a user's legal question into a vector.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # -----------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        in_force_only: bool = True,
    ) -> list[dict]:
        """
        Retrieve the most semantically relevant legal chunks.

        By default only law that is currently in force is searched.
        Answering with repealed law is the worst failure mode this
        system has, so the filter is opt-out rather than opt-in.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.embed_query(query)

        query_filter = None

        if in_force_only:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="in_force",
                        match=MatchValue(value=True),
                    )
                ]
            )

        results = self._search(
            query=query,
            query_vector=query_vector,
            query_filter=query_filter,
            top_k=top_k,
        )

        retrieved_chunks = []

        for result in results:
            payload = result.payload or {}

            chunk = dict(payload)
            chunk["score"] = float(result.score)

            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def _search(
        self,
        query: str,
        query_vector: list[float],
        query_filter: Filter | None,
        top_k: int,
        sparse_top_k: int = SPARSE_TOP_K,
    ):
        """
        Dense alone, or dense and lexical fused.

        Fusion is **Reciprocal Rank Fusion**, which combines the two
        result lists by RANK rather than by score. That matters here
        specifically: this project has twice been burned by absolute
        similarity scores not being comparable - the rerank floor, and
        the subject-match gate - and dense cosine and sparse dot product
        are on entirely different scales. RRF never has to reconcile
        them.
        """

        if self.lexical is None:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ).points

        sparse = self.lexical.encode_one(query)

        if sparse.is_empty:
            # Nothing lexical to match on - a query of pure stopwords, or
            # one the tokenizer reduced to specials. Dense alone is still
            # a correct answer, so degrade rather than fail.
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ).points

        # The budgets are deliberately unequal. Dense is the primary
        # signal; lexical is a corrective for exact anchors, and given an
        # equal list it drowns dense-only hits - see SPARSE_TOP_K.
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=top_k,
                ),
                Prefetch(
                    query=QdrantSparseVector(
                        indices=sparse.indices,
                        values=sparse.values,
                    ),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=min(sparse_top_k, top_k),
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("POCKET LAWYER — LEGAL RETRIEVER")
    print("=" * 60)
    print()

    retriever = LegalRetriever()

    print()
    query = input("Enter your legal question: ").strip()

    if not query:
        print("No question provided.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("LEGAL RETRIEVAL")
    print("=" * 60)
    print()

    print(f"Query: {query}")
    print()

    results = retriever.retrieve(query=query)

    print(f"Results: {len(results)}")
    print()

    for index, result in enumerate(results, start=1):
        print("-" * 60)
        print(f"Result #{index}")
        print(f"Score: {result['score']:.4f}")
        print(f"Citation: {result.get('citation')}")
        print(f"Title: {result.get('unit_title')}")

        print()
        print("Content:")
        print(result.get("content", ""))

        print()
        print(f"Chunk ID: {result.get('chunk_id')}")

    print()
    print("=" * 60)
    print("Retrieval completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
