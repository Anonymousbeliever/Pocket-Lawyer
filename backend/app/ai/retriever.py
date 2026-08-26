from pathlib import Path
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pocket_lawyer_legal"

EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_TOP_K = 15


class LegalRetriever:
    """
    Retrieves relevant legal chunks from Qdrant
    using BGE-M3 semantic embeddings.
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
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Retrieve the most semantically relevant legal chunks.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        retrieved_chunks = []

        for result in results:
            payload = result.payload or {}

            retrieved_chunks.append(
                {
                    "score": float(result.score),
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "document_type": payload.get("document_type"),
                    "title": payload.get("title"),
                    "chapter": payload.get("chapter"),
                    "article": payload.get("article"),
                    "content": payload.get("content"),
                    "source": payload.get("source"),
                }
            )

        return retrieved_chunks


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

    results = retriever.retrieve(
        query=query,
        top_k=15,
    )

    print(f"Results: {len(results)}")
    print()

    for index, result in enumerate(results, start=1):
        chapter = result.get("chapter") or {}
        article = result.get("article") or {}

        print("-" * 60)
        print(f"Result #{index}")
        print(f"Score: {result['score']:.4f}")

        print(
            f"Chapter: "
            f"{chapter.get('number')} — "
            f"{chapter.get('title')}"
        )

        print(
            f"Article: "
            f"{article.get('number')} — "
            f"{article.get('title')}"
        )

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