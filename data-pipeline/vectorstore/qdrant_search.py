import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "pocket_lawyer_legal"

EMBEDDING_MODEL = "BAAI/bge-m3"

TOP_K = 5


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[PASS] Embedding model loaded")

    return model


# ============================================================
# CONNECT TO QDRANT
# ============================================================

def get_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)

    # Verify connection.
    client.get_collections()

    return client


# ============================================================
# SEARCH
# ============================================================

def search(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K,
):
    # Convert the user's question into a BGE-M3 vector.
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(query: str, results) -> None:

    print()
    print("=" * 60)
    print("POCKET LAWYER — SEMANTIC SEARCH")
    print("=" * 60)

    print()
    print(f"Query: {query}")

    print()
    print(f"Results: {len(results)}")
    print()

    for index, result in enumerate(results, start=1):

        payload = result.payload or {}

        article = payload.get("article") or {}
        chapter = payload.get("chapter") or {}

        print("-" * 60)

        print(f"Result #{index}")
        print(f"Score: {result.score:.4f}")

        print(
            f"Chapter: "
            f"{chapter.get('number', 'N/A')} — "
            f"{chapter.get('title', 'N/A')}"
        )

        print(
            f"Article: "
            f"{article.get('number', 'N/A')} — "
            f"{article.get('title', 'N/A')}"
        )

        print()

        content = payload.get("content", "")

        print(f"Content:\n{content}")

        print()

        print(
            f"Chunk ID: "
            f"{payload.get('chunk_id', 'N/A')}"
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("POCKET LAWYER — QDRANT SEARCH")
    print("=" * 60)
    print()

    print(f"Qdrant: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Model: {EMBEDDING_MODEL}")
    print()

    print("Connecting to Qdrant...")

    client = get_client()

    print("[PASS] Connected to Qdrant")
    print()

    model = load_model()

    print()

    query = input(
        "Enter your legal question: "
    ).strip()

    if not query:
        raise SystemExit(
            "Query cannot be empty."
        )

    results = search(
        client=client,
        model=model,
        query=query,
    )

    display_results(
        query=query,
        results=results,
    )


if __name__ == "__main__":
    main()