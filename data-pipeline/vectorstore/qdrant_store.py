import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "pocket_lawyer_legal"

INPUT_PATH = Path(
    "data/processed/constitution_embeddings.json"
)

VECTOR_SIZE = 1024


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings() -> list[dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Embedding file must contain a list of records."
        )

    return data


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection(client: QdrantClient) -> None:
    collections = client.get_collections().collections

    existing_names = {
        collection.name
        for collection in collections
    }

    if COLLECTION_NAME in existing_names:
        print(
            f"Collection already exists: {COLLECTION_NAME}"
        )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection created: {COLLECTION_NAME}"
    )


# ============================================================
# BUILD QDRANT POINTS
# ============================================================

def build_points(embeddings: list[dict]) -> list[PointStruct]:
    points = []

    for index, item in enumerate(embeddings):

        vector = item.get("embedding")

        if not vector:
            raise ValueError(
                f"Missing embedding for record {index}"
            )

        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"Invalid vector size for record {index}: "
                f"expected {VECTOR_SIZE}, got {len(vector)}"
            )

        payload = {
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get("document_id"),
            "document_type": item.get("document_type"),
            "title": item.get("title"),
            "jurisdiction": item.get("jurisdiction"),
            "language": item.get("language"),
            "chapter": item.get("chapter"),
            "article": item.get("article"),
            "content": item.get("content"),
            "source": item.get("source"),
        }

        points.append(
            PointStruct(
                id=index,
                vector=vector,
                payload=payload,
            )
        )

    return points


# ============================================================
# INSERT VECTORS
# ============================================================

def insert_vectors(
    client: QdrantClient,
    points: list[PointStruct],
) -> None:

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Vectors inserted: {len(points)}"
    )


# ============================================================
# VALIDATE COLLECTION
# ============================================================

def validate_collection(
    client: QdrantClient,
    expected_count: int,
) -> None:

    collection = client.get_collection(
        COLLECTION_NAME
    )

    actual_count = collection.points_count

    print()
    print("=" * 60)
    print("QDRANT VALIDATION")
    print("=" * 60)
    print()

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Vectors stored: {actual_count}"
    )

    print(
        f"Expected vectors: {expected_count}"
    )

    if actual_count != expected_count:
        raise RuntimeError(
            "Vector count does not match expected count."
        )

    print()
    print("[PASS] Collection exists")
    print("[PASS] Vector count matches")
    print("[PASS] Qdrant ingestion successful")
    print()
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)
    print("POCKET LAWYER — QDRANT VECTOR STORE")
    print("=" * 60)
    print()

    print(f"Qdrant: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Input: {INPUT_PATH}")
    print()

    print("Connecting to Qdrant...")

    client = get_client()

    # Verify connection.
    client.get_collections()

    print("[PASS] Connected to Qdrant")
    print()

    print("Loading embeddings...")

    embeddings = load_embeddings()

    print(
        f"Embeddings loaded: {len(embeddings)}"
    )
    print()

    print("Creating collection...")

    create_collection(client)

    print()

    print("Building Qdrant points...")

    points = build_points(embeddings)

    print(
        f"Points prepared: {len(points)}"
    )
    print()

    print("Uploading vectors...")

    insert_vectors(client, points)

    validate_collection(
        client,
        expected_count=len(points),
    )


if __name__ == "__main__":
    main()