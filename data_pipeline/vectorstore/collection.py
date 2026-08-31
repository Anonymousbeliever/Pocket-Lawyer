"""
Qdrant collection management.

The collection uses a *named* dense vector plus a declared sparse
vector slot. The sparse slot is intentionally left unpopulated for now:
declaring it costs nothing, and it means adding hybrid (dense + lexical)
search later will not require another full re-ingest. Legal queries lean
on exact anchors — "Section 45", "Cap 141", defined terms — that dense
vectors alone match poorly.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from backend.app.core.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    EMBEDDING_DIM,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
)


def get_client(url: str = QDRANT_URL) -> QdrantClient:
    client = QdrantClient(url=url)

    # Verify connection early with a clear message rather than failing
    # deep inside an upsert.
    try:
        client.get_collections()
    except Exception as error:
        raise RuntimeError(
            f"Could not reach Qdrant at {url}. "
            "Is it running? Try: docker compose up -d"
        ) from error

    return client


def collection_exists(
    client: QdrantClient,
    name: str = COLLECTION_NAME,
) -> bool:
    return any(
        collection.name == name
        for collection in client.get_collections().collections
    )


def has_named_dense_vector(
    client: QdrantClient,
    name: str = COLLECTION_NAME,
) -> bool:
    """
    True when the collection uses the named-vector schema this code
    writes. A collection created by the older pipeline has a single
    unnamed vector and will reject named-vector upserts.
    """

    vectors = client.get_collection(name).config.params.vectors

    if isinstance(vectors, dict):
        return DENSE_VECTOR_NAME in vectors

    # A bare VectorParams means the legacy unnamed-vector schema.
    return False


def ensure_collection(
    client: QdrantClient,
    name: str = COLLECTION_NAME,
    recreate: bool = False,
) -> None:
    """
    Create the collection, or verify an existing one is compatible.

    The compatibility check must happen before any destructive step in
    the caller: discovering the schema mismatch during the upsert
    leaves the collection already emptied.
    """

    if collection_exists(client, name):
        if not recreate:
            if not has_named_dense_vector(client, name):
                raise RuntimeError(
                    f"Collection '{name}' exists but uses the old "
                    "unnamed-vector schema, which cannot accept the "
                    f"named '{DENSE_VECTOR_NAME}' vector.\n"
                    "         Rebuild it in place by re-running with "
                    "--recreate, or set COLLECTION_NAME in .env to "
                    "ingest into a separate collection."
                )

            return

        client.delete_collection(collection_name=name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )

    print(f"Collection created: {name}")


def point_count(
    client: QdrantClient,
    name: str = COLLECTION_NAME,
) -> int:
    return client.get_collection(name).points_count
