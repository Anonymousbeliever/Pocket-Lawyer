"""
Streaming embed-and-store.

Chunks are embedded in batches and upserted in batches. Vectors are
never all held in memory and never written to disk — the old
`*_embeddings.json` intermediate was 6.3 MB for a single document and
was being committed to git.

Re-ingesting a document deletes its existing points first, then writes
fresh ones. Combined with deterministic point IDs this makes ingestion
idempotent: running it twice leaves the collection identical, and a
document whose chunk set shrank leaves no orphaned vectors behind.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

from backend.app.core.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    UPSERT_BATCH_SIZE,
)
from data_pipeline.chunking.ids import point_id


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL,
) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    print("[PASS] Embedding model loaded")

    return model


def delete_document(
    client: QdrantClient,
    document_id: str,
    collection: str = COLLECTION_NAME,
) -> None:
    """Remove every point belonging to a document."""

    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def ingest_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[dict],
    collection: str = COLLECTION_NAME,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    upsert_batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """Embed and store chunks. Returns the number of points written."""

    if not chunks:
        raise ValueError("No chunks to ingest.")

    pending: list[PointStruct] = []
    written = 0

    for batch in _batched(chunks, embed_batch_size):

        vectors = model.encode(
            [chunk["content"] for chunk in batch],
            normalize_embeddings=True,
        )

        for chunk, vector in zip(batch, vectors):
            pending.append(
                PointStruct(
                    id=point_id(chunk["chunk_id"]),
                    vector={DENSE_VECTOR_NAME: vector.tolist()},
                    payload=chunk,
                )
            )

        if len(pending) >= upsert_batch_size:
            client.upsert(collection_name=collection, points=pending)
            written += len(pending)
            print(f"  stored {written}/{len(chunks)}")
            pending = []

    if pending:
        client.upsert(collection_name=collection, points=pending)
        written += len(pending)
        print(f"  stored {written}/{len(chunks)}")

    return written
