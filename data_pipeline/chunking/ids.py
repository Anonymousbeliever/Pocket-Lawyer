"""
Deterministic point identifiers.

Qdrant point IDs must be a function of the chunk's identity, never of
its position in a list. Positional IDs mean ingesting a second document
silently overwrites the first, and an amended law cannot be updated in
place because nothing identifies its existing points.
"""

import uuid


# Fixed namespace so IDs stay stable across machines and runs.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "pocket-lawyer")


def point_id(chunk_id: str) -> str:
    """Map a semantic chunk id to a stable Qdrant point id."""

    if not chunk_id or not chunk_id.strip():
        raise ValueError("chunk_id cannot be empty.")

    return str(uuid.uuid5(NAMESPACE, chunk_id))
