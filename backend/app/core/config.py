"""
Single source of truth for Pocket Lawyer configuration.

Both the ingestion pipeline (`data_pipeline`) and the AI layer
(`backend.app.ai`) import from here.

This matters more than it looks. The embedding model used to build the
vectors and the embedding model used to embed a user's question must be
identical. If they ever drift apart, retrieval quality collapses and
nothing raises an error — the system simply returns worse law.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

# backend/app/core/config.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

# Immutable source files, grouped by document type. Never regenerated,
# never tracked.
RAW_DIR = DATA_DIR / "raw"

# Derived artifacts, one directory per document version:
#   data/documents/<document_id>@v<version>/{extracted,cleaned,...}
#
# Document-first rather than stage-first: the pipeline stages live in
# the code, so encoding them in folder names only scatters a single
# document's files across five directories.
DOCUMENTS_DIR = DATA_DIR / "documents"


load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def _get(name: str, default: str) -> str:
    value = os.getenv(name)

    # Treat an empty env var the same as an unset one. The committed
    # .env.example ships blank placeholders.
    if value is None or not value.strip():
        return default

    return value.strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name, str(default))

    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(
            f"{name} must be an integer, got: {raw!r}"
        )


def _get_float(name: str, default: float) -> float:
    raw = _get(name, str(default))

    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(
            f"{name} must be a number, got: {raw!r}"
        )


# ---------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------

QDRANT_URL = _get("QDRANT_URL", "http://localhost:6333")

COLLECTION_NAME = _get("COLLECTION_NAME", "pocket_lawyer_legal")

# Named vectors. Declaring the sparse slot now means adding hybrid
# search later does not require a second full re-ingest.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-m3")

EMBEDDING_DIM = _get_int("EMBEDDING_DIM", 1024)

RERANKER_MODEL = _get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

# The candidate pool has to scale with the corpus. At 15 it was tuned
# against the Constitution alone (279 chunks). Indexing the Criminal
# Procedure Code took the corpus to 591, and "Can I be released on bail
# while waiting for my trial?" stopped retrieving Article 49 at all —
# it sat between rank 16 and 30, crowded out by Criminal Procedure Code
# bail sections. At 30 it returns, and the reranker then puts it first.
#
# Measured over the full 56-question set, 15 -> 30 restored retrieval to
# 41/41 and made no other question worse. The cost is real: reranking is
# the dominant request latency and this doubles the passages scored.
# Accuracy outranks speed here.
RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", 30)

RERANK_TOP_K = _get_int("RERANK_TOP_K", 5)

# Tail trimming for reranked sources.
#
# A result is kept when it scores at least this fraction of the best
# match for the SAME query. The top result always survives, so this
# can never cause a refusal on its own.
#
# There is deliberately NO absolute score floor. Cross-encoder scores
# rank candidates within one query; they are not a confidence measure
# comparable across queries. Measured against this corpus, the two are
# not merely noisy but inverted:
#
#   "Can police arrest me without telling me why?"  Art 49 -> 0.1344
#   "Can police arrest me without teling me why?"   Art 49 -> 0.0086
#   "...divorce in Kenya?" (answer not in corpus)   best   -> 0.0373
#
# One missing letter moved the correct answer 15x, to below the best
# wrong answer for a question the corpus cannot answer. No fixed floor
# passes the second and rejects the third. Article 49 ranked first in
# all three — only the magnitude moved.
#
# Deciding whether the sources actually answer the question is the
# LLM's job, under the grounding prompt in llm.py, which did it
# correctly before any threshold existed.
RERANK_RELATIVE_RATIO = _get_float("RERANK_RELATIVE_RATIO", 0.10)


# ---------------------------------------------------------
# INGESTION
# ---------------------------------------------------------

EMBED_BATCH_SIZE = _get_int("EMBED_BATCH_SIZE", 8)

UPSERT_BATCH_SIZE = _get_int("UPSERT_BATCH_SIZE", 128)

# Units longer than this are split into sub-chunks on paragraph
# boundaries. One vector for 24,000 characters of unrelated schedule
# text is semantically meaningless.
MAX_CHUNK_CHARS = _get_int("MAX_CHUNK_CHARS", 4000)
