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


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name, str(default)).strip().lower()

    if raw in ("1", "true", "yes", "on"):
        return True

    if raw in ("0", "false", "no", "off"):
        return False

    raise RuntimeError(
        f"{name} must be true or false, got: {raw!r}"
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
# Fuse dense and lexical retrieval rather than dense alone.
#
# Dense encodes meaning and handles paraphrase; it blurs exact terms. Asked
# "What is the sentence for murder in Kenya?", section 204 - eleven words
# answering it almost verbatim - came back at rank 53, beaten by the longer
# "Conspiracy to murder". Lexical matching on "murder" and "sentence" is
# what rescues that, and section numbers and defined terms are the same
# problem.
#
# Turn off to measure against dense-only. The sparse vectors stay in the
# collection either way; only the query changes.
HYBRID_RETRIEVAL = _get_bool("HYBRID_RETRIEVAL", True)

RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", 30)

# How many candidates the LEXICAL branch contributes to fusion, against
# RETRIEVAL_TOP_K from the dense branch.
#
# Deliberately asymmetric. Dense carried 66/67 on its own; sparse exists to
# rescue the cases where an eleven-word provision loses to mere vocabulary
# overlap. Given an equal budget it does the opposite of helping: asked
# "Can I be released on bail while waiting for my trial?", the Criminal
# Procedure Code's many sections containing the literal term "bail" filled
# the list and pushed Article 49 - the constitutional right - out of range
# entirely. Reciprocal Rank Fusion compounds this, because it rewards
# appearing in BOTH lists and Article 49 is a dense-only hit for that query.
#
# A smaller lexical budget means fewer sparse-only candidates can crowd out
# a dense-only one, while a strong lexical match still arrives near the top
# of its own short list.
#
# MEASURED, not chosen. Swept over all 82 questions at tier 0:
#
#      5   67/67   3 improved, 0 regressed
#     10   67/67   3 improved, 0 regressed
#     20   66/67   arrest-bail lost - Article 49 pushed out by CPC bail
#     30   66/67   same, equal budgets
#
# Safe at <= 10, broken at >= 20; the transition between them is not
# measured. 5 and 10 are indistinguishable on the evidence, so this takes
# the one with margin - the same reasoning as CONVERSATIONAL_MARGIN, which
# sits in open space between clusters rather than at the edge of one.
#
# Erring low is also the safer direction: too high displaces constitutional
# rights with statutory detail, while too low merely fails to rescue a
# lexical match, which is the behaviour before hybrid existed. Expect to
# revisit as documents are added, since each one changes how crowded the
# lexical list gets.
SPARSE_TOP_K = _get_int("SPARSE_TOP_K", 5)

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
