"""
Two-level result cache for the evaluation harness.

Retrieval and reranking are deterministic. Same question, same candidates,
same models, same code, same parameters - same result, every time. Most runs
recompute an identical answer, and reranking is where the hours go: thirty
cross-encoder forward passes per question, on CPU, over passages up to four
thousand characters.

**The levels are keyed differently, and that difference is the point.**

Reranking's inputs are the question, the candidate passages, the model and the
code. The CORPUS IS NOT AN INPUT - only the candidates that came out of it are.
So the level 2 key contains no corpus fingerprint:

    level 1  retrieval   keyed on the corpus    -> a new document invalidates all
    level 2  reranking   keyed on the CANDIDATES -> survives unless they moved

That is what makes adding a document affordable. When the Criminal Procedure
Code landed it reshaped the arrest questions and put nothing at all into the
top-30 for `dual-citizenship` or `land-types` - yet every one of those was
reranked from scratch. If the candidate list is byte-identical the rerank
result is provably identical: same inputs, same model, same code.

**What makes this safe rather than dangerous.** A stale cache hides a
regression, which is the failure this harness exists to prevent. So the keys
include a hash of the SOURCE of the modules that decide a result. Editing the
reranker's diversity rule changes output while corpus and config stay
identical; without that hash the cache would confidently serve the old answer.

It cannot see transitive changes - a sentence-transformers upgrade, say. Use
`--no-cache` for those, and note that a baseline is never served from cache.

Tier 2 is never cached at all: the LLM is non-deterministic, so a cached answer
would be reporting something the system did not produce.
"""

import hashlib
import json
from pathlib import Path

from backend.app.core import config


CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# The modules whose source decides a result. Retrieval depends on the first
# two; reranking on the last two.
RETRIEVAL_SOURCES = ("retriever.py", "passage.py")
RERANK_SOURCES = ("reranker.py", "passage.py")

_AI_DIR = Path(config.__file__).resolve().parent.parent / "ai"
_CORE_DIR = Path(config.__file__).resolve().parent


def _source_hash(filenames: tuple[str, ...]) -> str:
    digest = hashlib.sha256()

    for name in sorted(filenames):
        for directory in (_AI_DIR, _CORE_DIR):
            path = directory / name

            if path.exists():
                digest.update(path.read_bytes())
                break

    return digest.hexdigest()[:16]


def corpus_fingerprint(documents_dir: Path | None = None) -> str:
    """
    What is currently ingested, by content rather than by name.

    Uses the cleaned-text checksum from each document's metadata, so a
    re-clean that changes text without changing chunk ids still invalidates.
    """

    directory = documents_dir or config.DOCUMENTS_DIR

    entries = []

    for path in sorted(directory.glob("*/metadata.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):
            entries.append((path.parent.name, "unreadable"))
            continue

        checksums = data.get("checksums") or {}

        entries.append(
            (
                path.parent.name,
                checksums.get("cleaned") or checksums.get("raw") or "",
                (data.get("processing") or {}).get("chunk_count"),
            )
        )

    return _digest(entries)


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def retrieval_key(question: str, fingerprint: str) -> str:
    """Level 1. The corpus belongs here, and only here."""

    return _digest(
        [
            "retrieval-v1",
            question,
            fingerprint,
            config.EMBEDDING_MODEL,
            config.COLLECTION_NAME,
            config.RETRIEVAL_TOP_K,
            _source_hash(RETRIEVAL_SOURCES),
        ]
    )


def rerank_key(question: str, chunk_ids: list[str]) -> str:
    """
    Level 2. Keyed on the candidates, deliberately NOT on the corpus.

    `chunk_ids` is the retrieved list in order. Two corpora that hand this
    question the same candidates must rerank to the same answer, so there is
    no reason to recompute across a document that did not reach it.
    """

    return _digest(
        [
            "rerank-v1",
            question,
            chunk_ids,
            config.RERANKER_MODEL,
            config.RERANK_TOP_K,
            config.RERANK_RELATIVE_RATIO,
            _source_hash(RERANK_SOURCES),
        ]
    )


class ResultCache:
    """
    File-backed, one JSON per key.

    Counts hits and misses because silent caching is how a cache becomes
    untrustworthy: every run reports what it reused versus computed, and a
    run with any hits may not become a baseline.
    """

    def __init__(
        self,
        directory: Path | None = None,
        read: bool = True,
        write: bool = True,
    ):
        """
        Reading and writing are separate on purpose.

        A reference run - `--no-cache`, or one destined to become a baseline
        - must ignore what is stored and recompute. But the values it
        computes are correct by definition, so throwing them away would mean
        paying for a second full run to warm the cache. It reads nothing and
        writes everything.
        """

        self.directory = directory or CACHE_DIR
        self.read = read
        self.write = write

        self.hits = {"retrieval": 0, "rerank": 0}
        self.misses = {"retrieval": 0, "rerank": 0}

    @property
    def used(self) -> bool:
        """Whether any result was recalled rather than computed."""

        return any(self.hits.values())

    def get(self, level: str, key: str):
        if not self.read:
            # Counted as a miss so the run reports what it actually
            # computed rather than staying silent.
            self.misses[level] += 1
            return None

        path = self.directory / level / f"{key}.json"

        if not path.exists():
            self.misses[level] += 1
            return None

        try:
            value = json.loads(path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):
            # A corrupt entry is a miss, never an error. The cost of being
            # wrong here is recomputation, which is exactly what we want.
            self.misses[level] += 1
            return None

        self.hits[level] += 1

        return value

    def put(self, level: str, key: str, value) -> None:
        if not self.write:
            return

        path = self.directory / level / f"{key}.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            )

        except OSError:
            # Failing to cache must never fail a run.
            pass

    def summary(self) -> dict:
        return {
            "read": self.read,
            "write": self.write,
            "hits": dict(self.hits),
            "misses": dict(self.misses),
        }
