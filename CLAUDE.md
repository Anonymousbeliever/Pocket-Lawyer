# Pocket Lawyer Kenya — Agent Context

AI legal information assistant for Kenyan law. RAG over authoritative Kenyan
legal sources, with citations.

## The one non-negotiable principle

**Retrieve the law. Do not recall it.** The knowledge base is the authority; the
LLM only explains what was retrieved.

When the retrieved sources are insufficient, the system says so and points the
user to a qualified advocate. **That refusal is a feature, not a bug.** Never
"fix" a refusal by loosening the grounding prompt or letting the model fall back
on its own memory of Kenyan law. The fix for a refusal is always a broader
corpus or better retrieval.

Never invent statutes, sections, articles, cases, procedures, deadlines,
penalties, or citations. Never claim a source says something it does not.

## Current state

Retrieval → reranking → grounded generation works end to end, but **only the
Constitution of Kenya, 2010 is indexed** (270 chunks). FastAPI is not yet wired
to the RAG pipeline; there is no mobile app.

Full status, verified measurements, known blockers, and the roadmap live in
[`docs/project-plan.md`](docs/project-plan.md). Read it before planning work.

## Repo map

```
backend/app/main.py        FastAPI — only GET / so far, NOT connected to RAG
backend/app/ai/
  retriever.py             LegalRetriever  — BGE-M3 → Qdrant, top 15
  reranker.py              LegalReranker   — cross-encoder, top 15 → top 5
  llm.py                   LegalLLM        — gpt-4o-mini + grounding prompt
  rag.py                   LegalRAG.answer() — chains all three
  generator.py, prompt.py  EMPTY placeholders, unused
backend/app/core/config.py Single source of truth for config - BOTH sides import it
data_pipeline/
  run.py                   orchestrator: extract → clean → structure → chunk → store
  documents.yaml           document registry - add a document by adding an entry
  ir.py                    Document/Unit - the common shape every stage works on
  registry.py              loads the registry, derives artifact paths
  adapters.py              the type registry - register a new type HERE only
  structure/<type>.py      the ONLY type-specific stage; the adapter seam
  cleaners/, chunking/, validators/, vectorstore/
data/raw/<type>/           immutable sources, never tracked
data/documents/<id>@v<ver>/  extracted, cleaned, structure, chunks, metadata
tests/                     pytest
docs/project-plan.md       the living plan and progress tracker
mobile/                    empty (Flutter, not started)
```

The pipeline is generic and registry-driven. Adding a document of a type that
already has adapters means **one entry in `documents.yaml` and no new code**. A
genuinely new type (an Act, a judgment) needs one new `structure/` adapter that
emits the IR — nothing downstream changes.

## Running things

```bash
docker compose up -d              # Qdrant on localhost:6333
pytest                            # 33 tests, no Qdrant needed

python -m data_pipeline.run --document constitution-of-kenya-2010
python -m data_pipeline.run --all
python -m data_pipeline.run --document X --from chunk   # resume a stage
python -m data_pipeline.run --document X --recreate     # rebuild collection

python -m evaluation.run             # tier 1: retrieval only, free, no API key
python -m evaluation.run --tier 2    # adds the LLM (refusal correctness)
python -m evaluation.run --accept-last   # record the last run as baseline
python -m evaluation.run --id <qid>  # one question, for debugging

python -m backend.app.ai.rag         # full pipeline, interactive CLI
python -m backend.app.ai.retriever   # retrieval only
python -m backend.app.ai.reranker    # reranking only
python -m backend.app.ai.llm         # generation only
```

Ingestion is idempotent — point IDs are `uuid5(chunk_id)`, so re-running
updates in place rather than duplicating. `--recreate` is only needed when the
collection's vector schema changes.

**Identity is version-aware**: `chunk_id = {document_id}@v{version}-{slug}`.
Two editions of the same Act are two registry entries sharing a `document_id`,
differing in `version`, and they never overwrite each other. Exactly one may be
`in_force`; retrieval filters on that by default.

**Always use `python -m ...`, never `python backend/app/ai/rag.py`** — the
latter fails with `ModuleNotFoundError: No module named 'backend'` because these
modules use package imports. Run from the repo root.

Every AI module doubles as its own CLI test harness.

Environment: Windows, PowerShell, `.venv`, Python 3.14. Qdrant collection is
`pocket_lawyer_legal`.

## Pinned technical choices

| Role | Choice |
|------|--------|
| Embeddings | `BAAI/bge-m3` — 1024-dim, cosine, normalized, 8192 token limit |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| Generation | `gpt-4o-mini` via the OpenAI **Responses** API (`client.responses.create`) |
| Vector store | Qdrant (Docker, local) |
| Backend | FastAPI |
| Mobile (later) | Flutter |

**Do not silently change models.** The embedding model must be byte-identical at
ingest time and query time — if they drift apart, retrieval quality collapses
**with no error message**. `gpt-4o-mini` is what the project's API credentials
are provisioned for.

## Working rules

- **Don't casually rewrite** `retriever.py`, `reranker.py`, `llm.py`, `rag.py`.
  They work and have been tested. Change them for a concrete reason.
- **Secrets only via `.env`.** Never put `OPENAI_API_KEY`, `HF_TOKEN`, or
  database credentials in source. `.env` is gitignored; `.env.example` is not.
- **Prefer structured source metadata** over free-text. The backend holds
  authoritative citation data — don't rely on the LLM to format a sources list.
- **After any AI change, run the evaluation** (`python -m evaluation.run`) and
  check it against the baseline. Tier 1 is free and catches retrieval and
  reranking regressions; a single manual question does not.
- **Embed chunks with their citation context**, never the bare body. An
  article's topic often lives only in its title - Article 16 is "Dual
  citizenship" and its text never says "dual". Embedding content alone made it
  rank #27; with the header it ranks #2. `backend/app/core/passage.py` builds
  that text for the embedder and reranker; the LLM still receives raw content.
- **Design for concurrent users**, not one CLI process per question. Models must
  eventually load once at startup, not per request.
- **Avoid new dependencies** unless they solve a real problem. Don't add a module
  until it has a clear responsibility.
- **Never gate sufficiency on an absolute rerank score.** It was tried and
  removed. Cross-encoder scores rank candidates *within* one query; they are not
  comparable *across* queries. A typo moved the correct source from 0.1344 to
  0.0086 while it still ranked first — below the 0.0373 scored by a question the
  corpus cannot answer. No fixed floor works. Deciding whether sources answer
  the question is the LLM's job under its grounding prompt. Full measurements
  are in `docs/project-plan.md`.

## Gotchas

- The Hugging Face "unauthenticated requests" rate-limit warning on model load is
  benign and does not block anything.
- BGE-M3 + reranker occupy ~2.3–2.5 GB RAM and dominate startup time.
- CPU cross-encoder reranking is the dominant request latency — not the OpenAI
  call.
- Priority when trading off: **accuracy > grounding > usefulness > speed > cost**.
