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

Retrieval → reranking → grounded generation works end to end over **two
documents, 591 chunks**: the Constitution of Kenya, 2010 (279) and the Criminal
Procedure Code, Cap. 75 (312).

**FastAPI now serves it.** Models load once at startup and are reused across
requests, and a semantic router answers conversational input without touching
the pipeline. There is no conversation state, no sessions, no auth, and no
mobile app.

Full status, verified measurements, known blockers, and the roadmap live in
[`docs/project-plan.md`](docs/project-plan.md). Read it before planning work.

## Repo map

```
backend/app/main.py        FastAPI app — lifespan loads models ONCE
backend/app/api/
  routes.py                POST /ask, GET /health; the intent branch point
  schemas.py               request/response models; sources from the payload
backend/app/ai/
  intent.py                IntentClassifier — nearest-centroid routing
  retriever.py             LegalRetriever  — BGE-M3 → Qdrant, top 30
  reranker.py              LegalReranker   — cross-encoder, top 30 → top 5,
                           capped so one document cannot take every slot
  llm.py                   LegalLLM        — gpt-4o-mini + grounding prompt
  rag.py                   LegalRAG.answer() — chains retrieval/rerank/LLM
  generator.py, prompt.py  EMPTY placeholders, unused
backend/app/core/config.py Single source of truth for config - BOTH sides import it
data_pipeline/
  run.py                   orchestrator: extract → clean → structure → chunk → store
  documents.yaml           document registry - add a document by adding an entry
  ir.py                    Document/Unit - the common shape every stage works on
  registry.py              loads the registry, derives artifact paths
  adapters.py              the type registry - register a new document HERE only
  structure/<type>.py      per-TYPE and reusable: constitution.py, act.py
  cleaners/<document>.py   per-DOCUMENT: its patterns name the document itself
  validators/<document>.py per-DOCUMENT, plus generic.py for every document
  chunking/, vectorstore/
data/raw/<type>/           immutable sources, never tracked
data/documents/<id>@v<ver>/  extracted, cleaned, structure, chunks, metadata
tests/                     pytest
docs/project-plan.md       the living plan and progress tracker
mobile/                    empty (Flutter, not started)
```

The pipeline is generic and registry-driven, and adding the Criminal Procedure
Code proved it: `run.py`, `registry.py`, `ir.py`, `chunking/`, `metadata.py`,
`vectorstore/` and the entire `backend/` retrieval path needed **no changes**
for a document with a completely different hierarchy.

What that exercise corrected is the cost estimate. A new document needs:

- **a cleaner spec** (`cleaners/<document>.py`) — always. Its `start_pattern`
  and `running_header_pattern` embed the document's own title, and a
  `CleanerSpec` is static data with no access to the registry entry, so it
  cannot be derived.
- **a structure parser** — only for a genuinely new hierarchy. `act.py` handles
  the Part → Section shape every Kenya Law consolidated Act uses, so the next
  Act reuses it.
- **a validator** (`validators/<document>.py`) — in practice always, because
  without one a parse failure is silent: a broken pattern yields zero units and
  the only complaint is "No chunks were generated."

## Running things

```bash
docker compose up -d              # Qdrant on localhost:6333
pytest                            # 160 tests, no Qdrant or API key needed

python -m uvicorn backend.app.main:app --reload   # the API, models load once
python -m backend.app.ai.intent                   # routing accuracy + margins

python -m data_pipeline.run --document constitution-of-kenya-2010
python -m data_pipeline.run --document criminal-procedure-code
python -m data_pipeline.run --all
python -m data_pipeline.run --document X --from chunk   # resume a stage
python -m data_pipeline.run --document X --to chunk     # stop before Qdrant
python -m data_pipeline.run --document X --recreate     # rebuild collection

python -m evaluation.run             # tier 1: retrieval only, free, no API key
python -m evaluation.run --tier 2    # adds the LLM (refusal correctness)
python -m evaluation.run --accept-last   # record the last run as baseline
python -m evaluation.run --id <qid>  # one question, repeatable, for debugging

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
- **Scope the evaluation while iterating; run it whole before baselining.**
  `--id` is repeatable, and a full run costs real time that grows with both the
  question count and the corpus. Iterate on the questions a change can actually
  affect, then run all of them once before `--accept-last`. Note `--id` runs are
  deliberately excluded from `--save-baseline` and from `last_run.json`.
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
- **One document may not take every reranked slot.** A citizen's question
  usually has its *right* in the Constitution and its *procedure* in an Act.
  With only the Constitution indexed this could not arise; the moment the
  Criminal Procedure Code landed, CPC sections filled every slot for *"Do I
  have to answer police questions after being arrested?"* — s.36A contains
  "inquiries ... by the police" while Article 49 says "the right to remain
  silent" — and the system refused a question the corpus answers. `rerank()`
  now reserves a slot for the best passage from a shut-out document, exempt
  from the tail cut. Structural on purpose: cross-encoder scores are not
  trustworthy enough to tune a threshold against.
- **Routing is semantic, and never enumerated.** The first router matched
  regexes against the raw string and failed the way hand-listed language always
  fails: "what is your name" slipped past a pattern covering "what's your name",
  and "morning" past one requiring "good morning". Each miss cost a retrieval, a
  30-passage rerank and a paid LLM call. `IntentClassifier` now compares the
  query embedding against per-intent centroids — nearest centroid, **never a
  threshold**, for the same reason rerank scores cannot carry one. The regexes
  survive only as a 13-entry exact-match fast path so "hi" skips the embedding;
  **do not grow that list**, anything it misses falls through to the centroids,
  which is the point. Measured 39/39 on routing direction over 40 labelled
  inputs: `python -m backend.app.ai.intent`.
- **Routing errors are asymmetric, and the code leans one way on purpose.** A
  greeting sent to the pipeline wastes seconds and cents. A legal question sent
  to a canned reply tells someone asking about their arrest "Hello! Ask me about
  Kenyan law." So ties, near-ties, embedding failures and a missing classifier
  all resolve to LEGAL. `CONVERSATIONAL_MARGIN` is measured, not chosen — raise
  it if an unseen legal question ever scores positive, never lower it to rescue
  a greeting.
- **Retrieval breadth scales with the corpus.** `RETRIEVAL_TOP_K` was 15 for
  279 chunks; at 591 that put Article 49 out of range entirely for bail
  questions. It is 30 now. Expect to revisit it as the corpus grows, and note
  it doubles the reranking work — the dominant latency.

## Gotchas

- The Hugging Face "unauthenticated requests" rate-limit warning on model load is
  benign and does not block anything.
- BGE-M3 + reranker occupy ~2.3–2.5 GB RAM and dominate startup time.
- CPU cross-encoder reranking is the dominant request latency — not the OpenAI
  call.
- Priority when trading off: **accuracy > grounding > usefulness > speed > cost**.
