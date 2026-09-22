# Pocket Lawyer Kenya — Master Project Plan

This is the living project plan and progress tracker.

Update this file when a task starts, finishes, or the current position changes. Source inventory lives in [`legal-data/sources.md`](legal-data/sources.md).

**Last updated:** 2026-09-20

## How to use this document

| Mark | Meaning |
|------|---------|
| `[ ]` | Not started |
| `[x]` | Done |

Each phase also has a roll-up status: **Not started**, **In progress**, or **Done**.

A parent task can stay unchecked while a note records partial work (for example, Constitution only). Do not mark a phase done until its objective is met.

---

## Vision

Pocket Lawyer Kenya is an open-source, AI-powered legal information assistant designed to make Kenyan legal knowledge accessible to ordinary citizens.

The goal is not to replace lawyers or provide unauthorized legal representation. The goal is to make authoritative Kenyan legal information understandable, searchable, and accessible.

A user should be able to ask something like:

> Police have arrested me and I don't know why. What are my rights?

The system should retrieve the relevant Kenyan legal provisions, court decisions, or other authoritative sources and provide a clear answer with citations showing exactly where the information came from.

**The AI should not guess the law. It should retrieve the law, reason over it, and show the user the evidence.**

---

## What we are not building

Pocket Lawyer is not intended to:

- Replace advocates
- Represent users in court
- Make decisions on behalf of users
- Invent legal authorities
- Present uncertain information as fact
- Hide its sources
- Treat random internet pages as equivalent to primary law

When the system does not have enough reliable information, the desired behavior is:

> I don't have enough reliable information to answer this confidently. Please consult a qualified advocate.

That behavior is a feature. It has been tested and works (see Phase 3 notes).

---

## Architecture

The project is built from the data foundation upward, not from a chatbot UI.

```text
                 OFFICIAL KENYAN SOURCES
                         │
                         ▼
                  DATA COLLECTION
                         │
                         ▼
                  DATA PIPELINE
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           Extract      OCR        Clean
             │           │           │
             └───────────┼───────────┘
                         ▼
                    Structure
                         │
                         ▼
                      Chunk
                         │
                         ▼
                    Embeddings
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         PostgreSQL               Qdrant
         Metadata                Vector Search
              │                     │
              └──────────┬──────────┘
                         ▼
                    RAG Pipeline
                         │
                   ┌─────┴─────┐
                   ▼           ▼
              Retrieval      Reranking
                   │           │
                   └─────┬─────┘
                         ▼
                       LLM
                         │
                         ▼
                Citation Verification
                         │
                         ▼
                    FastAPI API
                         │
                         ▼
                    Flutter App
                         │
                         ▼
                       USER
```

### Important distinctions

- **FastAPI is not the AI brain.** It is the backend/API framework. Retrieval, Qdrant, reranking, prompt orchestration, and the LLM together form the AI system.
- **n8n is not the document processor.** It may later orchestrate collection. The Python pipeline remains the core engine.

### The retrieval principle

Pocket Lawyer should never work like this:

```text
User → LLM → "Here's what Kenyan law says..."
```

That is asking an LLM to remember the law.

It should work like this:

```text
User
 ↓
Question
 ↓
Retrieve Kenyan legal sources
 ↓
Relevant legal passages
 ↓
Rerank
 ↓
LLM
 ↓
Citations
 ↓
Verified answer
```

This is now implemented end-to-end for the Constitution.

---

## Current position

**Three documents, served over HTTP, with an evaluation harness that scales.**

```text
Question
   ↓
Intent routing (nearest centroid) ──→ conversational, no retrieval
   ↓ legal
BGE-M3 embedding
   ↓
Qdrant (962 vectors: Constitution 279, Criminal Procedure Code 312,
        Penal Code 371)
   ↓
Dense (top 30) + lexical (top 5), fused with RRF
   ↓
Top 30 candidates
   ↓
BGE reranker, no single document taking every slot
   ↓
Top 5 sources
   ↓
gpt-4o-mini (strict grounding prompt)
   ↓
Grounded answer + verified citations
```

`python -m uvicorn backend.app.main:app` serves it, loading models once.
`python -m backend.app.ai.rag` is still the CLI.

The generic pipeline has now absorbed three hierarchies — Chapter → Article,
Part → Section, and Part → Chapter → Section — without the orchestrator,
registry, IR, chunker or vector store changing once. Per document the cost is a
cleaner spec, a validator, and a structure adapter only for a genuinely new
shape.

**Current state:** 82 questions, **80/82**, with **retrieval at 67/67 — 100%**.

**What limits the product now:**

1. **Reranking, not retrieval.** Both remaining failures are now
   `RETRIEVED THEN DROPPED`: the cross-encoder has the right passage in its
   candidate list and discards it. Section 204 sits at rank 8 after hybrid
   pulled it from 53. The same cross-encoder scored a correct answer at
   0.042. It is the single weak link in the chain.
2. **Corpus breadth.** Three documents is a start, not a legal corpus.
3. **The application layer.** No conversation state, no persistence, no auth,
   no mobile app.

Still open and recorded: `debt-inability`'s false answer, the
`delete_document` version bug, and `county-government-role` awaiting a
lawyer's view.

---

## Phase 0 — Project Foundation

**Status:** Done (minimal)

**Objective:** Create a clean, reproducible development environment and establish the project's basic architecture.

- [x] Project structure
- [x] Python environment
- [x] FastAPI application
- [x] Git configuration
- [x] Dependency management
- [x] Documentation (initial)

The backend is still a minimal FastAPI app (`backend/app/main.py`) exposing only `GET /` — it is **not** yet connected to the RAG pipeline. Flutter is an empty `mobile/` folder.

---

## Phase 1 — Legal Data Foundation

**Status:** In progress — source inventory outstanding

**Objective:** Build the trusted legal corpus that will eventually become the foundation of the AI.

- [x] Identify authoritative sources
- [ ] Build source inventory
- [x] Collect initial corpus
- [x] Extract document text
- [x] Clean extracted text
- [x] Process document metadata
- [x] Legal structure detection
- [x] Build ingestion pipeline

### Notes

- Source hierarchy and collection principles: [`legal-data/sources.md`](legal-data/sources.md).
- Initial corpus: Constitution of Kenya (`data/raw/constitution/Constitution of Kenya.pdf`).
- Extractor: `data-pipeline/extractors/pdf_extractor.py` → `data/extracted/constitution.txt` (PyMuPDF).
- Cleaner: `data-pipeline/cleaners/constitution_cleaner.py` → `data/cleaned/constitution_cleaned.txt`.
- Validator: `data-pipeline/validators/constitution_validator.py` — checks required markers, the full article sequence 1–264, forbidden leftovers (page markers, TOC, ligatures), and structure. It initially failed, was fixed, and now passes.
- Metadata: `data-pipeline/pipelines/metadata_builder.py` → `data/metadata/constitution.json`, including SHA-256 checksums of the raw, extracted, and cleaned files.
- Structure: `data-pipeline/structure/constitution_structure.py` → `data/processed/constitution_structure.json` (chapters → articles → subsections → paragraphs → schedules).
- `collectors/` and `ocr/` are still empty. Everything else in `data-pipeline/` is implemented.
- **Remaining item — "Build ingestion pipeline":** the stages all exist but are separate, Constitution-specific scripts run by hand. The orchestrated, document-agnostic pipeline is the next piece of work.
- Kenya Law, Judiciary, and Supreme Court are investigated; Parliament is not yet.

### Target pipeline

```text
Official source
      ↓
Collector
      ↓
Download
      ↓
Validation
      ↓
Raw document
      ↓
PDF/HTML extraction
      ↓
OCR if necessary
      ↓
Cleaning
      ↓
Metadata extraction
      ↓
Legal structure detection
      ↓
Chunking
      ↓
Embeddings
      ↓
Knowledge base
```

Every stage above has been proven against the Constitution. What does not exist yet is a single entrypoint that runs them generically for any registered document.

---

## Phase 2 — Legal Knowledge Base

**Status:** Done and proven on two documents — corpus breadth outstanding

**Objective:** Turn the processed legal corpus into a searchable knowledge base.

> The subsection below describes the original single-document build and its
> then-current file paths. Both the layout and the numbers have since changed
> (`data-pipeline/` → `data_pipeline/`, 270 → 279 Constitution chunks, 591 in
> the collection). It is kept as the record of what was built when. The
> progress log carries the current state.

- [ ] PostgreSQL
- [x] Qdrant
- [x] Embeddings
- [x] Legal document chunking
- [x] Retrieval

### What was actually built

- **Qdrant** runs locally via `docker-compose.yml` on `localhost:6333`. Collection: `pocket_lawyer_legal`, 1024 dimensions, cosine distance.
- **Chunking** (`data-pipeline/chunkers/constitution_chunker.py`) produces **270 chunks**: 264 article chunks (articles 1–264, none missing) plus 6 schedule chunks. Chunks are article-level rather than fixed-size windows, so each chunk is a natural citation unit.
- **Embeddings** (`data-pipeline/embeddings/constitution_embeddings.py`) use `BAAI/bge-m3`, normalized, 1024-dim.
- **Vector store** (`data-pipeline/vectorstore/qdrant_store.py`) creates the collection, upserts points with legal metadata in the payload, and verifies the stored count.
- **Retrieval** (`backend/app/ai/retriever.py`) embeds the question with the same model and returns the top 15 payloads.

### PostgreSQL — still to do

Structured document metadata currently lives in flat JSON (`data/metadata/`) and inside the Qdrant payload. That is adequate for one document and will not stay adequate.

When Postgres arrives it should be the **ingestion ledger and document registry** — what has been ingested, at which version, with which checksum and processing status — and later the home for users, conversations, and feedback. It should *not* become a second copy of chunk metadata:

```text
Document
├── title
├── document type
├── source
├── publication date
├── effective date
├── version
├── URL
├── checksum
└── processing status
```

### Why vector search

Meaning matches even when wording differs. A user asking *"Can cops arrest me without explaining why?"* reaches text reading *"Every arrested person has the right to be informed promptly..."*. This was verified in testing.

---

## Phase 3 — AI / RAG

**Status:** In progress — **current phase**

**Objective:** Build the AI that answers questions using the legal knowledge base.

- [x] First RAG pipeline
- [x] Reranking
- [x] Citation generation
- [ ] Citation verification — *identity half done, entailment outstanding*
- [x] Evaluation — *71-question set, baseline recorded; question set awaiting legal review*

### What was actually built

- **Retriever** (`backend/app/ai/retriever.py`) — `LegalRetriever`, BGE-M3, top 15 from Qdrant.
- **Reranker** (`backend/app/ai/reranker.py`) — `LegalReranker`, `BAAI/bge-reranker-v2-m3` cross-encoder, top 15 → top 5.
- **LLM** (`backend/app/ai/llm.py`) — `LegalLLM`, `gpt-4o-mini` via the OpenAI **Responses** API, with a strict grounding system prompt.
- **Orchestration** (`backend/app/ai/rag.py`) — `LegalRAG.answer()` chains retrieval → reranking → generation and returns `{answer, sources}`.

Each module also runs standalone as its own CLI test harness.

### Verified test results

| Test | Result |
|------|--------|
| Retrieval: *"What are the rights of an arrested person?"* | Returned Article 49 first, then 51, 29, 37, 53 |
| Reranking: same question | Article 49 → 0.9993, Article 51 → 0.9392, Article 29 → 0.2933 |
| Generation: supported question | Correctly answered from Article 49 with citations |
| Generation: *"What is the legal process for filing for divorce in Kenya?"* | Correctly **refused** — corpus lacks divorce law |
| Full RAG: *"Can police arrest me without telling me why?"* | Grounded answer citing Article 49 |

The refusal is desired behaviour. **Do not weaken the prompt to make the model answer more often.** The fix for a refusal is a broader corpus, not looser grounding.

### Target flow

```text
USER QUESTION
      ↓
Query processing
      ↓
Embedding
      ↓
Qdrant retrieval
      ↓
Relevant legal chunks
      ↓
Reranking
      ↓
Best sources
      ↓
LLM
      ↓
Answer
      ↓
Citation verification
      ↓
USER
```

### Response-quality issues

Both reproduced on 2026-08-31 after the ingestion refactor. Neither was a
retrieval or grounding failure — the correct article was retrieved and cited
both times. They are *response layer* problems.

**1. Question polarity — FIXED.** For *"Can police arrest me without telling me
why?"* the system answered *"Yes, the police cannot arrest you without telling
you why."* — self-contradictory. It addressed the embedded proposition rather
than the yes/no question.

Polar questions with a negative framing require the model to extract the rule,
map it to yes/no, then *invert*. Free-form generation gave it no reason to
commit to a polarity before writing prose.

Fixed by prevention rather than detection: `verdict` is now a discrete enum the
model must fill, and the opening word is written by `render()` in code. The
stated verdict and the explanation can no longer disagree. A second verification
call was considered and rejected — it detects after the fact and doubles latency
and cost, for a problem that can be designed out.

Verified: the same question now answers **"No. An arrested person has the right
to be informed promptly of the reason for the arrest..."**

**2. Claim fidelity — STILL OPEN.** The original answer concluded *"it's legally
required for the police to inform you of the reason for your arrest **at the
time of arrest**"*. Article 49(1)(a) says **"promptly"**. The model strengthened
the law beyond its source; those are different legal standards.

Prompt guidance was added instructing the model to preserve legally-weighted
wording, and the drift did not recur on retest — "promptly" survived. **That is
one sample, not a fix.** Nothing structurally prevents it: a free-text
`explanation` can always overstate.

A residual instance remains even in the improved answer, which says *"police
cannot arrest someone without telling them why"* — Article 49(1)(a) actually
contemplates arrest followed by *prompt* notification, so the timing is
compressed. That nuance belongs in `qualifications`, which came back empty.

*Real fix, because schemas cannot prevent it:* span-grounded citation (require
the model to quote the words it relies on) plus entailment checking of each
claim against its cited passage. That is the outstanding half of Citation
verification.

### Answer schema

Implemented in `backend/app/ai/answer.py`. Recorded here because FastAPI and
Flutter both inherit it as a contract.

```json
{
  "sufficient":      true,
  "question_type":   "polar",
  "verdict":         "no",
  "explanation":     "...",
  "qualifications":  ["..."],
  "cited_chunk_ids": ["constitution-of-kenya-2010-chapter-four-article-49"]
}
```

Beyond fixing polarity this earned three things:

- **The refusal stopped being a magic string.** `sufficient: false` is a field
  to branch on, and the refusal text is defined once rather than duplicated
  between `llm.py` and `rag.py`. It also now carries a reason: the divorce
  question returns *"The retrieved sources do not contain any information about
  the legal process for filing for divorce in Kenya..."* instead of a bare
  decline.
- **Deterministic citation verification, identity half.** Code checks every
  `cited_chunk_ids` entry against the chunks actually retrieved, so a fabricated
  citation is caught with no LLM call and no cost. It does not verify that the
  claim matches the text — that still needs entailment checking — but invented
  citations are now structurally impossible.
- **Cited and considered sources are separated.** Previously a refusal still
  listed all five reranked chunks under "SOURCES USED", presenting Article 260
  and three Sixth Schedule fragments as if they supported the answer. `sources`
  now holds only what the answer cited; `considered` holds everything that
  reached the LLM.

---

## Phase 4 — Application Backend

**Status:** In progress — the service and routing exist; persistence and
conversation do not

**Objective:** Turn the RAG system into a proper application backend.

- [ ] Authentication
- [ ] Conversations
- [ ] Chat API
- [ ] Source API
- [ ] Feedback

Exact API design is deferred. Illustrative endpoints only:

```text
POST /auth/login
POST /chat
GET  /conversations
GET  /sources/{id}
POST /feedback
```

```text
Flutter
   │
   │ POST /chat
   ▼
FastAPI
   │
   ▼
RAG system
   │
   ├── Qdrant
   ├── PostgreSQL
   └── LLM
   │
   ▼
Answer + citations
   │
   ▼
Flutter
```

Two requirements are already known:

- **Load models once at startup.** The current CLI loads BGE-M3, the reranker, and the OpenAI client on every run. A production API must not do that per request.
- **Return structured sources.** The backend already holds authoritative source metadata. Do not depend on the LLM to format a Sources section; the LLM may reference Article numbers in prose, but the citation data should come from the payload.

---

## Phase 5 — Mobile Application

**Status:** Not started

**Objective:** Build the user-facing Pocket Lawyer application.

**Technology:** Flutter

- [ ] Flutter UI
- [ ] Chat interface
- [ ] Sources / citations
- [ ] Conversation history
- [ ] Voice interface

The UI comes after the intelligence underneath it works. The initial interface can follow a familiar chat pattern because users already understand that model. Screen design is deferred.

---

## Known limitations and blockers

Recorded 2026-08-31 after a full review of the codebase and generated data.

### Ingestion blockers — must be fixed before the corpus grows

| # | Issue | Consequence |
|---|-------|-------------|
| 1 | `qdrant_store.py` assigns point IDs from `enumerate` (`PointStruct(id=index)`) | Ingesting a second document **overwrites the Constitution's vectors**. No stable identity means amendments cannot update a document in place. |
| 2 | Every stage is a hardcoded `constitution_*.py` script (5 of them) | 50 Acts would mean 250 scripts. |
| 3 | Embeddings written as one 6.3 MB JSON file, **tracked in git**, loaded whole, upserted unbatched | Repo bloat and full-file diffs on every re-embed; will not scale in memory or request size. |
| 4 | `EMBEDDING_MODEL` / `QDRANT_URL` / `COLLECTION_NAME` duplicated across four files while `.env`'s `QDRANT_URL` is unused | If the ingest-time and query-time embedding models ever drift apart, retrieval quality collapses **with no error**. |

`data-pipeline/` also cannot be imported as a Python package because of the hyphen — which is *why* each stage duplicates config instead of sharing it.

### Secondary gaps

- Sufficiency is judged only by the LLM's grounding prompt. `rerank()` trims the weak tail relative to each query's own best match, but deliberately applies **no absolute score floor** — see *Measured: rerank scores are not a confidence signal* below.
- No metadata filtering at query time, despite the payload carrying `document_type` and `jurisdiction`.
- Schedule 6 is a single **24,146-character** chunk (Schedules 3 and 4 exceed 5,000). Not truncated — BGE-M3 accepts 8192 tokens — but one vector for that much unrelated text is semantically diluted and expensive to rerank and to put in context.
- ~~No `version` / `effective_date` / `in_force` in the Qdrant payload.~~ **Closed 2026-09-02.** The payload carries `version`, `effective_from`, `effective_to`, `in_force` and `as_at`, and identity is version-aware (`document_id@vversion`), so two editions of the same Act coexist rather than overwriting each other. Retrieval filters `in_force == true` by default.
- Sub-paragraph hierarchy is flattened: `(i)`/`(ii)`/`(iii)` collapse into the parent paragraph. Text is intact and faithful, but pinpoint citation below the paragraph level is not addressable. The Act adapter inherits the same limitation.
- ~~No automated tests anywhere; `pytest` is not a dependency.~~ **Closed.** 122 tests, no Qdrant or API key needed.
- ~~`.env.example` still lists a non-existent `OPENAI_MODEL=gpt-5.6-luna`.~~ **Closed.**
- `delete_document` in `vectorstore/ingest.py` filters on `document_id` alone while identity is `document_id@vversion`, so ingesting one version of a document would wipe every version of it. Not triggered today — one version of each document is registered — but it contradicts the version-awareness the rest of the system has.
- Evaluation cost grows with the question count *and* the corpus, and the runner has no scoping mechanism beyond a hand-typed `--id` list, no caching between runs, and no concurrency in tier 2. `Question` carries no document tag, so "run only what this document touches" cannot be expressed. See *Evaluation will not scale as written*.

### Measured: rerank scores are not a confidence signal

Recorded 2026-08-31 after attempting to add an absolute rerank threshold. It was
implemented, it broke a working question, and it was removed. The measurements
are kept here so the idea is not tried again without new evidence.

Scores from `BAAI/bge-reranker-v2-m3` against the indexed Constitution:

| Question | Best match | Score |
|---|---|---|
| "What are the rights of an arrested person?" | Article 49 | **0.9993** |
| "Police arrested me and did not say why. What are my rights?" | Article 49 | **0.7552** |
| "Can police arrest me without telling me why?" | Article 49 | **0.1344** |
| "Can police arrest me without **teling** me why?" (typo) | Article 49 | **0.0086** |
| "What is the legal process for filing for divorce in Kenya?" (**not in corpus**) | Article 260 | **0.0373** |

Two conclusions:

1. **Article 49 ranked first in every arrest phrasing, including the typo.**
   Retrieval and reranking were never the problem. BGE-M3's subword tokenisation
   makes dense retrieval degrade gracefully on misspellings.
2. **The scale is not comparable across queries — it is inverted.** The *correct*
   source for a typo'd question (0.0086) scores lower than the *best wrong*
   source for a question the corpus cannot answer (0.0373). No fixed floor can
   accept the first and reject the second.

Cross-encoder scores order candidates *within* one query. They are not a
calibrated confidence measure *across* queries. Gating sufficiency on them
produced false refusals on questions the corpus answers — the failure mode that
matters most here, since the product's value is answering correctly *and*
knowing when it cannot.

Sufficiency therefore stays with the LLM's grounding prompt, which handled it
correctly before any threshold existed (see the Phase 3 divorce test). The
durable fixes are the query analyser — which normalises typos and phrasing
before retrieval — and citation verification, both already on the roadmap. Any
future threshold must be calibrated against a labelled evaluation set, per query
type, never hand-picked.

### Measured: chunks must be embedded with their citation context

Recorded 2026-09-01. The first evaluation run scored 34/41, with 7 failures.
All 7 traced to a single cause: the pipeline embedded `chunk["content"]` — the
bare legal text — while the article's *title* went only into the payload.

An article's topic frequently lives in its title and nowhere in its body:

> **Article 16 — "Dual citizenship"**
> *"A citizen by birth does not lose citizenship by acquiring the citizenship of
> another country."*

The word "dual" never appears in the text. The article was unfindable by the
question that most obviously describes it.

Fixed by prepending document, citation and unit title to the text used for
**embedding and reranking only** (`backend/app/core/passage.py`). The stored
`content` is unchanged, so the header never reaches the LLM.

Measured effect on true rank against the full 279-chunk corpus:

| Question | Before | After |
|---|---|---|
| dual-citizenship | #27 | **#2** |
| foreigner-land | #16 | **#2** |
| court-system | #16 | **#1** |
| torture | #6 | **#1** |
| freedom-expression | #11 | **#1** |
| children-rights | #6 | **#1** |

And on reranking, where the effect is larger and selective — correct articles
rise sharply while distractors fall:

| Question | Correct article | Before | After |
|---|---|---|---|
| torture | Article 25 | 0.0014 | **0.2495** |
| freedom-expression | Article 33 | 0.0042 | **0.6469** |
| children-rights | Article 53 | 0.0183 | **0.9271** |

**A discarded theory, kept on record.** The schedules crowding early results
looked like cross-encoder length bias. It was measured and refuted:
correlation between passage length and rerank score was **−0.116**, and
passages over 3,000 characters scored *lower* on average (0.0930 vs 0.1409).

What was actually happening is visible in the "before" scores above — *every*
candidate scored near zero. The reranker was not preferring schedules; it had no
signal at all, and the ordering among near-zero noise was arbitrary. One missing
input starved both stages.

Evaluation result: **34/41 → 40/41**, six questions improved, none regressed.

### Measured: a second document changes retrieval, not just coverage

Recorded 2026-09-03, when the Criminal Procedure Code doubled the corpus from
279 to 591 chunks. The ingestion was correct and needed no downstream changes.
Retrieval was a different story: **52/56 questions passing, three arrest
questions regressed**, and they failed in two distinct ways.

**Capacity — `arrest-bail`.** Article 49 stopped being retrieved *at all*. It
sat between rank 16 and 30, crowded out by CPC bail sections. `RETRIEVAL_TOP_K`
was tuned at 15 against a 279-chunk corpus; at 591 it was simply too narrow.
Raising it to 30 restored retrieval to 41/41 and made **no other question
worse** — the candidate pool has to scale with the corpus. The cost is real:
reranking is the dominant latency and this doubles the passages it scores.

**Ranking — `arrest-reason` and `arrest-silence`.** Article 49 *was* retrieved,
at #9 and #2, and reranking discarded it. One document took all five slots.

| Question | What won | Why |
|---|---|---|
| "Do I have to answer police questions after being arrested?" | s.36A *Remand by court* | its text contains "inquiries ... by the police"; Article 49 says "the right to remain silent" |
| "Can police arrest me without telling me why?" | ss.29, 31, 33, 35 | arrest *powers* and *disposal*, none imposing a duty to give reasons |

The consequences differed, and the difference matters. `arrest-silence` was
given a single irrelevant source and **correctly refused** — the grounding
prompt worked, and it failed safe. `arrest-reason` **answered**, citing s.29 and
s.2. Nothing was fabricated; every citation was real and passed verification.
That is what made it dangerous: a confident, correctly-cited answer to a
question the sources do not address, on the archetypal question in this
project's own vision statement.

**The fix is structural, not another threshold.** `rerank()` now reserves a slot
for the best-scoring passage from a document that would otherwise be shut out,
exempt from the relative tail cut, capped at `top_k - 1` per document. When only
one document appears in the candidates the old path runs unchanged, so a
single-document corpus behaves exactly as before. Tuning `RERANK_TOP_K` or the
0.10 ratio was rejected: Article 49 scored under a tenth of s.36A here, and this
project already has measurements showing cross-encoder scores cannot carry a
hand-picked floor.

Result: **70/71 passing, no regressions**, both failures closed, and tier 2 back
to 0 false refusals and 0 false answers. `rerank_top1` did not fully recover and
was not expected to — diversity gets the right law *in front of the LLM*, it
does not make the cross-encoder rank it first. The three arrest questions that
lost top-1 lost it to CPC sections that genuinely speak to the same facts.

**A metric blind spot this exposed.** `false_answer` only counts answering an
*out-of-scope* question, so `arrest-reason` — in scope, answered, citing none of
the expected authorities — scored as a clean pass. The harness held both facts
(`rerank_hit: false`, `sufficient: true`) and never combined them. Added as
`wrong_authority`: counted and listed, but deliberately **not** a failure
condition, because `expect_any_of` lists the authorities we know of, not every
one that could be legitimate — CPC s.123 turned out to be a valid answer to a
bail question nobody had listed.

### Measured: one aggregate subject verdict cannot gate sufficiency

Recorded 2026-09-12 after building a subject-matter gate, measuring it twice,
and removing it. Kept so the idea is not tried again in this form.

**The failure it was built for.** Tier 2 over the 15 out-of-scope questions
found one false answer. *"What legally happens if I cannot pay back my debts?"*
was answered with `sufficient=True`, citing CPC s.174 (costs ordered on a
**criminal conviction** → imprisonment up to three months) and s.342
(commitment for non-payment of a **criminal fine** → up to six months). The
question is about civil debt. The answer told the reader they could be
imprisoned for owing money. Every citation was real and passed
`verify_citations` — a relevance failure, not a hallucination, and therefore
indistinguishable from a correct answer.

**What was tried.** Three required schema fields — `question_subject`,
`sources_subject`, `subject_match` — with code forcing `sufficient = False`
when the match was not good enough. This deliberately copied the pattern that
fixed question polarity: make the model state the judgement explicitly, then
enforce it outside the model rather than trusting the prompt.

**Two settings, two opposite failures:**

| `SUFFICIENT_MATCH` | `debt-inability` | `arrest-rights` / `children-rights` |
|---|---|---|
| `("same",)` | refused ✓ | **falsely refused** ✗ |
| `+ "related_but_different"` | **answered** ✗ | answered ✓ |

**Why no setting works.** The model returned `related_but_different` for *both*:

- `children-rights` — Article 53 *"Children"* ranked **first**, alongside
  Articles 21, 43, 14 and 19. The answer was right there.
- `debt-inability` — nothing in the set concerned civil debt at all. Its own
  `sources_subject` read *"imprisonment for non-payment of fines and costs in
  criminal cases"*, which is exactly correct.

Identical verdict, opposite truth. The model's *perception* was accurate; the
aggregate *judgement* carries no signal.

The cause is structural. `sources_subject` describes the whole reranked set,
and that set is diverse by design — reranking now deliberately reserves a slot
for a second document. Asked whether five mixed passages "match" a question,
the honest answer is nearly always "sort of", whether or not the governing
provision is among them. **The gate measured breadth, not wrongness.**

**Removed** rather than left in place: after relaxing, it fired on nothing
(`subject-gated 0` on every run), so it was carrying schema fields, prompt
tokens and per-call cost for no benefit.

**What would actually work** is per-source relevance — which specific chunk
ids govern the question, refusing when that set is empty — since Article 53
governs children's rights while nothing in the debt set governs civil debt.
That is the entailment item already on the roadmap. It was not attempted here:
two measurement cycles was the agreed budget for a cheap interim, and the
honest conclusion is that this problem does not have a cheap interim.

**`debt-inability` therefore remains an open, known false answer.**

### Measured: reranking earns its place, and must keep overriding retrieval

Recorded 2026-09-20, after hybrid retrieval reached 67/67 and both remaining
failures changed from `NEVER RETRIEVED` to `RETRIEVED THEN DROPPED`. That made
the cross-encoder look like the new bottleneck, and raised a fair question:
reranking was added when retrieval was dense-only and noisy, so does it still
earn 1.1 GB of RAM and the dominant request latency?

Measured with `--no-rerank`, which takes retrieval's own top-k:

| | reranked | retrieval top-k only |
|---|---|---|
| `rerank_hit` | **65/67** | 57/67 |
| `rerank_top1` | **44/67** | 37/67 |

**It rescues eight questions.** And the eight are all one pattern:

```
arrest-reason        Article 49 at retrieval rank 10
arrest-silence       Article 49 at rank 6
arrest-court-time    Article 49 at rank 18
arrest-bail          Article 49 at rank 26
state-funded-lawyer  Article 50 at rank 15
privacy-search       Article 31 at rank 11
```

Retrieval fills the top with Criminal Procedure Code sections — they are
lexically and semantically closer to how the question is phrased — and the
**constitutional right sits buried**. The reranker plus the per-document
diversity rule is what digs it out. Without it, "can police arrest me without
telling me why" returns five CPC sections on arrest powers and never surfaces
Article 49.

**A planned change was abandoned because of this.** The intended fix for the
two remaining failures was to fuse rerank order with retrieval order by RRF,
so a passage retrieved at rank 8 could not be erased by a single bad
cross-encoder score. The measurement shows that would break the eight above:
they depend on the cross-encoder overriding retrieval order *hard*, Article 49
moving from rank 26 into the top 5. Blending retrieval rank back in weakens
precisely the behaviour doing the work — trading eight rescues for two.

**Do not dilute the reranker's authority over ordering.** If the two remaining
failures are ever worth fixing, it needs a different mechanism: a better
cross-encoder, or per-question diagnosis. Not a blend.

Left open deliberately. `county-government-role` is a suspected question
defect awaiting legal review; `penal-murder-sentence` has s.204 at rank 8
after hybrid moved it from 53, and the cross-encoder makes the final call
badly. Two failures out of 67, against a risk of breaking eight.

### Evaluation will not scale as written

Recorded 2026-09-03 after a full read of `evaluation/`. The harness is sound in
what it measures; the problem is what it costs to run.

- **No scoping.** `Question` carries no document tag, so "run only what this
  document touches" cannot be expressed. The CPC work used a hand-typed `--id`
  list, chosen by reading section titles and guessing which existing questions a
  new document might endanger. That judgement was correct here and does not
  scale past a few documents.
- **No caching.** Nothing keys a result on question + corpus version + code
  version, so a docs-only change triggers the same full recompute as a retrieval
  change.
- **No concurrency.** `evaluate_tier2` is a plain sequential loop; every OpenAI
  call blocks the next, though they are entirely independent.
- **Out-of-scope questions pay full cost for nothing.** All three tier-1 metrics
  stay `None` for `answerable: false`, yet each still runs a full retrieve and
  rerank at top-30 on every run.
- **Models are loaded once**, correctly — that common failure mode is *not*
  present. They are constructed in `main()` and passed into the loop.

What is *not* wrong: the tier 1/tier 2 split, scoring retrieval and reranking
separately so a failure names its stage, keeping `false_refusal` and
`false_answer` apart, and deterministic citation-identity checking. Those earn
their keep. The gap is selectivity — running only what a change can affect.

This matters beyond convenience. An evaluation too slow to run is one that stops
being run, and this system's whole claim rests on being able to prove it still
retrieves the right law.

### Architectural gaps

- **Case law has no structural model.** Judgments are not chapters and articles. They need holdings vs obiter, court hierarchy, and overruled/distinguished status. The corpus plan includes case law; the pipeline has no concept of it.
- **Amendment/repeal lifecycle — foundations laid, policy outstanding.** Versions can now coexist and be filtered (`in_force`, `effective_from`, `effective_to`), and answers carry an "as at" date. What does not exist yet is the *process*: detecting that a law has been amended, ingesting the new version, and marking the old one superseded. Until that exists, correctness depends on someone remembering to update the registry.
- **Dense-only retrieval.** Legal queries need exact lexical anchors — "Section 45", "Cap 141", defined terms. BGE-M3 natively produces sparse vectors and Qdrant supports fusion; only dense is used today.
- ~~**No evaluation harness.**~~ **Closed 2026-09-01.** 71 questions, two tiers, baseline recorded.
- **Query vocabulary gap.** A citizen asks "do I have to answer police questions"; the law says "the right to remain silent". Nothing bridges the two, so a section merely containing "inquiries ... by the police" outranks the constitutional right. Source diversity now stops that costing the answer, but the ranking is still wrong. This is the query-analyser item, and the corpus growing makes it worse, not better.
- **Kenya Law licensing/terms not confirmed** for redistributing legal text in an open-source repository.

### Serving constraints

- BGE-M3 (568M) plus the reranker (568M) occupy roughly 2.3–2.5 GB of RAM. A 2 GB VPS will not run this.
- CPU cross-encoder reranking of 15 long legal passages is the dominant request latency — not the OpenAI call.
- The Hugging Face unauthenticated rate-limit warning during model load is benign.

---

## Roadmap

Revised 2026-08-31. The ordering changed for two reasons: the ingestion layer must be safe before the corpus grows, and nothing downstream can be measured without an evaluation set.

```text
1. Ingestion foundation        ← current work
   Generic, idempotent, registry-driven pipeline.
   Stable IDs, batched ingest, central config,
   versioning fields, named vectors.

2. Evaluation set                        ← DONE
   56 questions, two tiers, baseline recorded.
   Question set still needs legal review.

3. Corpus expansion                      ← IN PROGRESS
   Criminal Procedure Code (Cap. 75) ingested 2026-09-03.
   It validated the generic pipeline and broke retrieval
   ranking in three arrest questions - both findings recorded.

3b. Evaluation scoping                   ← NEW, blocks 3 at scale
   Tag questions by document; run only what a change can
   affect; cache between runs; parallelise tier 2. Without
   this, every added document makes every future change
   more expensive to verify.

4. Evidence sufficiency
   NOT an absolute rerank threshold - that was tried and
   removed; see "rerank scores are not a confidence signal".
   Calibrate against the evaluation set, or verify claims
   against cited passages.

5. FastAPI service
   Load models once at startup. Structured answer + sources.
   Moved much earlier: it forces the right serving design
   and lets the product be used and dogfooded.

6. Scenario understanding
   Fact extraction, legal-issue identification, multi-query.

7. Reasoning and structured citations
   fact → rule → application → conclusion.

8. Flutter application
```

**Why scenario understanding moved after corpus breadth:** with a Constitution-only corpus, a perfect scenario analyser produces perfectly decomposed queries that still retrieve nothing. The divorce test already demonstrated this. Better analysis over a thin corpus yields better-worded refusals, not better answers.

---

## Data automation (later)

Out of scope until the generic ingestion pipeline exists.

n8n (or similar) may later monitor official sources, detect new or updated documents, download them, and trigger the Python pipeline. It remains an orchestration layer, not the processor.

```text
              n8n
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
 Kenya Law  Judiciary  Other
      │        │        │
      └────────┼────────┘
               ▼
       Detect changes
               │
               ▼
         Download file
               │
               ▼
       Python pipeline
               │
               ▼
        Validate/process
               │
               ▼
          Update index
```

---

## Development philosophy

Build bottom-up. Do not rush to the chatbot.

```text
PHASE 0  Foundation                                 done
    ↓
PHASE 1  Understand + process Kenyan legal data     one item left
    ↓
PHASE 2  Build searchable legal knowledge base      done for 1 document
    ↓
PHASE 3  Build reliable RAG / AI                    ← we are here
    ↓
PHASE 4  Expose it through FastAPI
    ↓
PHASE 5  Build Flutter application
```

Priority order when trade-offs arise:

**accuracy > grounding > usefulness > speed > cost**

---

## Progress log

- **2026-08-23** — Recorded the master plan. Phase 0 marked done (minimal). Phase 1 marked in progress: Constitution collected, extracted, and cleaned; quality inspection is next.
- **2026-08-25** — Constitution validated (validator initially failed, then fixed and passed). Metadata built with SHA-256 checksums. Legal structure detected: chapters → articles → subsections → paragraphs → schedules. Chunked into 270 chunks (264 articles + 6 schedules).
- **2026-08-25** — Chose `BAAI/bge-m3` as the embedding model. Installed Docker and ran Qdrant locally. Generated 1024-dim embeddings for all chunks and ingested them into the `pocket_lawyer_legal` collection; vector count verified against chunk count.
- **2026-08-26** — Built `LegalRetriever`. The system can now take a natural-language legal question, turn it into a BGE-M3 vector, query Qdrant, and return the most relevant legal provisions. Verified on *"What are the rights of an arrested person?"*.
- **2026-08-28** — Added `LegalReranker` using `BAAI/bge-reranker-v2-m3`. Top 15 → top 5. Confirmed it materially improves ordering over raw semantic retrieval.
- **2026-08-30** — Built `LegalLLM` on `gpt-4o-mini` via the OpenAI Responses API with a strict grounding prompt. Verified both a supported answer (Article 49) and a correct refusal (divorce). Wired the LLM into `LegalRAG.answer()`, completing retrieval → reranking → generation end to end.
- **2026-08-31** — Full codebase and data review. Recorded verified measurements (270 chunks, 264/264 articles present, chunk size distribution, 6.3 MB embeddings file tracked in git). Corrected this plan: Phase 1 metadata and structure detection marked done; Phase 2 marked done for the first document; Phase 3 marked in progress. Documented ingestion blockers and architectural gaps, and resequenced the roadmap to put the ingestion foundation and an evaluation set ahead of corpus expansion and scenario understanding.
- **2026-08-31** — **Ingestion foundation built.** Replaced the five per-document scripts with one generic, registry-driven pipeline (`python -m data_pipeline.run`). Renamed `data-pipeline/` to `data_pipeline/` so stages can share code; added a common document IR so chunking, embedding and storage no longer know the document type. Fixed all four ingestion blockers: deterministic `uuid5` point IDs, generic stages behind per-type adapters, streaming batched embed-and-upsert with no on-disk vectors, and one central config imported by both the pipeline and the AI layer. Collection schema moved to named vectors with a sparse slot declared for future hybrid search, and the payload now carries `version`, `effective_date`, `in_force` and `as_at`. Added the first tests (33). Constitution re-ingested: **279 chunks**, longest 3,996 characters — Schedule 6's 24,146-character chunk is now split. Re-running produced **279 → 279 points**, proving ingestion is idempotent and a second document can no longer overwrite the first. Retrieval parity confirmed (Article 49 still ranks first) and the divorce refusal still holds.
- **2026-09-02** — **Version-aware, document-first ingestion.** Chunk identity became `document_id@vversion-slug`, so two editions of the same Act no longer produce identical ids and overwrite each other. Derived artifacts moved from five stage directories to one directory per document version (`data/documents/<id>@v<version>/`), with `raw/` kept separate as the immutable source. `effective_from` / `effective_to` joined `in_force` and `version` on the payload and in metadata. Adapter tables moved out of the orchestrator into `data_pipeline/adapters.py` behind a `StructureParser` Protocol, and `CleanerSpec.start_pattern` became optional — it was a Constitution assumption sitting in supposedly generic code. Added `tests/test_multi_document.py`, which proves two document shapes and two versions produce disjoint ids without needing a second real document. **The evaluation reproduced the baseline exactly — 41/41, 40/41, 33/41, no regressions** — confirming that identity and file locations moved without disturbing a single retrieval decision. Old `data/` directories deliberately left in place.
- **2026-09-01** — **Contextual embedding.** Chunks are now embedded and reranked with their document, citation and article title prepended (`backend/app/core/passage.py`); the stored content sent to the LLM is unchanged. Diagnosed offline before changing anything, which also refuted a length-bias theory. Evaluation went **34/41 → 40/41**, six questions improved, none regressed. See *Measured: chunks must be embedded with their citation context*. The one remaining failure, `county-government-role`, is a suspected defect in the question rather than the system — the Fourth Schedule distributes county functions and was never listed as an acceptable authority. Awaiting legal review.
- **2026-09-20** — **Hybrid retrieval: 100% retrieval, and the bottleneck
  moves.** Dense-only left a measured hole. Asked *"What is the sentence for
  murder in Kenya?"*, section 204 — *"Any person convicted of murder shall be
  sentenced to death"*, eleven words answering it almost verbatim — came back
  at **rank 53**, beaten by *"Conspiracy to murder"*, which is longer and
  shares more vocabulary (content-only cosine 0.6992 against 0.6026). Dense
  encodes meaning and blurs exact terms; legal queries need anchors.
  BGE-M3's lexical head turned out to be unreachable as assumed:
  `sentence_transformers` silently converts the model to a generic 4096-dim
  projection that is not term-based, and FlagEmbedding would load a second
  2.2 GB copy. The head is `Linear(1024, 1)`, so `backend/app/core/sparse.py`
  applies it to token embeddings from the model already in memory — no new
  dependency, no extra RAM. The sparse slot declared during the ingestion
  refactor was finally populated; re-ingest reported **962 → 962**.
  Fused with **Reciprocal Rank Fusion**, chosen because it combines *ranks*
  rather than scores: dense cosine and sparse dot product are on different
  scales, and this project has twice been burned reconciling incomparable
  scores.
  **The budgets are unequal, and that was measured rather than assumed.** At
  equal budgets hybrid fixed three questions and broke one — the Criminal
  Procedure Code's many sections containing the literal term "bail" filled the
  list and pushed Article 49 out of range, RRF compounding it by rewarding
  appearance in both lists. Sweep at tier 0: `SPARSE_TOP_K` 5 → 67/67,
  10 → 67/67, 20 → 66/67, 30 → 66/67. Took 5, for margin and because erring
  low merely fails to rescue while erring high displaces constitutional rights
  with statutory detail.
  **Result: retrieval 66/67 → 67/67, the first 100%.** `rerank_hit` 64 → 65,
  no regressions. `county-government-role`, failing since before the Criminal
  Procedure Code existed, is now retrieved.
  **The bottleneck has moved.** Both remaining failures changed from `NEVER
  RETRIEVED` to `RETRIEVED THEN DROPPED` — section 204 is at rank 8 and the
  cross-encoder discards it. Retrieval is solved on this question set;
  reranking is now the single weak link, and the same cross-encoder scored a
  correct answer at 0.042 earlier in the week. 220 tests.
- **2026-09-20** — **Third document, and an evaluation harness that scales.**
  The Penal Code (Cap. 63), 2023-12-11 consolidation: **371 chunks from 368
  live sections**, sections 1–398 complete with 18 lettered insertions, 48
  repealed dropped, no schedules at all. Collection **591 → 962**.
  It is **Part → Chapter → Section** where the Criminal Procedure Code is
  Part → Section, so `act.py` gained optional Chapter support — additive, with
  a test asserting the CPC still produces two-level paths. Citations now read
  *Part II — Chapter XVIII — Section 203*, putting the chapter title into the
  embedding header. `CleanerSpec` also gained `end_pattern`: this Act closes
  with an alphabetical index the document itself disclaims as *"not part of the
  Act"*, and indexing it would have embedded page-number fragments as law.
  **Zero regressions across a 63% corpus increase** — the reranking diversity
  fix held at three documents where the CPC had broken three questions at two.
  71 → 82 questions; new baseline 79/82.
  **The harness was the real work.** A full run had reached hours, all of it in
  one place: 30 cross-encoder passes per question on CPU. Now:
  **tier 0** (retrieval only, no reranker loaded, seconds) — which would alone
  have caught `arrest-bail`, the CPC's worst regression, since that was a pure
  retrieval miss; a **two-level cache** whose second level is keyed on the
  *retrieved candidates* and deliberately **not** on the corpus, so adding a
  document reranks only the questions whose top-k moved; **batched reranking**
  across questions; and `--document` for scoped runs, which needed no schema
  change because expected chunk ids already name their document. Both cache
  keys hash the source of `retriever.py`, `reranker.py` and `passage.py` — the
  property that makes caching safe rather than a way to hide regressions.
  Verified transparent: a warm run reproduced the cold run exactly
  (66/67, 64/67, 43/67) in seconds. 160 → **205 tests**.
  Also fixed: comparing runs across tiers reported spurious regressions,
  because `passed()` means something different at each tier. Every `--tier 2`
  run against the tier 1 baseline had been affected.
- **2026-09-19** — **FastAPI serving layer and semantic intent routing.** The
  engine now has an application around it. `main.py` builds one `LegalRAG` in a
  lifespan, so models load **once** instead of per process; `POST /ask` routes,
  `GET /health` does a live Qdrant count. Sources are returned as structured
  payload metadata, with `sources` (what the answer cited) kept separate from
  `considered` (everything the LLM saw). A sync endpoint plus one lock
  serialises the CPU-bound path — a property of running BGE in-process on CPU,
  not a ceiling; the lock lives in the route layer so the pipeline is unaware.
  `rag.py`, `retriever.py`, `reranker.py`, `llm.py` and `answer.py` were **not
  modified**, which is what kept the evaluation harness valid throughout.
  **Routing moved from regex to semantics after two live misses.** The first
  router matched patterns against the raw string: "what is your name" slipped
  past one covering "what's your name", and "morning" past one requiring "good
  morning". Each miss cost a retrieval, a 30-passage rerank and a paid LLM call
  to produce an unhelpful refusal — the inevitable result of enumerating an open
  set by hand. `IntentClassifier` now compares the query embedding against
  per-intent centroids, with LEGAL as one of the classes: **nearest centroid,
  never a threshold**, for exactly the reason rerank scores cannot carry one.
  Measured over 40 labelled inputs, the classes separate with no overlap
  (conversational +0.094..+0.361, legal −0.222..−0.006), giving **39/39 on
  routing direction and zero legal questions misrouted to chat**.
  `CONVERSATIONAL_MARGIN = 0.05` sits in the gap between the clusters, not
  threaded through it. `gm`, `sup`, `habari yako` and `niaje` route correctly
  without appearing in any pattern or exemplar, and "hello, can police arrest me
  without a warrant" scores −0.217 — meaning beating spelling, which was the
  point. The regexes survive as a 13-entry fast path so "hi" skips the
  embedding. 130 → **160 tests**, still no Qdrant or API key required. See
  *Measured: one aggregate subject verdict cannot gate sufficiency* for the
  unrelated dead end removed the same week.
- **2026-09-03** — **Second document: the Criminal Procedure Code (Cap. 75).**
  223-page consolidation "as at 11 December 2023", ingested as
  `criminal-procedure-code@v2023-12-11` — **312 chunks from 311 live sections**,
  sections 1–394 complete with no gaps plus 33 lettered insertions
  (`36A`, `137A`–`137N`, `379A`) across 13 Parts. 116 repealed sections and all
  four schedules are deliberately excluded: the schedules are multi-column
  tables that line-based PDF extraction shreds into one line per cell, which
  would have added ~29 chunks of noise matching arrest queries. Collection
  went **279 → 591**, and re-ingesting the Constitution afterwards reported
  **591 → 591**, proving the two documents coexist and ingestion is idempotent.
  The generic pipeline needed **no changes**; what it needed was a cleaner spec,
  a reusable `structure/act.py`, and a validator — correcting the "one new
  structure adapter" estimate, since a cleaner spec names the document itself
  and cannot be derived. Three arrest questions regressed and were fixed by
  scaling `RETRIEVAL_TOP_K` 15 → 30 and reserving a reranked slot for a
  shut-out document; added `wrong_authority` to the harness after finding that
  answering an in-scope question from the wrong law scored as a clean pass.
  Question set 56 → 71. **70/71 passing, no regressions, 122 tests.** Full
  analysis in *Measured: a second document changes retrieval, not just
  coverage*.
- **2026-09-01** — **Evaluation harness.** 56 questions (41 answerable, 15 out of scope) in `evaluation/questions.yaml`, scored in two tiers: tier 1 runs retrieval and reranking with no API key and no cost, tier 2 adds the LLM for refusal correctness. Retrieval and reranking are scored separately so a failure names the stage that broke, and `false_refusal` and `false_answer` are counted apart rather than averaged. `baseline.json` records the accepted state and a run reports per-question status changes against it, exiting non-zero on regression. First baseline: 34/41. Question set drafted against article titles and **not yet reviewed for legal correctness**.
- **2026-08-31** — **Structured answers.** The LLM now returns a schema-constrained object (`backend/app/ai/answer.py`) instead of prose: `sufficient`, `question_type`, `verdict`, `explanation`, `qualifications`, `cited_chunk_ids`. Fixes question polarity by construction — the opening word of a polar answer is written by code from the `verdict` enum, so it cannot contradict the explanation. *"Can police arrest me without telling me why?"* now answers **"No."** Citations are checked against the retrieved chunks deterministically, so fabricated ones are stripped and reported with no extra model call. The refusal became a field rather than a magic string and now explains what was missing. Cited sources are separated from considered sources, so a refusal no longer lists irrelevant chunks as if they supported it. Phase 3 Citation generation ticked; verification remains open on the entailment half. 50 tests.
- **2026-08-31** — Attempted an absolute rerank score threshold as a sufficiency gate. It caused a false refusal on *"Can police arrest me without telling me why?"*, a question the corpus answers. Measurement showed the approach is unworkable, not merely miscalibrated — see *Measured: rerank scores are not a confidence signal*. Removed the floor; reranking now trims only relative to each query's own best match and can never refuse on its own. Sufficiency stays with the LLM's grounding prompt. Recorded the planned structured answer schema to fix question polarity by prevention rather than by a second verification call.
