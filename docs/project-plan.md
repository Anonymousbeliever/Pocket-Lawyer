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

**Five documents served over HTTP, three more ready to ingest, with an
evaluation harness that scales.**

```text
Question
   ↓
Intent routing (nearest centroid) ──→ conversational, no retrieval
   ↓ legal
BGE-M3 embedding
   ↓
Qdrant (1157 vectors: Constitution 279, Criminal Procedure Code 312,
        Penal Code 371, Law of Succession 103, Employment Act 92)
        + 407 prepared: Children 258, Marriage 97, Small Claims 52
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

The generic pipeline has now absorbed four hierarchies — Chapter → Article,
Part → Section, Part → Chapter → Section, and sections belonging to no Part at
all — without the orchestrator, registry, IR, chunker or vector store changing
once. Per document the cost is a cleaner spec, a validator, and a small,
*document-specific* fix to `act.py` that nobody predicts in advance. Three
documents running, that fix has been mandatory; see *Measured: Kenya Law is
not internally consistent*.

**Current state — baselined 2026-09-23 over the complete corpus:**

```text
133 questions (118 answerable, 15 out of scope)
121/133 passing

retrieval_hit   116/118   98.3%
rerank_hit      106/118   89.8%
rerank_top1      74/118   62.7%
```

**Twelve failures, in four distinct classes** — the distinction matters
because they need different fixes, and three of the four are not fixable by
anything currently on the table:

| Class | Questions | Cause |
|---|---|---|
| **Retrieval breadth** | `state-funded-lawyer`, `divorce-process` | outside `RETRIEVAL_TOP_K` 30 — a **measured, deliberate** trade, see *Measured: RETRIEVAL_TOP_K stays at 30* |
| **Cross-document displacement** | `county-government-role` | **a regression** — the Children Act's Part VI, *Role of County Governments*, now outranks the Fourth Schedule |
| **Cross-encoder resemblance** | `employment-notice`, `penal-murder-sentence`, `succession-left-out-of-will`, `succession-grant-application`, `children-maintenance-order`, `children-adoption-eligibility`, `children-child-labour` | the measured defect, both fixes rejected |
| **Rank-#1 pathologies** | `children-parental-responsibility`, `smallclaims-lawyer` | retrieved **first** and discarded — the starkest form of the same defect |

`smallclaims-lawyer` is the one to look at when this needs explaining: the
question is *"Do I need an advocate to go to the small claims court?"*,
section 20 is titled **"Representation before the Court"**, retrieval ranks it
**first**, and the cross-encoder replaces it with the appointment of
Adjudicators and the interpretation section.

**MVP scope is fixed at eight documents**, and all eight are now written.

| | Document | Chunks | State |
|---|---|---|---|
| 1 | Constitution of Kenya, 2010 | 279 | ingested |
| 2 | Criminal Procedure Code (Cap. 75) | 312 | ingested |
| 3 | Penal Code (Cap. 63) | 371 | ingested |
| 4 | Employment Act (Cap. 226) | 92 | ingested |
| 5 | Law of Succession Act (Cap. 160) | 103 | ingested |
| 6 | Children Act (Cap. 141) | 258 | **prepared, dry-run verified** |
| 7 | Marriage Act (Cap. 150) | 97 | **prepared, dry-run verified** |
| 8 | Small Claims Court Act (Cap. 10A) | 52 | **prepared, dry-run verified** |

**1,564 chunks** once the last three are in. That set covers criminal justice
end to end, employment, inheritance, family formation and dissolution,
children, and a route to actually claim money.

Two question flips are worth naming. `writing-a-will` opened with document
five. `divorce-process` opens with document seven — **the original refusal
test case**, the question this system correctly declined over a
Constitution-only corpus in the very first Phase 3 tests, and which stayed out
of scope for five documents. Its wording is unchanged, so the flip is honest.

Deliberately cut, and not to be re-proposed without saying so: the Law of
Contract Act (about four sections; it *imports* English common law rather than
containing contract law), the Evidence Act and the Civil Procedure Act (written
for advocates, and civil procedure's operative detail is in the Rules), and the
Companies and Data Protection Acts (business users, not this product's
citizen). The Land Act and Land Registration Act are deferred, not cut — high
value, two large paired documents, and wrong answers about land do the most
damage.

**What limits the product now:**

1. **Reranking, not retrieval.** Both remaining failures are
   `RETRIEVED THEN DROPPED`, and both are the same defect rather than two
   coincidences — see *Measured: the cross-encoder ranks by resemblance, not
   by answerhood*. **Both proposed fixes were measured and rejected**; the
   limit turned out to be legal domain knowledge, not model class. It is the
   single weak link in the chain, and it is being carried deliberately.
2. **Corpus breadth.** Four documents of the eight.
3. **The application layer.** No conversation state, no persistence, no auth,
   no mobile app.

Still open and recorded: `debt-inability`'s false answer and the
`delete_document` version bug.

**Closed:** `county-government-role` was not a system failure. It failed for
eighteen runs because its expectations named Article 186, which does not list
county functions — its operative text is a pointer to the Fourth Schedule.
Reranking had been surfacing the Schedule correctly the whole time. The
expectations were corrected; Articles 183 (county *executive committee*
functions) and 174 (objects of devolution) were removed as not answering the
question. **A failing test is not automatically a failing system.**

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
- ~~**Remaining item — "Build ingestion pipeline":**~~ **Done 2026-09-02.** The orchestrated, document-agnostic pipeline is `data_pipeline/run.py`, and four documents across three hierarchies have gone through it. The only outstanding Phase 1 item is the source inventory.
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

**Status:** Done and proven on four documents — corpus breadth outstanding
(four of the eight in MVP scope)

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

> **Resolved 2026-09-23.** `county-government-role` was indeed a question
> defect, and the suspicion is now confirmed rather than pending. See
> *Current position*.

---

### Measured: a civil document costs nothing in retrieval — and the rerank cache nothing at all

The Employment Act (Cap. 226) is the first document in the corpus that is
neither criminal nor constitutional. It was added as a probe as much as for
coverage: every retrieval finding this project had was drawn from criminal and
constitutional text, and two load-bearing rules — per-document rerank
diversity, and `SPARSE_TOP_K` held at 5 — were tuned on a corpus where the
*right* is constitutional and the *procedure* is in a criminal Act.

**Ingestion: 45 pages, 114,422 cleaned characters, 91 live sections, 92
chunks.** Corpus 962 → 1054.

**Retrieval was unaffected.** Tier 0 across all 96 questions: `retrieval_hit
80/80`, **zero regressions**. The expected failure — employment vocabulary
("notice", "rights", "pay", "termination") displacing constitutional articles
the way Criminal Procedure Code sections pushed Article 49 out of range —
did not happen. `RETRIEVAL_TOP_K = 30` holds at 1054 chunks.

**Per-document diversity held too.** `employment-discrimination-at-work` was
added deliberately as the cross-document probe, expecting both Article 27 and
section 5. It keeps its authority through reranking. A civil Act does not shut
the Constitution out.

**The level-2 cache saved nothing, and this contradicts its design note.**

```text
Cache: retrieval 96 reused / 0 computed
       rerank     0 reused / 96 computed
```

The rerank cache is keyed on the *retrieved candidates* rather than the corpus,
specifically so that adding a document reranks only the questions whose top-k
actually moved. The premise was that adding the Penal Code would not disturb
`dual-citizenship` or `official-languages`.

It moved the top-30 for **all 82 existing questions**. Not one was reused.
Plausibly the sparse side: employment text is ordinary English, so it enters
the lexical top-5 for questions that have nothing to do with employment, which
perturbs the fused list everywhere. Hybrid retrieval and candidate-keyed
caching work against each other.

The cache design is not wrong, but its stated benefit did not materialise for
document four, and the cost estimate for documents five through eight should
assume a **full rerank per document** until measured otherwise.

**One parser bug, and the silent kind.** The Act opens `Part 1 – PRELIMINARY`
in Arabic and switches to Roman for Parts II–XIII. `act.py`'s Roman-only
`PART_PATTERN` did not mislabel Part 1 — it failed to match, so `_parse_tree`
went on treating the text as the Act's opening block and dropped every section
until Part II. **Sections 1–3 disappeared, taking section 2's definitions of
"employee", "wages" and "termination" with them**, and the only downstream
symptom was a section count three short.

This is exactly why a per-document validator is non-negotiable. The fix is one
alternation, `(\d+|[IVXLCDM]+[A-Z]?)`, Arabic tried first — `"III"` cannot
match `\d+`, so every Act already ingested parses identically and no chunk ids
churn. `validators/employment_act.py` now checks Part 1's sections **by name**,
so the regression cannot return as a count that merely looks three short.

---

### Measured: Kenya Law is not internally consistent, and every failure is silent

Documents four through eight each needed a `act.py` change that nobody
predicted. The claim that "the next Act reuses the parser unchanged" has now
been **false four times running**. What survives is the weaker, true claim:
the orchestrator, registry, IR, chunker and vector store need no changes, and
the parser needs a small one.

Every one of these failed **silently**. None raised. In each case the only
symptom was a section count lower than it should be, and the only reason it
was caught is that per-document validators assert a measured count.

| Document | What broke | Cost if unnoticed |
|---|---|---|
| **Employment (Cap. 226)** | `Part 1` in Arabic, rest Roman. Roman-only pattern did not mislabel it — it failed to match, so the parser kept treating text as the opening block | sections 1–3 dropped, including s.2 **Interpretation** |
| **Succession (Cap. 160)** | **No Part I at all.** Divisions begin at `Part II – WILLS` and ss. 1–4 sit above them ungrouped | 4 sections dropped, including s.3 **Interpretation** |
| **Children (Cap. 141)** | s.218 ends *"...in accordance with the **Fifth Schedule.**"*; the line wrapped and a case-insensitive prefix match read it as a schedule heading | **32 sections dropped** — the parse *stopped* at 218, discarding all of Part XV, Children in Conflict with the Law |
| **Marriage (Cap. 150)** | s.18 ends *"...under section **50.**"*, wrapping `50.` onto its own line; and the lone schedule is headed bare `SCHEDULE [s. 97]`, not `FIRST SCHEDULE` | a phantom s.50 (benign — empty, so dropped) and the repealed-acts schedule attaching to the last section |

**The generalisation, learned the expensive way.** A pattern whose match
*terminates* the parse is categorically more dangerous than one that creates a
unit. `SCHEDULE_PATTERN` ends the operative text, so one false positive cost
32 sections; `SECTION_NUMBER_PATTERN` merely creates a unit, and its false
positive self-healed through the empty-content drop. **Make terminating
patterns as strict as the real thing allows** — case, anchoring, full-line.
Real schedule headings are UPPERCASE and occupy the whole line, optionally
with a bracketed reference (`FIFTH SCHEDULE [s. 218(5)]`); a wrapped sentence
fragment is title-case and carries the sentence's punctuation, so requiring
both excludes it.

**Two structural decisions worth keeping.** Sections belonging to no Part get
a **one-level citation path** (`["Section 3"]`) rather than a synthesised
`Part I` — inventing a division the Act does not contain would put a false
citation in front of a user. And the Marriage Act's phantom section 50 is
recorded in its validator as `KNOWN_TEXT_DUPLICATES` with an assertion that
the *surviving* section 50 is the real one in Part VIII, rather than the
duplicate check simply being switched off.

**Every change was verified non-destructive**: after each, all
already-ingested Acts reparsed to **byte-identical chunk ids**.

---

### Measured: the cross-encoder ranks by resemblance, not by answerhood

Both remaining failures are the same defect, seen in two unrelated documents.
This is no longer an anecdote about one question.

**`employment-notice`** — *"How much notice must my employer give before
terminating my job?"* Section 35 *Termination notice* is retrieved at #7 of 30
and dropped. What wins is section 78, *Notification of termination of
employment*:

| | chars | content |
|---|---|---|
| s.35 — correct, dropped | 2166 | the rule is buried in (1)(c): "terminable by either party at the end of the period of **twenty-eight days** next following the giving of notice" |
| s.78 — reranked #1 | 192 | "An employer shall notify the termination of every employment... to the nearest employment service office **within two weeks**" |

s.78 matches every content word of the question and supplies a duration — while
being an administrative duty to notify a government office, not notice to the
employee. **The top-ranked source states a period that is prominent and wrong.**

**`penal-murder-sentence`** — section 204 is eleven words that answer the
question verbatim, and loses to *"Conspiracy to murder"*, which is longer and
shares more vocabulary.

**This is not a plumbing bug.** `passage.py` was verified to include
`"Section 35: Termination notice"` in what the reranker judges. The header is
there. The model reads it and still prefers the resemblance.

**The grounding prompt held.** Tier 2 on `employment-notice`:
`false answers 0`, `wrong authority 0`, `false refusals 1`. Handed s.78 and its
confident "two weeks", the system **refused** rather than answering wrongly.
Under `accuracy > grounding > usefulness`, that is the failure mode to have.

**Why no amount of tuning fixes this.** `bge-reranker-v2-m3` is not
malfunctioning; it scores how *related* a passage is, and s.78 genuinely is
related. What separates it from s.35 is **answerhood**, which a 568M relatedness
model does not represent. A capability the model lacks cannot be reached by a
threshold (tried, removed), a blend (tried, abandoned), or a per-document boost
(the enumeration trap the regex router died of).

Two levers were proposed, and **both were then measured. Neither is worth
building.** See the section below — it supersedes the hypotheses that stood
here, one of which was wrong.

**A related chunking weakness, recorded separately because it is real even
though lever 1 failed.** The Fourth Schedule is cut mid-list by the
4,000-character cap: county functions 1–4 end `-part-1` and 5 onward open
`-part-2`, and `-part-1` mixes *national* and *county* functions in one blob.
A character cap split a legal list and ignored the Schedule's own
Part 1 / Part 2 boundary. That is a citation-quality problem worth fixing on
its own merits; it is **not** a fix for reranking.

---

### Measured: neither lever fixes the reranker — and the experiment paid for itself

Run 2026-09-23, before any production code was written. Both levers tested
against the real corpus; scripts were throwaway and are not in the repo.

**The decision rule, set before running:** the judge must *lose nothing*.
Fixing 2 and breaking 0 is a build. Fixing 2 and breaking 3 is not, whatever
the aggregate says. The cross-encoder already gets **78 of 80** right, so
there was far more to lose than to gain.

#### Lever 1 — subsection chunking: no effect whatsoever

The hypothesis recorded above — that s.35(1)(c) as its own chunk would be
"short, dense and beat s.78 outright" — was **wrong**, and it was written here
as a finding when it was only a guess. Section 35 was split into six
subsection chunks, dropped into the same 30 candidates, and reranked by the
unchanged cross-encoder. The top five came back **byte-identical** to before.
Not one piece reached contention.

The cross-encoder does not prefer s.78 because s.35 is long. It prefers it
because s.78 **resembles the question more**, and reshaping s.35 does not
change that. Chunking is not a reranking fix.

#### Lever 2 — an LLM judge: fixes one, breaks one, and the break is worse

Every answerable question run through the proposed production shape —
`retrieve 30 → cross-encoder narrows to 10 → gpt-4o-mini picks 5`:

```text
             current        judge
  hit@5       78/80         78/80        no change
  top1        54/80         66/80        +12
  FIXED       employment-notice
  BROKEN      arrest-bail
```

**`arrest-bail` is a legal error, not a near-miss.** Asked *"Can I be released
on bail while waiting for my trial?"*, the judge returned CPC **s.356**
(*bail pending the entering of an appeal*), **s.357** (*bail pending appeal*)
and **s.124** (*bail bond* — the mechanics), dropping **Article 49**, which the
cross-encoder had at **rank 1**. It also failed to pick **s.123 "Bail in
certain cases"**, the correct pre-trial provision, which was in front of it.

It answered a **pre-trial** question with **post-conviction** law. The same
result holds when judging all 30, so it is not a narrowing artifact.

**`penal-murder-sentence` was lost before the judge saw it.** s.204 is outside
the cross-encoder's top 10, so narrowing to 10 is too tight; given all 30 the
judge finds it at rank 1. But an all-30 design still **fixes 2 and breaks 1**,
which fails the rule.

#### The conclusion, which is not the one expected

The judge's error is **the same class as the cross-encoder's** — matching on
"bail" without distinguishing the stage of proceedings, exactly as the
cross-encoder matched "notify/termination" without distinguishing who is
notified.

The earlier framing here — *"a 568M relatedness model has no representation of
answerhood"* — was too flattering to the alternative. `gpt-4o-mini` does have
that representation and **still** makes stage-confusion errors on Kenyan
criminal procedure. **The limit is legal domain knowledge, not model class.**

`top1` +12 is a genuine gain and was not bought, because `hit@5` stayed flat —
the judge only reordered what was already reachable — and the price was one
wrong-stage bail answer.

#### Do not tune the prompt

The obvious next move is an instruction about procedural stage, then re-run
the 80. That is **prompt-fitting to the test set**: a prompt that works on
these 80 questions and generalises to nothing — the hand-listed regex router's
failure, one level up. One break in 80 is not enough signal to tune against.

#### Standing position

**Carry the two failures.** They are false refusals on questions the corpus
answers — bad for usefulness, and exactly the failure mode
`accuracy > grounding > usefulness` accepts. Revisit only with a materially
stronger judge, not a differently-worded one, and re-run this experiment
rather than reasoning about it.

The experiment cost about twenty cents and stopped a build that would have
shipped appeal-stage bail law to someone asking about pre-trial release.
**Measure the lever before pulling it.**

### Measured: at eight documents the reranker earns MORE than it did at four

Re-run of `--no-rerank` over the complete corpus, because the previous
measurement was two documents and a thousand chunks out of date, and twelve
failures made it reasonable to suspect the cross-encoder had turned
net-harmful. **It has not. The opposite.**

| | with reranking | retrieval order only |
|---|---|---|
| questions passing | **121/133** | 110/133 |
| rerank_hit | **106/118** | 95/118 |
| rerank_top1 | **74/118** | 62/118 |

**Thirteen rescues against two breaks — net +11 questions.** At four documents
the contribution was +8. It grows with the corpus, because a bigger corpus
buries the right passage deeper under plausible statutory detail, and digging
it out is precisely what the cross-encoder does.

What it rescues is the documented pattern, now at scale:

```text
arrest-reason        Article 49 at #10, behind five CPC arrest sections
arrest-court-time    Article 49 at #18
arrest-bail          Article 49 at #26
privacy-search       Article 31 at #11, behind CPC search provisions
unfair-government-   Articles 22/47 at #12
```

Without it, *"Can police arrest me without telling me why?"* returns five
Criminal Procedure Code sections and **no constitutional right at all**.

What it breaks is the two rank-#1 pathologies. Real, and the sharpest
statement of the defect available — but 2 against 13. **The reranker stays,
and the case for it is stronger than when it was first justified.**

---

### Measured: tier 0 is NOT a sufficient regression canary

Recorded because the opposite was asserted in this project, acted on, and
proved wrong within a day.

The reasoning was: retrieval regressions are the likely failure when a
document is added, tier 0 catches those in seconds, so tier 0 after each
document plus one full tier 1 at the end is a safe way to save hours.

`county-government-role` disproves it. **Tier 0 passed the question** — the
Fourth Schedule was retrieved at #6 of 30, comfortably inside the budget. Only
tier 1 revealed the failure: the Children Act's Part VI, *Role of County
Governments*, now reranks **above** it, and the expected authority is dropped
from the top five.

**Cross-document rerank displacement is invisible to tier 0 by construction.**
Tier 0 asks whether the authority is in the candidate pool; displacement
happens after that, when a newly-added document's sections out-score it. The
two are different questions and only one of them is cheap.

Tier 0 keeps its value — it catches breadth failures in seconds and it caught
both of those at eight documents. But **it does not substitute for tier 1
when a document is added.** If the full run is deferred, the risk being
accepted is displacement, not breadth, and it should be named as such.

---

### Measured: RETRIEVAL_TOP_K stays at 30 at 1,564 chunks — the arithmetic says so

Completing the corpus produced the first retrieval regressions since 591
chunks. Tier 0 at eight documents caught two `NEVER RETRIEVED` failures in
seconds — `state-funded-lawyer` and `divorce-process` — which is precisely
what tier 0 exists for, and it cost nothing.

`RETRIEVAL_TOP_K` was 15 at 279 chunks and 30 at 591. At 1,564 it had not
moved, so raising it was the obvious hypothesis. It was measured rather than
assumed, and the answer is **no**.

#### A hypothesis that was wrong, and worth recording as such

Inspecting the divorce query showed **13 of its 30 slots** going to chunks
that cannot answer a citizen question: seven Constitution schedules, six
unrelated articles (the swearing-in of the Deputy President, electoral
disputes, the Cabinet). The obvious conclusion was that schedules were eating
the retrieval budget and that curating them would recover the slots.

Measured across all 118 answerable questions, Constitution schedules other
than the Fourth occupy **0.5 slots per question — 1.8% of the budget.** The
divorce query was an outlier. There is nothing meaningful to reclaim, and
generalising from one query was the error.

#### The sweep

Each value retrieved for real rather than inferred from one deep run, because
RRF fuses a dense prefetch of `limit=top_k` with a sparse one of
`limit=min(5, top_k)`, so a deeper dense list can reorder the fused result.

| k | retrieval hit | recovers | rerank pairs/question |
|---|---|---|---|
| 30 | 116/118 | — | 30 |
| 40 | 116/118 | nothing | 40 |
| **50** | **117/118** | `state-funded-lawyer` (rank 42) | 50 |
| 60 | 117/118 | nothing | 60 |
| **75** | **118/118** | `divorce-process` (rank 74) | 75 |
| 100 | 118/118 | nothing | 100 |

No deeper k ever lost a hit. Only 50 and 75 are meaningful steps.

#### Recall is not the measurement that matters

`retrieval_hit` is necessary, not sufficient — a passage entering at rank 74
of 75 still has to survive the cross-encoder into the top 5. Reranking the two
questions end to end at each k:

```text
question                k=30     k=50     k=75
state-funded-lawyer     fail     PASS     PASS
divorce-process         fail     fail     fail
```

**`state-funded-lawyer` is genuinely a breadth problem.** Article 50 enters at
retrieval rank 42 and the cross-encoder promotes it to **rank 1**. Useful in
its own right: the reranker demonstrably rescues from deep in the candidate
list, which had been doubted.

**`divorce-process` is not.** It fails at every k. Even entering at rank 74,
s.66 never reaches the top 5 — the same three registration and
foreign-recognition sections win every time, because the Act says
*"dissolution"* where the question says *"divorce"* and "divorce" appears
literally in the registration sections' titles. This is the **sixth** instance
of the defect in *Measured: the cross-encoder ranks by resemblance*.

#### k=75 is strictly dominated

It fixes exactly what k=50 fixes at 2.5× instead of 1.7×, and it *degrades*
the fix: `state-funded-lawyer` reranks to #1 at k=50 but falls to **#3** at
k=75, displaced by Article 260 *Interpretation* and s.4 *Interpretation* —
generic definitional sections that rerank well against anything. **Deeper
retrieval is not monotonically better.**

#### The decision

Stay at **30**. The trade is **one question in 118** against **+67% on
reranking, which is the dominant request latency** — every user waiting longer
on every question, permanently, and every future evaluation costing 70% more.
The failure mode being bought off is a *refusal*, not a wrong answer, which is
the failure `accuracy > grounding > usefulness` accepts.

There is also an unmeasured displacement risk: 20 additional candidates could
knock out questions that currently pass, and measuring that means a full run
at k=50.

**Revisit when the corpus grows again** — the pattern is real, just not yet
worth paying for. And measure end to end, not on `retrieval_hit`: this is the
case that shows a question can be retrieved and still fail.

---

### Measured: rewrite the question — and let the reranker read the rewrite

Adopted 2026-10-12 (`QUERY_REWRITE` on). Recorded because the first half alone
looked like a success at tier 0 and was worth nothing at tier 1.

**The problem.** People do not use the statute's words. *"Is torture allowed
in Kenya?"* retrieves Article 29 at rank 1; *"The police beat me while
questioning me. Is that legal?"* did not retrieve it in the top 100. The
statutory question set hid this — 35% of its questions restate the heading of
the section they expect, and `passage.py` embeds those headings. A product
question set, 32 questions in people's own words, was built to measure it.

**Phase B — rewrite the question for retrieval.** `rewrite.py` asks the
answering model for up to three search phrases in legal register. It
translates; it may not answer, name a provision, or state a number — the last
two enforced in code, because reading all 165 first-prompt rewrites found about
a dozen carrying recalled law (*"not less than twenty-one working days after
twelve consecutive months"* for an annual-leave question). Tightening the
prompt cost nothing: tier 0 went statutory **116 → 118/118**, product
**19 → 22/22**. Each rewrite is one extra dense branch in the same fused Qdrant
query, and the question always stays in the fusion.

**But at tier 1 Phase B alone gained nothing: product 15 → 15/22**, one
question fixed and one broken. The right law now reached the candidates — CPC
s.177 at retrieval **#1** for *"The police took my phone and laptop"* — and the
cross-encoder, scoring passages against the lay question, threw it away. The
break was a displacement: the rewrites brought Article 69 into the pool, the
reranker preferred it, and Article 70 — the article that actually gives a
person a remedy — lost the Constitution's place.

**Phase C — the reranker reads the question followed by the rewrites**
(`rerank_query`). Measured against the same baselines:

| | before | Phase B + C |
|---|---|---|
| statutory, reach the LLM | 106/118 | **115/118** |
| statutory, first place | 74/118 | **93/118** |
| statutory, passing | 121/133 | **130/133** |
| product, reach the LLM | 15/22 | **20/22** |
| product, tier 2 passing | 19/32 | **23 and 22/32**, two runs |

No regressions anywhere. The 13 questions the reranker rescues all still pass,
and their first-place count rose 4 → 8. Tier 2 kept **0 false answers** in both
runs and wrong-authority answers fell 1 → 0.

**This corrects the resemblance finding above.** It held that the limit was
legal domain knowledge rather than the model. Phase C fixed five of those
failures — `penal-murder-sentence`, `succession-left-out-of-will`,
`divorce-process`, and the two retrieved at #1 and discarded,
`smallclaims-lawyer` and `children-parental-responsibility`. For many of them
the limit was the vocabulary the reranker was handed. The defect itself is
real: **`employment-notice` survives Phase C** with s.36 and s.35 at retrieval
#1 and #2 and s.78 still first.

**What it costs.** One extra model call per question for the rewrite, and a
longer reranker input. Not yet measured in serving. Serving stores no
rewrites; the evaluation's are committed in `evaluation/rewrites/`, because
the rewriter varies between calls and deleting the store changes results.

**What it does not fix.** Tier 2 is now limited by the answering model, not by
retrieval: in 7 of the 10 failures the right law reached it and it refused,
because the sources did not answer the WHOLE question.

**Found along the way.** Qdrant's fusion scores 1/(position + 1) in each list,
so candidates tie exactly and came back in no fixed order between identical
runs. The retriever now breaks ties by chunk id.

---

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

Revised 2026-09-23. The ordering changed for two reasons: the ingestion layer must be safe before the corpus grows, and nothing downstream can be measured without an evaluation set.

```text
1. Ingestion foundation                  ← DONE
   Generic, idempotent, registry-driven pipeline.
   Stable IDs, batched ingest, central config,
   versioning fields, named vectors.

2. Evaluation set                        ← DONE
   96 questions, three tiers, baseline recorded.
   Question set still needs legal review - and the
   county-government-role correction shows why.

3. Corpus expansion                      ← 5 of 8 ingested, 8 of 8 written
   CPC 2026-09-03, Penal Code 2026-09-19, Employment
   Act and Law of Succession 2026-09-23. Children,
   Marriage and Small Claims are registered and
   dry-run verified, awaiting ingestion. Scope is
   FIXED at eight - see "Current position".

3b. Evaluation scoping                   ← DONE
   Tier 0, --document, two-level cache, batched
   reranking. Caveat: the rerank cache reused
   nothing when document four landed.

4. Evidence sufficiency                  ← OPEN, hardest item
   NOT an absolute rerank threshold - that was tried and
   removed; see "rerank scores are not a confidence signal".
   A subject-matter gate was also tried and removed.
   debt-inability's false answer is still open.

5. FastAPI service                       ← DONE
   Models load once at startup; semantic intent routing;
   structured answer + sources. No state, no auth.

5b. Reranker intervention                ← CLOSED, measured, not built
   Both remaining failures are one defect: the
   cross-encoder ranks by resemblance, not answerhood.
   Both proposed levers were measured 2026-09-23 and
   BOTH FAILED. Subsection chunking changed nothing;
   an LLM judge fixed one question and answered a
   pre-trial bail question with post-conviction law.
   Carry the two failures. Reopen only with a
   materially stronger judge, and re-run the
   experiment rather than reasoning about it.

6. Scenario understanding
   Fact extraction, legal-issue identification, multi-query.

7. Reasoning and structured citations
   fact → rule → application → conclusion.

8. Application layer + Flutter
   Conversations, persistence, auth, feedback - none of
   which exist. Four documents and a stateless endpoint
   is not yet a product.
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
- **2026-09-23** — **The MVP corpus is complete and baselined: eight
  documents, 1,564 chunks, 121/133.** Ingested the Children Act (249 sections,
  258 chunks), Marriage Act (97/97) and Small Claims Court Act (52/52), every
  count matching its dry run. 253 tests. Baseline promoted over 133 questions:
  retrieval **116/118**, rerank_hit **106/118**, rerank_top1 **74/118**.
  `divorce-process` — the original refusal test case — is answerable at last,
  though it does not yet pass.
  Three measurements, all recorded above. **(1)** `RETRIEVAL_TOP_K` stays at
  30: the sweep showed k=50 fixes exactly one question of 118 and k=75 is
  strictly dominated, against +67% on the dominant request latency. The
  hypothesis that Constitution schedules were eating the budget was measured
  and **wrong** — 1.8%, not the 23% one query suggested. **(2)** The reranker
  now contributes **+11 questions** (13 rescues, 2 breaks), up from +8 at four
  documents; the suspicion that it had turned net-harmful was wrong.
  **(3)** **Tier 0 is not a sufficient canary** — `county-government-role`
  regressed through cross-document rerank displacement (the Children Act's
  *Role of County Governments* outranking the Fourth Schedule) and tier 0
  passed it. That correction matters: an earlier plan to run tier 0 per
  document and tier 1 only at the end would have missed this.
  Twelve failures now sort into four classes, and the cross-encoder's
  resemblance defect accounts for nine of them, including two questions whose
  answer is retrieved **first** and discarded.
- **2026-09-23** — **Documents five to eight: the corpus is written.** Ingested
  the **Law of Succession Act (Cap. 160)** — 101 sections, 103 chunks,
  collection 1054 → 1157 — and prepared the **Children Act (Cap. 141)**,
  **Marriage Act (Cap. 150)** and **Small Claims Court Act (Cap. 10A)**: 258,
  97 and 52 chunks, all dry-run verified against their validators but not yet
  ingested. Question set 112 → **133**, 118 answerable. `writing-a-will`
  flipped to answerable with document five; **`divorce-process` flips with
  document seven — the original refusal test case**, correctly declined over a
  Constitution-only corpus in the first Phase 3 tests and out of scope ever
  since.
  Four more silent parser failures, all found by per-document validators and
  all recorded in *Measured: Kenya Law is not internally consistent*. The
  worst by a wide margin: the Children Act's section 218 ends *"...in
  accordance with the Fifth Schedule."*, the line wrapped, and a
  case-insensitive prefix match read the fragment as a schedule heading —
  **stopping the parse and discarding sections 219 to 250**, the whole of Part
  XV, Children in Conflict with the Law. Fixed by requiring what a real
  heading looks like: uppercase, whole-line, optional bracketed reference.
  Also: the Law of Succession has **no Part I**, so sections 1 to 4 sit above
  the first Part and now get a one-level citation path rather than a
  synthesised one; the Marriage Act's lone schedule is headed bare `SCHEDULE`
  so the cleaner cuts it, and its apparent duplicate section 50 is a wrapped
  cross-reference, recorded rather than suppressed. Every change verified
  non-destructive — all previously ingested Acts reparse to byte-identical
  chunk ids. Tier 0 at five documents: **retrieval 95/95**.
- **2026-09-23** — **Measured both proposed reranker fixes. Built neither.**
  The two remaining failures are one defect — the cross-encoder ranking by
  resemblance rather than answerhood — and two levers had been recorded as
  the way out. Both were tested against the real corpus before any production
  code, under a rule set in advance: *fix something, break nothing*.
  **Subsection chunking changed nothing** — splitting section 35 into six
  subsection chunks left the reranked top five byte-identical, disproving a
  hypothesis that had been written into these docs as a finding. **An LLM
  judge fixed one and broke one**: `hit@5` flat at 78/80, `top1` 54 → 66, but
  asked *"Can I be released on bail while waiting for my trial?"* it returned
  Criminal Procedure Code s.356 and s.357 — both *pending appeal* — plus the
  bail-bond mechanics, dropping Article 49 which the cross-encoder had at rank
  1, and missing s.123 entirely. **Pre-trial question, post-conviction law.**
  Consistent when judging all 30, so not a narrowing artifact. The
  unexpected conclusion: the judge makes the *same class* of error as the
  cross-encoder, so the limit is legal domain knowledge, not model class.
  Declined to tune the prompt — one break in 80 is prompt-fitting to the test
  set. Two failures carried; both are false refusals, the failure mode the
  priority order accepts. Cost about twenty cents, and prevented shipping
  appeal-stage bail law to someone asking about pre-trial release.
- **2026-09-23** — **Fourth document, and a failing test that was not a
  failing system.** Ingested the **Employment Act (Cap. 226)**, the first
  civil document in the corpus: 45 pages, 91 live sections, 92 chunks,
  collection **962 → 1054**. Question set 82 → **96**; `employment-notice`
  flipped from out-of-scope to answerable. Tier 0 reported `retrieval_hit`
  **80/80 with zero regressions** — the expected displacement of
  constitutional articles by ordinary employment vocabulary did not happen,
  and `RETRIEVAL_TOP_K = 30` holds. Tier 1: **93/96**, rerank 77/80,
  first-place 54/80.
  Three findings, all recorded in full above. **(1)** `act.py` was silently
  dropping sections 1–3: the Act numbers its first Part in Arabic
  (`Part 1`) and the rest in Roman, and a Roman-only pattern did not
  mislabel it but failed to match, discarding every section until Part II —
  including section 2's definitions. Fixed with one alternation; the
  validator now checks those sections by name. **(2)** The level-2 rerank
  cache **saved nothing** — 0 reused, 96 computed — because adding 92 chunks
  moved the top-30 for all 82 existing questions, contradicting its own
  design note; assume a full rerank per document until measured otherwise.
  **(3)** `county-government-role`, failing since the harness was built, was
  a **question defect, not a system defect**: Article 186 does not list
  county functions, it points to the Fourth Schedule, which reranking had
  been surfacing correctly all along. Expectations corrected, Articles 183
  and 174 removed. Failures 3 → 2, and both survivors are one defect —
  the cross-encoder ranking by resemblance rather than answerhood, now seen
  in two unrelated documents with a fix direction and a cheap experiment to
  choose between levers. Also fixed `--id` runs printing a meaningless
  baseline comparison and exiting 1 on a phantom regression. MVP scope fixed
  at **eight documents**. 234 tests.
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
- **2026-10-12** — **Query rewriting adopted: the right law now reaches the
  answering model for 20 of 22 questions asked in people's own words, up from
  15.** Since 2026-09-26: answers moved to `gpt-6-luna` on the project's own
  account (the judge test: it caught 5 of 5 overstatements where gpt-4o-mini
  caught 1); the judge checks claim by claim; advice-shaped questions are
  answered as information with a fixed advocate note; baselines are per suite
  and per tier and only complete, uncached runs can be promoted. Then Phases B
  and C — see *Measured: rewrite the question — and let the reranker read the
  rewrite*: statutory 121 → 130/133 at tier 1, product tier 2 19 → 22–23/32,
  no regressions, 0 false answers. Baselines to be re-recorded with rewriting
  on. The next limit is the answering model refusing questions the sources only
  partly answer.
