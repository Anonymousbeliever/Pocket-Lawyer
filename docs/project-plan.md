# Pocket Lawyer Kenya — Master Project Plan

This is the living project plan and progress tracker.

Update this file when a task starts, finishes, or the current position changes. Source inventory lives in [`legal-data/sources.md`](legal-data/sources.md).

**Last updated:** 2026-08-31

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

**Phase 2 and Phase 3 are proven end-to-end on a single document.**

The full chain works today:

```text
Question
   ↓
BGE-M3 embedding
   ↓
Qdrant (270 vectors, Constitution only)
   ↓
Top 15 candidates
   ↓
BGE reranker
   ↓
Top 5 sources
   ↓
gpt-4o-mini (strict grounding prompt)
   ↓
Grounded answer + sources
```

Run it with `python -m backend.app.ai.rag`.

**Two things now limit the product, and they are different problems:**

1. **Corpus breadth.** Only the Constitution is indexed. Questions outside it correctly return "I don't have enough reliable information" — the right behaviour, but not yet a useful product.
2. **The ingestion layer is single-document code.** It cannot safely take a second document as written. See *Known limitations and blockers*.

**Next:** the ingestion-foundation refactor (a generic, idempotent, registry-driven pipeline). That work must land before the corpus grows, because it changes point IDs and the collection schema and therefore requires a full re-ingest either way.

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

**Status:** In progress — one item remaining

**Objective:** Build the trusted legal corpus that will eventually become the foundation of the AI.

- [x] Identify authoritative sources
- [ ] Build source inventory
- [x] Collect initial corpus
- [x] Extract document text
- [x] Clean extracted text
- [x] Process document metadata
- [x] Legal structure detection
- [ ] Build ingestion pipeline

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

**Status:** Done for the first document — corpus breadth outstanding

**Objective:** Turn the processed legal corpus into a searchable knowledge base.

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
- [ ] Citation generation
- [ ] Citation verification
- [ ] Evaluation

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

### Known response-quality issues

Both reproduced again on 2026-08-31 after the ingestion refactor. Neither is a
retrieval or grounding failure — the correct article was retrieved and cited
both times. They are *response layer* problems.

**1. Question polarity.** For *"Can police arrest me without telling me why?"*
the system answered *"Yes, the police cannot arrest you without telling you
why."* — self-contradictory. It addressed the embedded proposition rather than
the yes/no question. The natural answer is **"No."**

Polar questions with a negative framing require the model to extract the rule,
map it to yes/no, then *invert*. Free-form generation gives it no reason to
commit to a polarity before writing prose.

*Fix — prevention, not detection:* schema-constrained output with `verdict` as a
discrete enum field the model must fill before writing any explanation, and the
final sentence rendered by code rather than the LLM. A contradiction then
becomes structurally impossible. A second verification call is the wrong tool
here: it detects after the fact, and doubles latency and cost.

**2. Claim fidelity — the more serious of the two.** The same answer concluded
*"it's legally required for the police to inform you of the reason for your
arrest **at the time of arrest**"*. Article 49(1)(a) says **"promptly"**. The
model strengthened the law beyond its source. "Promptly" and "at the moment of
arrest" are different legal standards.

The citation was correct and no source was invented, so grounding held — but
fidelity slipped, and this is the exact drift warned about above.

*Fix — detection, because schemas cannot prevent it:* a free-text explanation
can always overstate. This needs span-grounded citation (require the model to
quote the words it relies on) plus entailment checking of each claim against its
cited passage. That is the Citation verification item in this phase.

### Planned answer schema

Recording the contract now, because FastAPI and Flutter will both depend on it —
defining it after building an API around free text means rewriting both.

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

Beyond fixing polarity this earns two things:

- **The refusal stops being a magic string.** `sufficient: false` is a field to
  branch on, instead of the hardcoded "I don't have enough reliable
  information..." text currently duplicated between `llm.py` and `rag.py`.
- **Deterministic citation verification, identity half.** Code checks every
  `cited_chunk_ids` entry against the chunks actually retrieved. A fabricated
  citation is then caught with no LLM call and no cost. It does not verify that
  the claim matches the text — that still needs entailment checking — but it
  makes invented citations structurally impossible.

---

## Phase 4 — Application Backend

**Status:** Not started

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
- No `version` / `effective_date` / `in_force` in the Qdrant payload. The metadata JSON has them; the vectors do not. **Current law cannot be distinguished from repealed law.**
- Sub-paragraph hierarchy is flattened: `(i)`/`(ii)`/`(iii)` collapse into the parent paragraph. Text is intact and faithful, but pinpoint citation below the paragraph level is not addressable.
- No automated tests anywhere; `pytest` is not a dependency.
- `.env.example` still lists a non-existent `OPENAI_MODEL=gpt-5.6-luna`.

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

### Architectural gaps

- **Case law has no structural model.** Judgments are not chapters and articles. They need holdings vs obiter, court hierarchy, and overruled/distinguished status. The corpus plan includes case law; the pipeline has no concept of it.
- **No amendment/repeal lifecycle.** Answering with repealed law is the worst possible failure for a legal product. Answers should eventually carry an "as at" date.
- **Dense-only retrieval.** Legal queries need exact lexical anchors — "Section 45", "Cap 141", defined terms. BGE-M3 natively produces sparse vectors and Qdrant supports fusion; only dense is used today.
- **No evaluation harness.** Without a golden set of questions mapped to expected authorities, every future change is unfalsifiable.
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

2. Evaluation set
   40–60 golden questions → expected authorities.
   Cheap now; compounding value.

3. Corpus expansion
   Each new document validates the generic pipeline.

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
- **2026-08-31** — Attempted an absolute rerank score threshold as a sufficiency gate. It caused a false refusal on *"Can police arrest me without telling me why?"*, a question the corpus answers. Measurement showed the approach is unworkable, not merely miscalibrated — see *Measured: rerank scores are not a confidence signal*. Removed the floor; reranking now trims only relative to each query's own best match and can never refuse on its own. Sufficiency stays with the LLM's grounding prompt. Recorded the planned structured answer schema to fix question polarity by prevention rather than by a second verification call.
