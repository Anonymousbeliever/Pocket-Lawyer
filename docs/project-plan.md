# Pocket Lawyer Kenya — Master Project Plan

This is the living project plan and progress tracker.

Update this file when a task starts, finishes, or the current position changes. Source inventory lives in [`legal-data/sources.md`](legal-data/sources.md).

**Last updated:** 2026-08-23

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

That behavior is a feature.

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

---

## Current position

**Phase 1 — Legal Data Foundation** (in progress)

```text
Phase 1
│
├── Identify authoritative sources     ✓ (hierarchy defined; official portals still to investigate)
├── Build source inventory             ← next after quality inspection
├── Collect initial corpus             ✓ Constitution collected
├── Extract document text              ✓ first Constitution extract exists
├── Clean extracted text               ✓ first cleaner exists; quality still needs inspection
├── Process document metadata
├── Legal structure detection
└── Build ingestion pipeline
```

We started with one document on purpose: understand the data before processing thousands of documents.

**Next:** inspect extraction and cleaning quality for the Constitution, then metadata and legal-structure processing around the real output.

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

The backend is a minimal FastAPI app (`backend/app/main.py`). Environment placeholders exist in `.env.example` (PostgreSQL, Qdrant, LLM). Flutter is an empty `mobile/` folder.

---

## Phase 1 — Legal Data Foundation

**Status:** In progress — **current phase**

**Objective:** Build the trusted legal corpus that will eventually become the foundation of the AI.

This is one of the most important phases of the entire project.

- [x] Identify authoritative sources
- [ ] Build source inventory
- [x] Collect initial corpus
- [x] Extract document text
- [x] Clean extracted text
- [ ] Process document metadata
- [ ] Legal structure detection
- [ ] Build ingestion pipeline

### Notes

- Source hierarchy and collection principles: [`legal-data/sources.md`](legal-data/sources.md).
- Initial corpus: Constitution of Kenya (`data/raw/constitution/Constitution of Kenya.pdf`).
- Extractor: `data-pipeline/extractors/pdf_extractor.py` → `data/extracted/constitution.txt`.
- Cleaner: `data-pipeline/cleaners/constitution_cleaner.py` → `data/cleaned/constitution_cleaned.txt`.
- Cleaning quality still needs inspection before we treat it as final.
- Collectors, validators, OCR, chunkers, embeddings, and orchestrated pipelines are empty folders.
- Kenya Law, Judiciary, Supreme Court, and Parliament are listed as future sources, not yet investigated.

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

We are not running this end-to-end yet. We are proving each step against the Constitution first.

---

## Phase 2 — Legal Knowledge Base

**Status:** Not started

**Objective:** Turn the processed legal corpus into a searchable knowledge base.

- [ ] PostgreSQL
- [ ] Qdrant
- [ ] Embeddings
- [ ] Legal document chunking
- [ ] Retrieval

### PostgreSQL

Structured document metadata, and later application data (users, conversations):

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

### Qdrant

Vector search so meaning can match even when wording differs.

Example: a user asks *"Can cops arrest me without explaining why?"* while the legal text says *"Every arrested person has the right to be informed promptly..."*.

### Embeddings

Legal chunks become vectors. Qdrant uses those vectors to find semantically relevant passages. Embedding model is not chosen yet.

---

## Phase 3 — AI / RAG

**Status:** Not started

**Objective:** Build the AI that answers questions using the legal knowledge base.

- [ ] First RAG pipeline
- [ ] Reranking
- [ ] Citation generation
- [ ] Citation verification
- [ ] Evaluation

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

---

## Phase 4 — Application Backend

**Status:** Not started

**Objective:** Turn the RAG system into a proper application backend.

- [ ] Authentication
- [ ] Conversations
- [ ] Chat API
- [ ] Source API
- [ ] Feedback

Exact API design is deferred until this phase. Illustrative endpoints only:

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

The UI comes after the intelligence underneath it works. The initial interface can follow a familiar chat pattern (ChatGPT-like) because users already understand that model. Screen design is deferred.

---

## Data automation (later)

Out of scope until the Python ingestion pipeline exists.

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
PHASE 0  Foundation
    ↓
PHASE 1  Understand + process Kenyan legal data     ← we are here
    ↓
PHASE 2  Build searchable legal knowledge base
    ↓
PHASE 3  Build reliable RAG / AI
    ↓
PHASE 4  Expose it through FastAPI
    ↓
PHASE 5  Build Flutter application
```

---

## Progress log

- **2026-08-23** — Recorded the master plan. Phase 0 marked done (minimal). Phase 1 marked in progress: Constitution collected, extracted, and cleaned; quality inspection is next.
