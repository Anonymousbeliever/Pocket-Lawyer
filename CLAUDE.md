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

Retrieval → reranking → grounded generation works end to end over **four
documents, 1054 chunks**: the Constitution of Kenya, 2010 (279), the Criminal
Procedure Code, Cap. 75 (312), the Penal Code, Cap. 63 (371) and the
Employment Act, Cap. 226 (92).

**MVP scope is fixed at eight documents.** Still to come: the Law of Succession
(Cap. 160), the Children Act 2022, the Marriage Act 2014 and the Small Claims
Court Act 2016. Do not propose documents outside that set without saying it
expands agreed scope — `docs/project-plan.md` records what was cut and why.

**96 evaluation questions, 93/96 passing, retrieval 80/80.**

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
  retriever.py             LegalRetriever  — BGE-M3 → Qdrant, top 30,
                           dense + lexical fused with RRF
  reranker.py              LegalReranker   — cross-encoder, top 30 → top 5,
                           capped so one document cannot take every slot
  llm.py                   LegalLLM        — gpt-4o-mini + grounding prompt
  rag.py                   LegalRAG.answer() — chains retrieval/rerank/LLM
  generator.py, prompt.py  EMPTY placeholders, unused
backend/app/core/config.py Single source of truth for config - BOTH sides import it
backend/app/core/sparse.py LexicalEncoder — BGE-M3's sparse head on the
                           ALREADY-LOADED model; no second model
data_pipeline/
  run.py                   orchestrator: extract → clean → structure → chunk → store
  documents.yaml           document registry - add a document by adding an entry
  ir.py                    Document/Unit - the common shape every stage works on
  registry.py              loads the registry, derives artifact paths
  adapters.py              the type registry - register a new document HERE only
  structure/<type>.py      per-TYPE and reusable: constitution.py, act.py
                           act.py handles Part -> Section (CPC, Employment)
                           and Part -> Chapter -> Section (Penal Code);
                           Part numerals may be Roman OR Arabic
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
  Act reuses it. **But expect small pattern surprises per document.** The
  Employment Act numbers its first Part in Arabic (`Part 1 – PRELIMINARY`) and
  the rest in Roman. The Roman-only pattern did not mislabel it — it failed to
  match, so the parser kept treating the text as the Act's opening block and
  **dropped sections 1 to 3 in silence**, definitions included. Kenya Law is
  not internally consistent, and the failures are quiet.
- **a validator** (`validators/<document>.py`) — in practice always, because
  without one a parse failure is silent: a broken pattern yields zero units and
  the only complaint is "No chunks were generated." Check the document's
  specific risk **by name**, not only by count: a missing Part 1 shows up as a
  section total three short, which nobody notices.

## Running things

```bash
docker compose up -d              # Qdrant on localhost:6333
pytest                            # 234 tests, no Qdrant or API key needed

python -m uvicorn backend.app.main:app --reload   # the API, models load once
python -m backend.app.ai.intent                   # routing accuracy + margins

python -m data_pipeline.run --document constitution-of-kenya-2010
python -m data_pipeline.run --document criminal-procedure-code
python -m data_pipeline.run --document employment-act
python -m data_pipeline.run --all
python -m data_pipeline.run --document X --from chunk   # resume a stage
python -m data_pipeline.run --document X --to chunk     # stop before Qdrant
python -m data_pipeline.run --document X --recreate     # rebuild collection

python -m evaluation.run --tier 0    # retrieval only, seconds, no reranker
python -m evaluation.run             # tier 1: + reranking, free, no API key
python -m evaluation.run --tier 2    # adds the LLM (refusal correctness)
python -m evaluation.run --document penal-code   # only that document's questions
python -m evaluation.run --no-cache  # recompute; still writes results
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
  `--id` and `--document` are the scoping tools, and `--id` runs are
  deliberately excluded from `--save-baseline` and from `last_run.json`.
- **Know which tier a change actually needs.** A full tier 1 run is warranted
  only when something can move a *retrieval decision*: `retriever.py`,
  `reranker.py`, `passage.py`, the retrieval settings in `config.py`, anything
  in `data_pipeline/` that changes chunk ids or text, a new document, or the
  embedding model. It is **not** warranted for the API layer, routing, schemas,
  tests or docs - the harness imports none of them. `llm.py` and `answer.py`
  are tier 2 territory; tier 1 never calls the LLM.
- **Results are cached, and the key is what makes that safe.** Level 1
  (retrieval) is keyed on the corpus; level 2 (reranking) is keyed on the
  *retrieved candidates*, deliberately **not** on the corpus - so adding a
  document reranks only the questions whose top-k actually moved. Both keys
  include a hash of the source of `retriever.py`, `reranker.py` and
  `passage.py`, so editing any of them invalidates everything downstream.
  It cannot see a library upgrade: use `--no-cache` for that, which recomputes
  while still writing. **A baseline is never served from cache** -
  `--accept-last` refuses a run with cache hits, and `--id` and `--no-rerank`
  runs never become the record.
  **Measured, and it did not work as designed:** adding the Employment Act
  reused **0 of 96** rerank results. Its 92 chunks moved the top-30 for all 82
  existing questions - plausibly the sparse side, since ordinary employment
  English enters the lexical top-5 for unrelated questions and perturbs the
  fused list everywhere. Hybrid retrieval and candidate-keyed caching work
  against each other. **Budget a full rerank per new document** until a
  measurement says otherwise.
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
- **Retrieval is hybrid, and the two budgets are deliberately unequal.**
  Dense encodes meaning and handles paraphrase; it blurs exact terms. Asked
  *"What is the sentence for murder in Kenya?"*, section 204 — eleven words
  answering it almost verbatim — sat at **rank 53**, behind *"Conspiracy to
  murder"*, which is longer and shares more vocabulary. Lexical matching on
  `murder` and `sentence` is what rescues it. The two are fused with
  **Reciprocal Rank Fusion**, which combines *ranks* not scores — the same
  reason rerank floors do not work applies to reconciling cosine against dot
  product. `SPARSE_TOP_K` is 5 against `RETRIEVAL_TOP_K` 30 because lexical
  is a **corrective, not an equal partner**: at 20+ the Criminal Procedure
  Code's many sections containing the literal word "bail" pushed Article 49
  out of range entirely. Erring low merely fails to rescue; erring high
  displaces constitutional rights with statutory detail. Full sweep in the
  config comment.
- **Do not dilute the reranker's authority over ordering.** Measured with
  `--no-rerank`: without the cross-encoder, `rerank_hit` drops 65 → 57 and
  `rerank_top1` 44 → 37. All eight rescues are the same pattern — retrieval
  fills the top with Criminal Procedure Code sections while the
  **constitutional right sits buried**, Article 49 at rank 26 for
  `arrest-bail`, rank 18 for `arrest-court-time` — and the reranker plus
  per-document diversity digs it out. A planned change to fuse rerank order
  with retrieval order by RRF was **abandoned** because of this: it would
  weaken exactly that override, trading eight rescues for two. If the two
  remaining failures are ever worth fixing, it needs a different
  cross-encoder, not a blend.
- **The cross-encoder ranks by resemblance, not by answerhood — and no
  tuning fixes that.** Both remaining evaluation failures are this one defect,
  in two unrelated documents. *"How much notice must my employer give before
  terminating my job?"* ranks **s.78** first — *"An employer shall notify the
  termination of every employment... to the nearest employment service office
  **within two weeks**"*, 192 characters, matching every content word and
  offering a duration — over **s.35 "Termination notice"**, 2,166 characters
  whose rule ("**twenty-eight days**") is buried in subsection (1)(c). s.78 is
  an administrative duty to notify a government office, not notice to the
  employee. Likewise s.204's eleven words lose to *"Conspiracy to murder"*.
  This is **not** a plumbing bug: `passage.py` was verified to put
  `"Section 35: Termination notice"` in front of the reranker. The model reads
  the header and still prefers the resemblance. You cannot reach the missing
  capability with a threshold (tried, removed), a blend (tried, abandoned) or
  a per-document boost (the enumeration trap).
  **Two levers were proposed and BOTH were measured on 2026-09-23. Neither is
  worth building. Do not re-propose either without re-running the experiment.**
  *Subsection chunking changed nothing* - splitting s.35 into six subsection
  chunks left the reranked top five byte-identical, so the problem is not that
  s.35 is long. *An LLM judge* (`gpt-4o-mini` over the candidates) fixed
  `employment-notice` and **broke `arrest-bail`**, answering a **pre-trial**
  bail question with **post-conviction** law - CPC s.356 and s.357, both
  *pending appeal* - while dropping Article 49, which the cross-encoder had at
  rank 1. `hit@5` stayed flat at 78/80; only `top1` improved, 54 → 66.
  The lesson is the one that was not expected: the judge makes the **same
  class of error** as the cross-encoder. **The limit is legal domain
  knowledge, not model class.** Do not tune the judge's prompt to rescue
  `arrest-bail` either - one break in 80 is prompt-fitting to the test set.
  Full numbers in `docs/project-plan.md`. The grounding prompt holds
  meanwhile: tier 2 gave `false answers 0`, the system **refused** rather than
  repeating s.78's confident wrong number, so these are false refusals - the
  failure mode `accuracy > grounding > usefulness` accepts.
- **A failing test is not automatically a failing system.**
  `county-government-role` failed for eighteen runs. The cause was its
  expectations: Article 186 does not list county functions - its operative
  text is a pointer, *"as set out in the Fourth Schedule"* - and reranking had
  been surfacing the Schedule correctly the whole time. Articles 183 (county
  *executive committee* functions) and 174 (objects of devolution) were
  removed as not answering the question. **Before tuning retrieval against a
  failing question, read the law and check the expectation is right.** The
  question set was drafted against article titles, not full text, and still
  awaits legal review.
- **`sparse.py` exists instead of a dependency.** `sentence_transformers`
  will not give you BGE-M3's lexical weights — asking its `SparseEncoder`
  for the model silently converts it to a generic 4096-dim projection that
  is not term-based. FlagEmbedding exposes the real head but loads a second
  2.2 GB copy of the model. The head is `Linear(1024, 1)`, so it runs on the
  token embeddings of the model already in memory.
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
