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

Retrieval → reranking → grounded generation works end to end over the
**complete MVP corpus: eight documents, 1,564 chunks.**

| Document | Chunks |
|---|---|
| Constitution of Kenya, 2010 | 279 |
| Criminal Procedure Code (Cap. 75) | 312 |
| Penal Code (Cap. 63) | 371 |
| Children Act (Cap. 141) | 258 |
| Law of Succession Act (Cap. 160) | 103 |
| Marriage Act (Cap. 150) | 97 |
| Employment Act (Cap. 226) | 92 |
| Small Claims Court Act (Cap. 10A) | 52 |

**MVP corpus scope is CLOSED at these eight.** Do not propose another document
without saying it expands agreed scope — `docs/project-plan.md` records what
was cut and why.

**Baselined 2026-09-23: 133 questions, 121 passing** — retrieval 116/118,
rerank_hit 106/118, rerank_top1 74/118. The twelve failures sort into four
classes and `docs/project-plan.md` names each; **nine of the twelve are the
cross-encoder resemblance defect**, whose two proposed fixes were measured and
rejected. Two of those nine are retrieved at rank **#1** and discarded.

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
  rewrite.py               QueryRewriter   — Phase B: the question restated as
                           legal-register search phrases; never names a
                           provision; NOT yet wired into rag.py
  reranker.py              LegalReranker   — cross-encoder, top 30 → top 5,
                           capped so one document cannot take every slot
  llm.py                   LegalLLM        — OPENAI_MODEL (gpt-6-luna) + grounding
                           prompt; drops `temperature` if the model refuses it
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
                           act.py handles Part -> Section, Part -> Chapter
                           -> Section (Penal Code), and sections belonging
                           to NO Part (Law of Succession, which has no
                           Part I). Part numerals may be Roman OR Arabic.
                           Schedule headings must be UPPERCASE and whole-line
                           - see the wrapped-marker rule below
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
python -m data_pipeline.run --document law-of-succession-act
python -m data_pipeline.run --document children-act
python -m data_pipeline.run --document marriage-act
python -m data_pipeline.run --document small-claims-court-act
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
python -m evaluation.judge           # the answer judge vs judge_cases.yaml
python -m evaluation.run --tier 0 --rewrite   # Phase B: retrieve with rewrites
                                     # too; stored in evaluation/rewrites/,
                                     # which is COMMITTED - the rewriter
                                     # varies, so deleting it changes results

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
| Generation | `gpt-6-luna` (`OPENAI_MODEL` in `.env`) via the OpenAI **Responses** API (`client.responses.create`) |
| Answer judge (evaluation only) | follows `OPENAI_MODEL`, so `gpt-6-luna` for now; `--model` overrides it per run |
| Vector store | Qdrant (Docker, local) |
| Backend | FastAPI |
| Mobile (later) | Flutter |

**Do not silently change models.** The embedding model must be byte-identical at
ingest time and query time — if they drift apart, retrieval quality collapses
**with no error message**.

**Generation moved from `gpt-4o-mini` to `gpt-6-luna` - decided 2026-10-11,**
with the move from a borrowed API key to the project's own OpenAI account.
**Why, measured:** as the answer judge on `evaluation/judge_cases.yaml`,
gpt-4o-mini caught **1 of 5** real overstatements across three runs and gpt-4o
caught 2 of 5, while Luna caught **5 of 5 in both of its runs (19/19)** - as did
6.1 Sol and Astra - without flagging any faithful answer. gpt-4o-mini could not
see "entitled" where the law says "the labour officer may recommend" even with
the section in front of it, and it was the model writing those claims. Luna is
also the cheapest of the three; Sol was ruled out on cost for now.
**What was NOT measured before switching:** Luna *writing* answers - whether
its quotes survive verification, and whether it refuses the right questions.
The first product tier-2 run on Luna is that measurement, and it becomes the
product baseline.
**Two consequences to keep in mind:**
- **GPT-6 refuses `temperature`.** `llm.py` learns that from the API and
  retries once without it (`rejects_temperature`, matched on the error's
  `param` field, never on a list of model names). Temperature 0 was what
  narrowed misquoting between calls, so Luna's answers vary more, and tier-2
  results on it need repeating before they are trusted.
- **Luna currently judges its own answers**, because the judge follows
  `OPENAI_MODEL`. A judge rejection still counts; an approval counts for less.
  The intended fix is a separate `JUDGE_MODEL` set to 6.1 Sol once the budget
  allows - the judge runs only in evaluation, so its cost stays small.

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
  **Adding a document costs exactly ONE expensive run if sequenced right:**

      python -m evaluation.run --tier 0      # seconds, catches retrieval loss
      python -m evaluation.run --no-cache    # THE expensive one, promotable
      python -m evaluation.run --accept-last # instant

  The plain invocation is for iterating, **not** for adding a document: its
  cache hits make it unpromotable, and since a new document invalidates the
  rerank cache wholesale it saves nothing anyway. Running plain and then
  `--no-cache` pays the cross-encoder twice - which has happened.
  **Baselines are per suite AND per tier** - `evaluation/baselines/`
  holds `statutory-tier1.json`, `product-tier2.json` and so on, and a run
  is compared with, and promotable over, only its own suite and tier. This
  closed a real gap: with one `baseline.json`, a cold-cache tier-0 or
  `--document` run could be promoted over the tier-1 record and silently
  become a partial baseline. Every run now records its scope, and
  `promotion_refusal()` - shared by `--save-baseline` and `--accept-last` -
  refuses `--id`, `--document`, `--no-rerank`, cache hits, and any run
  that predates scope tracking. A `--suite product` run is complete for its
  suite and may be promoted; it writes only `product-tier<N>.json`.
- **Tier 0 is NOT a sufficient regression canary when a document is added.**
  This project asserted the opposite, acted on it, and was wrong within a day.
  `county-government-role` **passed tier 0** — the Fourth Schedule sat at #6
  of 30, comfortably retrieved — and **failed tier 1**, because the Children
  Act's Part VI, *Role of County Governments*, now reranks above it.
  **Cross-document rerank displacement is invisible to tier 0 by
  construction:** tier 0 asks whether the authority is in the candidate pool,
  and displacement happens after that. Tier 0 keeps its value for breadth
  failures — it caught both of those at eight documents, in seconds — but a
  plan of "tier 0 per document, tier 1 only at the end" accepts displacement
  risk, and that should be named rather than assumed away.
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
- **Do not dilute the reranker's authority over ordering — and at eight
  documents it earns MORE than it did at four.** Re-measured with
  `--no-rerank` over the complete corpus: **121/133 with it, 110/133 without**
  — `rerank_hit` 106 → 95, `rerank_top1` 74 → 62. **Thirteen rescues against
  two breaks, net +11 questions**, up from +8 at four documents. Twelve
  failures made it reasonable to suspect the cross-encoder had turned
  net-harmful; measuring showed the opposite, because a bigger corpus buries
  the right passage deeper and digging it out is exactly its job. Without it,
  *"Can police arrest me without telling me why?"* returns five Criminal
  Procedure Code sections and **no constitutional right at all**.
  The original four-document measurement, which still holds: `rerank_hit`
  65 → 57 and `rerank_top1` 44 → 37. All eight rescues are the same pattern — retrieval
  fills the top with Criminal Procedure Code sections while the
  **constitutional right sits buried**, Article 49 at rank 26 for
  `arrest-bail`, rank 18 for `arrest-court-time` — and the reranker plus
  per-document diversity digs it out. A planned change to fuse rerank order
  with retrieval order by RRF was **abandoned** because of this: it would
  weaken exactly that override, trading eight rescues for two. If the two
  remaining failures are ever worth fixing, it needs a different
  cross-encoder, not a blend.
- **The cross-encoder ranks by resemblance, not by answerhood — and no
  tuning fixes that. This is now the single largest source of failure in the
  system.** Five evaluation failures across four documents are this one
  defect: `employment-notice`, `penal-murder-sentence`,
  `succession-left-out-of-will`, `succession-grant-application` and
  `divorce-process` — the last confirmed to fail at retrieval breadths of 30,
  50 and 75, so it is not a breadth problem.
  *"How much notice must my employer give before
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
- **A wrapped line can manufacture a false structural marker, and the
  damage scales with what the marker does.** Three documents in a row hit
  this. PDF extraction breaks sentences at column width, so a fragment can
  land at the start of a line looking exactly like a heading.
  **Worst case — the Children Act.** Section 218 ends *"...carried out in
  accordance with the **Fifth Schedule.**"*, the line wrapped, and
  `SCHEDULE_PATTERN` was a case-insensitive prefix match. A schedule heading
  *ends the parse*, so this did not mislabel a unit - it **truncated the Act
  at section 218 and silently discarded 219 to 250**, the whole of Part XV,
  Children in Conflict with the Law. Now fixed by requiring what real
  headings look like: UPPERCASE, whole-line, optionally followed by a
  bracketed reference (`FIFTH SCHEDULE [s. 218(5)]`).
  **Benign case — the Marriage Act.** Section 18 ends *"...appointed by the
  registrar under section **50.**"*, wrapping "50." onto its own line, which
  `SECTION_NUMBER_PATTERN` reads as a heading. That spurious section has no
  body, so the empty-content drop discards it and no chunk id collides; the
  only cost is section 18 losing its last token. Recorded in
  `validators/marriage_act.py` as `KNOWN_TEXT_DUPLICATES` rather than
  papered over, with an assertion that the surviving section 50 is the real
  one in Part VIII.
  **The rule:** when a pattern's match *terminates* parsing or *creates* a
  unit, make it as strict as the real thing allows - case, anchoring,
  full-line. And always check the parsed section count against a measured
  expectation, because every one of these failed silently.
- **`sparse.py` exists instead of a dependency.** `sentence_transformers`
  will not give you BGE-M3's lexical weights — asking its `SparseEncoder`
  for the model silently converts it to a generic 4096-dim projection that
  is not term-based. FlagEmbedding exposes the real head but loads a second
  2.2 GB copy of the model. The head is `Linear(1024, 1)`, so it runs on the
  token embeddings of the model already in memory.
- **Retrieval breadth scales with the corpus — but 30 was MEASURED and kept
  at 1,564 chunks. Do not raise it without re-running the sweep.**
  `RETRIEVAL_TOP_K` was 15 for 279 chunks; at 591 that put Article 49 out of
  range entirely for bail questions, so it became 30. Completing the corpus
  produced two `NEVER RETRIEVED` failures — `state-funded-lawyer` (Article 50
  at rank 42) and `divorce-process` (s.66 at rank 74) — and raising k was
  measured, not assumed.
  **Sweep:** k=40 and k=60 buy nothing. k=50 gives 117/118, k=75 gives
  118/118. **But recall is not the measurement that matters.** Reranked end
  to end: `state-funded-lawyer` PASSES at k=50 (the cross-encoder promotes
  Article 50 from retrieval rank 42 to **rank 1** — the reranker does rescue
  from deep), while **`divorce-process` fails at 30, 50 AND 75** — the
  resemblance-over-answerhood defect, not a breadth problem.
  **k=75 is strictly dominated:** same single fix as k=50 at 2.5x instead of
  1.7x, and it makes the fix *worse* — Article 50 falls from rank 1 to rank 3,
  displaced by two generic *Interpretation* sections. Deeper retrieval is not
  monotonically better.
  **Decision: stay at 30.** One question in 118 is not worth **+67% on
  reranking, the dominant request latency**, on every query forever, when the
  failure being bought off is a refusal rather than a wrong answer.
  Also measured and **wrong**: the hypothesis that Constitution schedules were
  eating the budget. One query showed 7 of 30 slots going to schedules; across
  all 118 questions it is **0.5 slots, 1.8%**. Do not generalise retrieval
  behaviour from a single query.

## Gotchas

- The Hugging Face "unauthenticated requests" rate-limit warning on model load is
  benign and does not block anything.
- BGE-M3 + reranker occupy ~2.3–2.5 GB RAM and dominate startup time.
- CPU cross-encoder reranking is the dominant request latency — not the OpenAI
  call.
- Priority when trading off: **accuracy > grounding > usefulness > speed > cost**.
