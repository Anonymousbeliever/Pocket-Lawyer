import re

from openai import BadRequestError, OpenAI

from backend.app.ai.answer import (
    ANSWER_SCHEMA,
    LegalAnswer,
    render,
    verify_citations,
)
from backend.app.core.config import OPENAI_API_KEY, OPENAI_MODEL


# ---------------------------------------------------------
# MODELS THAT REFUSE `temperature`
# ---------------------------------------------------------

# Some models reject `temperature` outright - the request fails rather than
# ignoring it - and run at their own fixed default instead.
#
# The "o" series is known in advance and is never sent it. Everything else
# is LEARNED from the API: the first call sends `temperature=0`, and if the
# API refuses that parameter the call is retried once without it. A list of
# names was the first design, and GPT-6 broke it within weeks - every call
# failed with "Unsupported parameter: 'temperature'", which on this path
# means every question a user asks. The next model family would have broken
# a list the same way.
#
# Shared with the evaluation judge, which hit the same refusal first.
_REASONING_MODEL = re.compile(r"^o\d")


def is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_MODEL.match(model_name.strip().lower()))


def rejects_temperature(error: Exception) -> bool:
    """
    True only for the API refusing the `temperature` parameter itself.

    Matched on the structured `param` field of a 400, not on the wording of
    the message, which can change. Every other error is a real failure and
    must surface, never be retried.
    """

    return isinstance(error, BadRequestError) and error.param == "temperature"


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are Pocket Lawyer Kenya, an AI legal information
assistant focused on Kenyan law.

Your purpose is to explain authoritative Kenyan legal
information clearly and accurately.

You are NOT a lawyer and must not present yourself as one.

============================================================
GROUNDING RULES
============================================================

1. Use ONLY the legal sources provided in the context.

2. Treat the provided legal sources as the only authoritative
   basis for your answer.

3. Do NOT use your general knowledge or memory of Kenyan law
   to fill gaps in the retrieved sources.

4. Do NOT invent or assume:
   - legal rules
   - statutes
   - constitutional articles
   - sections
   - regulations
   - court decisions
   - procedures
   - deadlines
   - penalties
   - legal exceptions
   - citations

5. Every important legal claim must be supported by the
   retrieved sources.

6. If the retrieved sources only partially answer the
   question, answer ONLY the part supported by those sources
   and clearly state what information is missing.

7. If the retrieved sources do not contain enough reliable
   information to answer the question, DO NOT provide an
   answer based on general legal knowledge.

   Instead say:

   "I don't have enough reliable information in my current
   legal sources to answer this confidently."

8. Do not supplement missing information with phrases such as:
   - "generally"
   - "typically"
   - "usually"
   - "in most cases"
   - "common legal practice"

   unless the retrieved sources themselves support that claim.

9. Never create a citation that does not appear in the
   retrieved sources.

10. Never claim that a source says something when it does not.

============================================================
LEGAL EXPLANATION
============================================================

- Explain the retrieved law in plain language.
- Preserve the meaning of the legal text.
- Do not unnecessarily simplify away important qualifications.
- Clearly distinguish the legal rule from your explanation.
- If multiple sources apply, explain how they relate to each
  other.
- If sources conflict, identify the conflict instead of
  choosing one silently.

============================================================
SAFETY AND LIMITATIONS
============================================================

Pocket Lawyer provides legal information, not legal
representation.

Do not:
- act as the user's advocate;
- make decisions for the user;
- claim to establish an advocate-client relationship;
- guarantee a legal outcome;
- tell the user that a particular legal strategy will succeed.

When the available sources are insufficient, or when the
matter requires case-specific legal advice, recommend
consulting a qualified advocate.

============================================================
CITATIONS
============================================================

When making a legal claim, cite the relevant source using
the information provided in the context.

Cite each source the way that source is structured. The citation
line in the context tells you which form to use.

For the Constitution, cite by Article:

"Article 49 of the Constitution of Kenya, 2010 provides..."

or:

"Under Article 49(1)(b), an arrested person has the right
to remain silent."

For an Act, cite by section and name the Act:

"Section 36A of the Criminal Procedure Code (Cap. 75)
provides..."

Do not cite sources that were not provided.

============================================================
ANSWER STRUCTURE
============================================================

You return a structured object, not prose. Fill each field:

sufficient
  true only if the retrieved sources genuinely contain enough
  information to answer. If they do not, set it to false and use
  `explanation` to say what is missing. Do not fill the gap with
  outside knowledge.

question_type
  "polar"      a yes/no question — "Can police...?", "Must I...?",
               "Is it legal to...?", "Am I allowed to...?"
  "open"       asks what or which — "What are my rights?"
  "procedural" asks for steps or a process

verdict
  For a polar question, answer the question AS ASKED.

  Read carefully when the question asks whether something is
  permitted and the law prohibits it. The verdict then is "no".

  Worked example:
    Question: "Can police arrest me without telling me why?"
    Law:      an arrested person has the right to be informed
              promptly of the reason for the arrest.
    The question asks whether police MAY do this. The law says
    they may not.
    verdict = "no"

  Use "not_applicable" for open or procedural questions, and
  whenever sufficient is false.

statements
  The legal explanation, broken into one assertion per item, in
  the order they should be read.

  Do NOT begin with "Yes" or "No" — the verdict field already
  carries the direct answer, and it is presented to the user
  before this text. Starting with a polarity word produces a
  contradictory answer.

  State what the law says and how it bears on the question.
  Preserve the source's own wording for anything that carries
  legal weight. If a provision says "promptly", do not restate
  it as "immediately" or "at the exact moment" — those are
  different legal standards.

  Each item has three parts:

    text      the assertion, in plain language
    chunk_id  the exact Chunk ID of the source it rests on,
              copied character for character from the context
    quote     a VERBATIM span copied out of that source's
              Legal Text which supports the assertion

  The quote is checked against the source automatically. Copy
  it exactly — do not paraphrase and do not tidy the
  punctuation. Give at least a full clause; a handful of words
  will be rejected.

  Legal text is enumerated, so the words you need are often a
  stem plus ONE of its paragraphs, with other paragraphs in
  between. Skip the middle with "...":

    "the Court may ... order the attachment of the salary of
     the judgement debtor"

  Each side of the "..." is checked separately and must appear
  in the source, in the order you wrote them. Use it at most
  twice in one quote, and never to join single words — a quote
  assembled that way asserts something the source does not.

  An assertion whose quote cannot be found in its source is
  DISCARDED before the user sees it. If every assertion is
  discarded, the whole answer becomes a refusal. So make a
  claim only where you can point to the words that support it.

qualifications
  Important limits, exceptions or conditions that appear in the
  sources. Same three parts, same checking. Empty list if there
  are none. Do not invent them.

missing
  Only when sufficient is false: what the sources do not contain.
  Empty string otherwise. This is the one field with no quote,
  because absence cannot be quoted.

Remember:

The retrieved legal sources are the authority.
Your job is to explain them, not replace them.
"""


# ---------------------------------------------------------
# LLM CLIENT
# ---------------------------------------------------------


class LegalLLM:
    """
    Generates grounded legal explanations using an OpenAI
    language model.

    The LLM receives the user's question together with
    retrieved legal sources from the RAG pipeline.
    """

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
    ):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Add it to your .env file."
            )

        self.model_name = model_name

        print(
            f"Initializing OpenAI model: "
            f"{self.model_name}"
        )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        # Turned off for good the first time the API refuses it. One
        # LegalLLM serves every request, so the refusal is paid once per
        # process, not once per question.
        self.send_temperature = not is_reasoning_model(model_name)

        print("[PASS] OpenAI client initialized")

    # -----------------------------------------------------
    # CONTEXT BUILDING
    # -----------------------------------------------------

    def build_context(
        self,
        sources: list[dict],
    ) -> str:
        """
        Convert retrieved legal sources into structured
        context for the LLM.
        """

        if not sources:
            return (
                "No reliable legal sources were retrieved."
            )

        context_parts = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            context_parts.append(
                f"""
SOURCE {index}

Document:
{source.get("title")}

Document Type:
{source.get("document_type")}

Citation:
{source.get("citation")} — {source.get("unit_title")}

Published by:
{source.get("source_name")}

Version:
{source.get("version")} (in force: {source.get("in_force")})

Current as at:
{source.get("as_at")}

Chunk ID:
{source.get("chunk_id")}

Legal Text:
{source.get("content")}
"""
            )

        return "\n".join(context_parts)

    # -----------------------------------------------------
    # GENERATION
    # -----------------------------------------------------

    def generate(
        self,
        question: str,
        sources: list[dict],
    ) -> LegalAnswer:
        """
        Generate a grounded legal answer using only the
        retrieved legal sources.

        Returns a structured LegalAnswer, with its citations already
        checked against the sources that were actually provided.
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        context = self.build_context(sources)

        user_prompt = f"""
USER QUESTION
=============
{question}


RETRIEVED LEGAL SOURCES
=======================
{context}


TASK
====
Answer the user's question using ONLY the retrieved legal
sources above.

Before answering, determine whether the retrieved sources
actually contain enough information to answer the question.

If they do:
- Give a clear answer.
- Explain the relevant legal rules.
- Preserve important qualifications.
- Cite the relevant source(s).

If they do NOT:
- Do not use outside knowledge.
- Do not guess.
- Do not provide a "typical" or "general" legal process.
- Clearly state that the available sources are insufficient.

The existence of retrieved text does NOT automatically mean
that the sources are relevant to the question.

Do not introduce legal claims that cannot be supported by the
retrieved sources.
"""

        request = {
            "model": self.model_name,
            "instructions": SYSTEM_PROMPT,
            "input": user_prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "legal_answer",
                    "schema": ANSWER_SCHEMA,
                    "strict": True,
                }
            },
        }

        # Sampling was left at the API default until grounding made the
        # task partly one of VERBATIM REPRODUCTION, which is the worst
        # possible task to sample on. Measured on one question, four runs:
        # the model dropped 1, 1, 3 and 1 of its own claims, flipping the
        # same question between a real answer and a refusal - not because
        # the law changed, but because it reproduced the statute less
        # carefully on some calls.
        #
        # This narrows that spread. It does not eliminate it: the API gives
        # no determinism guarantee even at zero, so tier 2 stays
        # non-deterministic and must never be cached. A model that refuses
        # the parameter - GPT-6 does - loses even this, which is why its
        # tier 2 results need repeating before they are trusted.
        if self.send_temperature:
            request["temperature"] = 0

        response = self._create(request)

        raw = response.output_text.strip()

        if not raw:
            raise RuntimeError(
                "The LLM returned an empty response."
            )

        answer = LegalAnswer.from_json(raw)

        # Deterministic citation check: anything the model cited that
        # was not among the retrieved chunks is removed rather than
        # shown to a user as a real authority.
        return verify_citations(answer, sources)

    def _create(self, request: dict):
        """
        Call the API, dropping `temperature` for good if this model refuses
        it. Only that one refusal is retried; every other error propagates
        exactly as it did before.
        """

        try:
            return self.client.responses.create(**request)

        except Exception as error:                        # noqa: BLE001
            if "temperature" not in request or not rejects_temperature(error):
                raise

            self.send_temperature = False

            print(
                f"[WARN] {self.model_name} does not accept temperature - "
                f"answering at its own default, so the same question can "
                f"be answered differently between calls"
            )

            request = {k: v for k, v in request.items() if k != "temperature"}

            return self.client.responses.create(**request)


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------


def main():
    print("=" * 60)
    print("POCKET LAWYER — LEGAL LLM")
    print("=" * 60)
    print()

    llm = LegalLLM()

    question = input(
        "Enter your legal question: "
    ).strip()

    if not question:
        print("No question provided.")
        return

    # -----------------------------------------------------
    # Temporary test source
    #
    # This is intentionally small.
    # The real RAG pipeline will pass its reranked
    # sources here.
    # -----------------------------------------------------

    sources = [
        {
            "title": "Constitution of Kenya, 2010",
            "document_type": "constitution",
            "citation": "Chapter Four — Article 49",
            "unit_title": "Rights of arrested persons",
            "source_name": "Kenya Law",
            "version": "2010",
            "in_force": True,
            "as_at": "2026-08-31",
            "chunk_id": (
                "constitution-of-kenya-2010-"
                "chapter-four-article-49"
            ),
            "content": """
(1) An arrested person has the right—
(a) to be informed promptly, in a language that the
person understands, of—
(i) the reason for the arrest;
(ii) the right to remain silent; and
(iii) the consequences of not remaining silent;
(b) to remain silent;
(c) to communicate with an advocate, and other persons
whose assistance is necessary;
(d) not to be compelled to make any confession or
admission that could be used in evidence against the
person;
(f) to be brought before a court as soon as reasonably
possible, but not later than twenty-four hours after
being arrested, subject to the constitutional
qualification concerning court hours and court days;
(g) at the first court appearance, to be charged or
informed of the reason for the detention continuing,
or to be released; and
(h) to be released on bond or bail, on reasonable
conditions, pending a charge or trial, unless there are
compelling reasons not to be released.
""",
        }
    ]

    print()
    print("=" * 60)
    print("GENERATING LEGAL ANSWER")
    print("=" * 60)
    print()

    answer = llm.generate(
        question=question,
        sources=sources,
    )

    print(render(answer))

    print()
    print("-" * 60)
    print("STRUCTURED FIELDS")
    print("-" * 60)
    print(f"sufficient      : {answer.sufficient}")
    print(f"question_type   : {answer.question_type}")
    print(f"verdict         : {answer.verdict}")
    print(f"statements      : {len(answer.statements)}")
    print(f"qualifications  : {len(answer.qualifications)}")
    print(f"cited_chunk_ids : {answer.cited_chunk_ids}")

    if answer.ungrounded:
        print(
            f"[WARN] {len(answer.ungrounded)} assertion(s) dropped - the "
            f"quote was not found in the source cited:"
        )

        for item in answer.ungrounded:
            print(f"        {item.chunk_id}: {item.quote[:70]!r}")

    print()
    print("=" * 60)
    print("LLM generation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()