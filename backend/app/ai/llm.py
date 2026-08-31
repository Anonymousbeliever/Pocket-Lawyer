from openai import OpenAI

from backend.app.core.config import OPENAI_API_KEY, OPENAI_MODEL


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

Prefer citations such as:

"Article 49 of the Constitution of Kenya, 2010 provides..."

or:

"Under Article 49(1)(b), an arrested person has the right
to remain silent."

Do not cite sources that were not provided.

============================================================
ANSWER STRUCTURE
============================================================

When the sources sufficiently answer the question, prefer:

Direct answer

Relevant legal rules

Important qualifications or exceptions

Sources

When the sources are insufficient, clearly explain the
limitation instead of filling the gap with outside knowledge.

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
    ) -> str:
        """
        Generate a grounded legal answer using only the
        retrieved legal sources.
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

        response = self.client.responses.create(
            model=self.model_name,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )

        answer = response.output_text.strip()

        if not answer:
            raise RuntimeError(
                "The LLM returned an empty response."
            )

        return answer


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

    print(answer)

    print()
    print("=" * 60)
    print("LLM generation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()