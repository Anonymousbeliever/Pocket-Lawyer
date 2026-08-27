import os

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured. "
        "Add it to your .env file."
    )


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are Pocket Lawyer Kenya, an AI legal information
assistant focused on Kenyan law.

Your purpose is to explain authoritative Kenyan legal
information clearly and accurately.

You are NOT a lawyer and must not present yourself as one.

CRITICAL RULES:

1. Use ONLY the legal sources provided in the context.
2. Do not invent legal rules, cases, statutes, sections,
   articles, or citations.
3. Do not rely on your general memory of Kenyan law when
   the provided sources do not support an answer.
4. Every important legal claim must be supported by one
   or more provided sources.
5. Clearly distinguish between what the source says and
   any explanation you provide.
6. If the provided sources are insufficient to answer the
   question confidently, say so.
7. Do not fabricate missing information.
8. Do not provide unauthorized legal representation.
9. Encourage the user to consult a qualified advocate when
   the matter requires specific legal advice.

When citing a source, identify it using the document title,
chapter/article information, and source information provided
in the context.

The answer should be understandable to an ordinary Kenyan
citizen without unnecessary legal jargon.

Prefer a concise structure:

- Direct answer
- Relevant legal rights/rules
- Important qualifications or exceptions
- Sources

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
            chapter = source.get("chapter") or {}
            article = source.get("article") or {}

            context_parts.append(
                f"""
SOURCE {index}

Document:
{source.get("title")}

Document Type:
{source.get("document_type")}

Chapter:
{chapter.get("number")} — {chapter.get("title")}

Article:
{article.get("number")} — {article.get("title")}

Source:
{source.get("source")}

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
        Generate a grounded legal answer using the
        retrieved legal sources.
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        context = self.build_context(sources)

        user_prompt = f"""
USER QUESTION

{question}

RETRIEVED LEGAL SOURCES

{context}

TASK

Answer the user's question using the retrieved legal
sources above.

Do not introduce legal claims that are not supported
by the provided sources.

Where appropriate, cite the relevant Article or other
legal authority directly in the answer.

If the sources do not provide enough information to answer
the question confidently, clearly state that limitation.
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
            "document_type": "Constitution",
            "chapter": {
                "number": "Four",
                "title": "THE BILL OF RIGHTS",
            },
            "article": {
                "number": 49,
                "title": "Rights of arrested persons",
            },
            "source": "Kenya Law",
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
conditions, pending a charge or trial, unless there
are compelling reasons not to be released.
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