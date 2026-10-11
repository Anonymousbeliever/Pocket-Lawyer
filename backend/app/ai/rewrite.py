"""
Query rewriting: a person's question, restated in the vocabulary of the law.

WHY
---

Retrieval matches words and meanings, and people do not use the statute's
words. Measured on 2026-09-24:

    "Is torture allowed in Kenya?"                        Art 29 at rank 1
    "The police beat me while questioning me. Is that
     against the law?"                                    beyond rank 100

Article 29 is written in terms of torture and cruel, inhuman or degrading
treatment. The second question contains none of those words, so the law it
needs never reaches the reranker.

WHAT THE REWRITER MAY DO, AND WHAT IT MAY NOT
---------------------------------------------

It TRANSLATES. It does not answer, decide whether something is legal, or
advise. And it never names a statute, the Constitution, or any article or
section - that rule departs from the external suggestion that first
proposed rewriting, whose example rewrite named "Article 29". The reason is
measured: asked which provision answers a pre-trial bail question, an LLM
returned two sections on bail PENDING APPEAL. A rewriter that names the
wrong provision steers retrieval toward it with full confidence. Legal
concepts carry the bridging vocabulary without that risk, and `clean()`
enforces the rule in code rather than trusting the prompt.

A bad rewrite costs a retrieval miss. It cannot produce a wrong citation:
the answering model still sees only real retrieved sections, and every
claim is still verified against them.

DEGRADE, NEVER FAIL
-------------------

No key, an API error, or nothing usable returned: `rewrite()` gives `[]`,
and retrieval proceeds exactly as it does without rewriting.

THE CACHE IS A RECORD, NOT A CONVENIENCE
----------------------------------------

The rewriter runs on a model that refuses `temperature`, so the same
question can come back as different phrases on different calls. The
evaluation therefore stores every rewrite it uses, and a stored rewrite is
reused for good. That is what keeps tier 0 and tier 1 repeatable - and free
and keyless once the store is warm. Unlike evaluation/.cache, deleting it
CHANGES results, which is why it is committed.

Serving passes no cache directory: writing users' questions to disk is not
something to do by default.
"""

import hashlib
import json
import re
from pathlib import Path

from openai import OpenAI

from backend.app.ai.llm import is_reasoning_model, rejects_temperature
from backend.app.core.config import OPENAI_API_KEY, OPENAI_MODEL, REWRITE_COUNT


REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "phrases": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["phrases"],
}

# Phrases are search text, not prose. Longer than this is the model writing
# an answer rather than a search phrase.
MAX_PHRASE_CHARS = 120

# The example is invented and deliberately unlike every question in both
# evaluation suites - a rewriter shown a test question would be measured on
# its memory, not its translation. tests/test_rewrite.py checks that.
INSTRUCTIONS = """
You help search the text of Kenyan law. A member of the public has asked a
question in their own words. Restate it as short search phrases written the
way statutes and the Constitution express the same ideas, so that a search
over the text of the law finds the provisions that apply.

Rules:
- Translate. Do NOT answer the question, decide what the law is, say
  whether something is legal, or suggest what the person should do.
- Never name a statute, Act, Code or the Constitution, and never give an
  article, section, chapter, part or schedule number. Name legal concepts
  only.
- Use the words a provision would use, not everyday words.
- Name the concepts to search for, never what the law says about them.
  Include no numbers, ages, durations, amounts or rates, and do not list
  the categories, conditions or outcomes a provision sets out - those are
  what the search must find, not what you supply. Use only what the
  question itself raises.
- Cover different angles where they exist: the right or protection
  involved, the duty or prohibition, and the procedure or remedy.
- Each phrase under 12 words. Return at most {count} phrases.
- If the input is not a question about law, return no phrases.

Example - invented, for illustration only:
  Question: "My neighbour's dog bit my son on the way to school."
  Phrases:  "injury caused by an animal",
            "duty of the keeper of an animal",
            "compensation for bodily injury"
  Not:      "keeper liable for double the damage" - that states what a
            law provides, which is for the search to find.

Respond in the required JSON only.
"""

# A provision named by number, or a named instrument. Either would steer
# retrieval toward a specific provision on the model's say-so.
_CITATION = re.compile(
    r"(?i)\b(?:article|art|section|sec|cap|chapter|part|schedule)\.?\s*\(?\d"
    r"|\bs\.\s*\d"
    r"|\bconstitution\b"
)

# Case-sensitive on purpose: "Employment Act" is a name, while "an act or
# omission" is the ordinary legal register the rewriter is asked to use.
_INSTRUMENT = re.compile(r"\b(?:Act|Code)\b")

# A number in a search phrase is almost always the law's content, recalled
# from the model's memory rather than taken from the question. Found in the
# first 165 stored rewrites: "not less than twenty-one working days after
# twelve consecutive months of service" for an annual-leave question is the
# answer itself, and "sick leave with half pay" restates another section.
# Recalled law can be wrong or out of date, and a phrase carrying it steers
# retrieval toward it - so this is enforced in code, not left to the prompt.
# Whole words only: "everyone", "once" and "someone" are untouched.
_NUMBER = re.compile(
    r"(?i)\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven"
    r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen"
    r"|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
    r"|hundred|thousand|million|billion|half|double|treble)\b"
)


def clean(phrases: list, count: int = REWRITE_COUNT) -> list[str]:
    """
    The phrases that may be searched with: non-empty, short, distinct,
    naming no provision or instrument, and stating no number - at most
    `count` of them.

    Dropped rather than repaired. Removing "Article 29" from a phrase leaves
    words chosen to fit Article 29, which is the steering this prevents.
    """

    kept: list[str] = []
    seen: set[str] = set()

    for phrase in phrases or []:
        if not isinstance(phrase, str):
            continue

        text = " ".join(phrase.split()).strip(" \"'.;,")

        if not text or len(text) > MAX_PHRASE_CHARS:
            continue

        if (
            _CITATION.search(text)
            or _INSTRUMENT.search(text)
            or _NUMBER.search(text)
        ):
            continue

        folded = text.casefold()

        if folded in seen:
            continue

        seen.add(folded)
        kept.append(text)

        if len(kept) == count:
            break

    return kept


def rerank_query(question: str, rewrites: list[str] | None) -> str:
    """
    What the reranker reads in Phase C: the person's own words, then the
    rewrites as legal terms.

    Why: rewriting alone fixed retrieval and gained nothing after
    reranking (product tier 1: 15 -> 15/22). The cross-encoder scored the
    passages against the lay question, and discarded the right law even
    where retrieval now ranked it first - CPC s.177 for "The police took
    my phone and laptop".

    The question stays first and whole, deliberately. The reranker's
    measured rescues - Article 49 promoted from retrieval rank 26 for a
    bail question - depend on it reading the real question; replacing the
    question with its rewrites would put exactly those at risk.

    One combined query, not one score per rewrite: the cross-encoder is
    the dominant request latency, and scoring every passage once per
    rewrite would multiply it by four.
    """

    terms = [term.strip() for term in rewrites or [] if term.strip()]

    if not terms:
        return question

    return f"{question}\nRelated legal terms: {'; '.join(terms)}"


class QueryRewriter:
    """
    One question in, at most `count` search phrases out.

    `hits`, `calls` and `failures` are counted so a run can say how many
    rewrites it reused, made fresh, or could not get - a rewrite run where
    nothing was rewritten would otherwise look like a real measurement.
    """

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
        count: int = REWRITE_COUNT,
        cache_dir: Path | None = None,
        client=None,
    ):
        self.model_name = model_name
        self.count = count
        self.cache_dir = cache_dir

        if client is None and OPENAI_API_KEY:
            client = OpenAI(api_key=OPENAI_API_KEY)

        self.client = client

        # Learned from the API, exactly as llm.py does.
        self.send_temperature = not is_reasoning_model(model_name)

        self.hits = 0
        self.calls = 0
        self.failures = 0
        self.last_error = ""

    # -----------------------------------------------------

    def rewrite(self, question: str) -> list[str]:
        question = question.strip()

        if not question:
            return []

        key = self._key(question)

        stored = self._read(key)

        if stored is not None:
            self.hits += 1
            return stored

        if self.client is None:
            return []

        try:
            phrases = self._ask(question)

        except Exception as error:                        # noqa: BLE001
            # Never stored: a failure is not a rewrite, and storing it would
            # make the miss permanent.
            self.failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return []

        self.calls += 1
        self._write(key, question, phrases)

        return phrases

    # -----------------------------------------------------

    def _instructions(self) -> str:
        return INSTRUCTIONS.format(count=self.count)

    @property
    def prompt_id(self) -> str:
        """A short fingerprint of the exact instructions sent."""

        return hashlib.sha256(
            self._instructions().encode("utf-8")
        ).hexdigest()[:12]

    def _ask(self, question: str) -> list[str]:
        request = {
            "model": self.model_name,
            "instructions": self._instructions(),
            "input": question,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "search_phrases",
                    "schema": REWRITE_SCHEMA,
                    "strict": True,
                }
            },
        }

        if self.send_temperature:
            request["temperature"] = 0

        try:
            response = self.client.responses.create(**request)

        except Exception as error:                        # noqa: BLE001
            if "temperature" not in request or not rejects_temperature(error):
                raise

            self.send_temperature = False
            del request["temperature"]

            response = self.client.responses.create(**request)

        data = json.loads(response.output_text.strip())

        return clean(data.get("phrases"), self.count)

    # -----------------------------------------------------
    # THE STORE
    # -----------------------------------------------------

    def _key(self, question: str) -> str:
        """
        A stored rewrite is valid only for the same question, model and
        instructions - so changing the prompt or the model starts afresh
        instead of silently reusing phrases a different prompt produced.
        """

        payload = json.dumps(
            ["rewrite-v1", self.model_name, self._instructions(), question]
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _read(self, key: str) -> list[str] | None:
        if self.cache_dir is None:
            return None

        path = self.cache_dir / f"{key}.json"

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return list(data["phrases"])

        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write(self, key: str, question: str, phrases: list[str]) -> None:
        if self.cache_dir is None:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # The question, model and prompt are stored beside the phrases
            # so the record can be read and reviewed, not just reused - and
            # so rewrites made by different prompts, which sit side by side
            # in one folder, can be told apart. Records without "prompt"
            # came from the first prompt.
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(
                    {
                        "question": question,
                        "model": self.model_name,
                        "prompt": self.prompt_id,
                        "phrases": phrases,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        except OSError:
            # Failing to store must never fail a run.
            pass


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------

def main() -> None:
    rewriter = QueryRewriter()

    if rewriter.client is None:
        print("[FAIL] OPENAI_API_KEY is not configured.")
        return

    print(f"Rewriter model: {rewriter.model_name}")
    print("Blank line or Ctrl-C to quit.")

    while True:
        print()

        try:
            question = input("Question: ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            break

        phrases = rewriter.rewrite(question)

        if rewriter.last_error:
            print(f"[ERROR] {rewriter.last_error}")
            rewriter.last_error = ""

        for phrase in phrases or ["(no phrases)"]:
            print(f"  - {phrase}")


if __name__ == "__main__":
    main()
