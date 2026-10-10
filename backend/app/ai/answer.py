"""
What a Pocket Lawyer answer is, how it is checked, and how it is rendered.

The LLM returns a structured object rather than prose. Three things
follow from that, and each fixes a real defect:

1. Question polarity. Asked "Can police arrest me without telling me
   why?", the model previously replied "Yes, the police cannot arrest
   you without telling you why." — self-contradictory, because nothing
   made it commit to a polarity before it started writing. `verdict` is
   now a discrete field the model must fill, and the opening word is
   rendered here by code. The contradiction becomes impossible rather
   than merely discouraged.

2. The refusal stops being a magic string matched in two places.
   `sufficient` is a field to branch on.

3. Fabricated citations become detectable for free. Citations are
   checked against the chunks actually retrieved, deterministically,
   with no second model call.

4. Claims become checkable against their source. Each assertion carries
   a verbatim `quote` from the passage it cites, and code confirms the
   quote really occurs there. This closes the gap this docstring used
   to describe: a free-text explanation could overstate the law, saying
   "at the time of arrest" where Article 49 says "promptly", and
   nothing caught it.

   An assertion whose quote cannot be found is DROPPED rather than
   shown — the same treatment `verify_citations` already gives a
   fabricated id. If nothing survives, the answer becomes the standard
   refusal.

   The honest limit: this verifies *grounding*, not *inference*. A
   model can quote a passage accurately and still draw the wrong
   conclusion from it. Catching that needs a second model pass, which
   is deliberately not done here — see the project plan.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field


INSUFFICIENT_ANSWER = (
    "I don't have enough reliable information in my current legal "
    "sources to answer this confidently. Please consult a qualified "
    "advocate."
)


# A subject-matter gate was tried here and removed. See "Measured: one
# aggregate subject verdict cannot gate sufficiency" in
# docs/project-plan.md before considering it again.


VERDICT_OPENING = {
    "yes": "Yes.",
    "no": "No.",
    "it_depends": "It depends.",
}


# ---------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------

# Every assertion the answer makes carries the passage it came from and
# a verbatim quote out of that passage. `verify_grounding` checks the
# quote is really there, which is what makes the claim checkable rather
# than merely attributed.
#
# `missing` is deliberately NOT grounded: a refusal describes what the
# sources do not contain, and absence cannot be quoted.
GROUNDED_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {
            "type": "string",
            "description": "One assertion, in plain language.",
        },
        "chunk_id": {
            "type": "string",
            "description": (
                "Exact Chunk ID of the source this assertion rests on, "
                "copied from the provided context. Never invent one."
            ),
        },
        "quote": {
            "type": "string",
            "description": (
                "A VERBATIM span copied from that source's Legal Text "
                "which supports the assertion. Copy it exactly, at "
                "least a full clause. An assertion whose quote cannot "
                "be found in the source is discarded."
            ),
        },
    },
    "required": ["text", "chunk_id", "quote"],
}


# OpenAI strict mode requires every property to appear in `required`
# and `additionalProperties` to be false. Fields that do not apply are
# expressed with an explicit value ("not_applicable", []), never by
# omission.
ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": (
                "True only if the retrieved sources actually contain "
                "enough information to answer the question."
            ),
        },
        "question_type": {
            "type": "string",
            "enum": ["polar", "open", "procedural"],
            "description": (
                "polar: a yes/no question. open: asks what/which. "
                "procedural: asks for steps or a process."
            ),
        },
        "verdict": {
            "type": "string",
            "enum": ["yes", "no", "it_depends", "not_applicable"],
            "description": (
                "The direct answer to a polar question, answering it "
                "as asked. 'not_applicable' for non-polar questions "
                "or when sufficient is false."
            ),
        },
        "statements": {
            "type": "array",
            "items": GROUNDED_ITEM,
            "description": (
                "The legal explanation, one assertion per item, in "
                "reading order. Must not open with 'Yes' or 'No' - the "
                "verdict field carries that. Empty when sufficient is "
                "false."
            ),
        },
        "qualifications": {
            "type": "array",
            "items": GROUNDED_ITEM,
            "description": (
                "Important limits, exceptions or conditions stated in "
                "the sources. Empty if none."
            ),
        },
        "missing": {
            "type": "string",
            "description": (
                "Only when sufficient is false: what the sources do "
                "not contain. Empty string otherwise."
            ),
        },
    },
    "required": [
        "sufficient",
        "question_type",
        "verdict",
        "statements",
        "qualifications",
        "missing",
    ],
}


# ---------------------------------------------------------
# ANSWER
# ---------------------------------------------------------

@dataclass
class Grounded:
    """One assertion, the source it rests on, and the quote proving it."""

    text: str
    chunk_id: str
    quote: str

    # Set by verify_citations when the assertion is dropped:
    # "unknown_source" (cited a chunk it was never shown) or
    # "quote_not_found" (misquoted a chunk it was shown).
    failure: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Grounded":
        return cls(
            text=str(data.get("text", "")).strip(),
            chunk_id=str(data.get("chunk_id", "")).strip(),
            quote=str(data.get("quote", "")).strip(),
        )


@dataclass
class LegalAnswer:
    sufficient: bool
    question_type: str
    verdict: str
    statements: list[Grounded] = field(default_factory=list)
    qualifications: list[Grounded] = field(default_factory=list)

    # Only meaningful when sufficient is false: what the sources lack.
    missing: str = ""

    # Populated by verify_grounding: assertions whose quote could not be
    # found in the source they cited. Kept rather than discarded
    # silently, so the evaluation can count them.
    ungrounded: list[Grounded] = field(default_factory=list)

    # WHO refused. Verification can overwrite `sufficient`, and when it did
    # there was no record left of what the model itself decided - so a
    # refusal the model made and a refusal forced by the grounding check
    # looked identical. On the first product-suite run, six questions were
    # refused with the correct law among their sources, and it was not
    # possible to say which mechanism refused them.
    #
    #   model_sufficient      the model's own verdict, before verification
    #   refused_by_grounding  True only when verification turned the
    #                         model's "sufficient" into a refusal
    #   statements_dropped    how many STATEMENTS verification removed -
    #                         `ungrounded` mixes in qualifications, and
    #                         the refusal rule counts statements only
    model_sufficient: bool | None = None
    refused_by_grounding: bool = False
    statements_dropped: int = 0

    @property
    def cited_chunk_ids(self) -> list[str]:
        """
        Derived, never asserted by the model.

        A citation cannot exist without a grounded claim behind it,
        which removes a whole class of mismatch between what an answer
        cites and what it actually relies on.
        """

        seen: list[str] = []

        for item in self.statements + self.qualifications:
            if item.chunk_id and item.chunk_id not in seen:
                seen.append(item.chunk_id)

        return seen

    @classmethod
    def from_json(cls, raw: str) -> "LegalAnswer":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"The LLM returned invalid JSON: {error}"
            ) from error

        absent = [
            key for key in ANSWER_SCHEMA["required"] if key not in data
        ]

        if absent:
            raise RuntimeError(
                "The LLM response is missing required fields: "
                + ", ".join(absent)
            )

        sufficient = bool(data["sufficient"])

        return cls(
            sufficient=sufficient,
            model_sufficient=sufficient,
            question_type=data["question_type"],
            verdict=data["verdict"],
            statements=[
                Grounded.from_dict(item) for item in data["statements"]
            ],
            qualifications=[
                Grounded.from_dict(item) for item in data["qualifications"]
            ],
            missing=str(data["missing"]).strip(),
        )


# ---------------------------------------------------------
# GROUNDING
# ---------------------------------------------------------

# Below this, a "quote" proves nothing - "the" occurs in every passage,
# and a floor is what stops the check being theatre. Roughly a clause.
MIN_QUOTE_CHARS = 40

# Two limits, together, or the rule above is satisfiable by stitching
# single words: "the Court ... may ... not ... order execution" would
# total forty characters while asserting the opposite of the source.
#
# The floor alone does not work. A legitimate elision often has a short
# stem - "the Court may" is thirteen characters - so a floor high enough
# to exclude "not" also excludes the real thing. What separates the two
# is the NUMBER of fragments: quoting a stem and one paragraph takes two,
# assembling a false meaning takes several.
MIN_FRAGMENT_CHARS = 10
MAX_FRAGMENTS = 3

# Legal text is enumerated, and the meaningful unit is usually a stem plus
# ONE of its paragraphs:
#
#     ...the Court
#     may-
#     (a)  ...order execution by attachment and sale...;
#     (b)  order the attachment of the salary of the judgement debtor; or
#
# "the Court may order the attachment of the salary" is how anyone quotes
# that, and it is not a contiguous substring - paragraph (a) sits in the
# middle. Requiring one unbroken span rejected every claim the model made
# about section 39 and collapsed a real answer into a refusal.
#
# So a quote may elide, and each fragment is then checked separately. The
# fragments must appear IN ORDER, which is what stops a model assembling a
# meaning from words scattered through the passage.
ELLIPSIS = re.compile(r"\s*(?:\.\.\.+|…)\s*")

# Characters that differ between the corpus and model output while
# meaning the same thing. The corpus carries em dashes (U+2014) from
# Kenya Law's typesetting, and models freely emit typographic quotes
# where the source has straight ones.
_EQUIVALENT = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    " ": " ",
}

_WHITESPACE = re.compile(r"\s+")

# Paragraph markers - (a), (b), (i), (1) - standing alone between a stem
# and its text.
#
# Quoting enumerated law naturally drops them. Asked how to write a will,
# the model quoted "No written will shall be valid unless-the testator has
# signed or affixed his mark", where the source reads "...unless- (a) the
# testator has signed...". Every word is verbatim and in order; only the
# marker is absent, and the quote was rejected.
#
# Removing them from BOTH sides is safe in a way that general gap-skipping
# is not: a marker carries no meaning, so eliding one cannot change what
# the passage says. Skipping arbitrary text could drop a "not".
_ENUMERATOR = re.compile(r"\((?:[a-z]{1,3}|[ivxl]{1,6}|\d{1,2})\)")

# Punctuation a model puts at the END of a quote it has cut short.
#
# Found in production, and it refused two answerable questions. Asked about
# a factory polluting a river, the model quoted Article 70 as "...the person
# may apply to a court for redress." where the source continues "...for
# redress in addition to any other legal remedies". Asked about being
# refused a job for tribe, it quoted section 5 ending "...or HIV status."
# where the source has "HIV status;". Four quotes, every word verbatim and
# in order, all rejected for their final character - and with them gone the
# refusal rule turned both answers into refusals.
#
# Ignoring it cannot admit anything new. A quote cut short WITHOUT the
# stop already matched, because a fragment only has to occur in the source,
# not end where a sentence ends. The stop added no fidelity; it only turned
# faithful quotes into rejections. Only the end of a fragment is touched,
# so punctuation inside a quote must still match exactly.
_TRAILING_PUNCTUATION = ".;,:"


def normalize_for_match(text: str) -> str:
    """
    Reduce text to what a quote comparison should actually care about.

    Without this the check raises false alarms on correct quotes: a
    model reproduces a passage faithfully but re-wraps the whitespace,
    straightens a dash, or changes case at a sentence boundary. None of
    those mean the quote is invented.

    `cleaners/base.py:normalize_unicode` already removes ligatures from
    the corpus at clean time, so chunk text is clean - it is the model's
    output that needs this.
    """

    text = unicodedata.normalize("NFKC", text)

    for source, replacement in _EQUIVALENT.items():
        text = text.replace(source, replacement)

    text = _ENUMERATOR.sub(" ", text)

    # Kenya Law breaks the line after the dash that introduces a list, so
    # the source reads "unless- (a) the testator" while a quote reads
    # "unless-the testator". Padding the dash on both sides makes the
    # spacing around it irrelevant; applied to both texts it can never
    # create a match between passages that differ in anything but layout.
    text = text.replace("-", " - ")

    return _WHITESPACE.sub(" ", text).strip().casefold()


def quote_supports(quote: str, content: str) -> bool:
    """
    True when every fragment of `quote` occurs in `content`, in order.

    A quote with no ellipsis is the single-fragment case, and must appear
    as one unbroken span.
    """

    fragments = [
        stripped
        for part in ELLIPSIS.split(quote)
        if (stripped := part.strip().rstrip(_TRAILING_PUNCTUATION).rstrip())
    ]

    if not fragments or len(fragments) > MAX_FRAGMENTS:
        return False

    if sum(len(part) for part in fragments) < MIN_QUOTE_CHARS:
        return False

    if any(len(part) < MIN_FRAGMENT_CHARS for part in fragments):
        return False

    haystack = normalize_for_match(content)

    position = 0

    for fragment in fragments:
        needle = normalize_for_match(fragment)

        if not needle:
            return False

        found = haystack.find(needle, position)

        if found < 0:
            return False

        # Search for the next fragment only AFTER this one, so the order
        # in the quote must match the order in the source.
        position = found + len(needle)

    return True


def verify_citations(
    answer: LegalAnswer,
    sources: list[dict],
) -> LegalAnswer:
    """
    Drop assertions that cannot be traced to the source they cite.

    Two failures are treated identically because they are the same
    thing from a user's point of view - the answer claims support it
    does not have:

      * the cited id matches no retrieved chunk (an invented citation)
      * the quote does not occur in that chunk (an invented claim)

    Dropped assertions are kept on `ungrounded` rather than discarded,
    so the evaluation can count them. If nothing survives an answer
    that claimed to be sufficient, the answer is downgraded to the
    standard refusal - it has nothing left to stand on.
    """

    content = {
        source["chunk_id"]: source.get("content") or ""
        for source in sources
        if source.get("chunk_id")
    }

    def grounded(item: Grounded) -> bool:
        """
        Two failures were originally collapsed into one, and that cost a
        debugging session: an unknown chunk_id fell through to matching
        against an empty string, so an INVENTED CITATION was reported as
        though the quote had been fabricated. They need different fixes -
        one is the model naming a source it was not shown, the other is
        the model misquoting a source it was.
        """

        if item.chunk_id not in content:
            item.failure = "unknown_source"
            return False

        if not quote_supports(item.quote, content[item.chunk_id]):
            item.failure = "quote_not_found"
            return False

        return True

    def split(items: list[Grounded]) -> tuple[list[Grounded], list[Grounded]]:
        kept, lost = [], []

        for item in items:
            (kept if grounded(item) else lost).append(item)

        return kept, lost

    kept_statements, lost_statements = split(answer.statements)
    kept_qualifications, lost_qualifications = split(answer.qualifications)

    answer.statements = kept_statements
    answer.qualifications = kept_qualifications
    answer.ungrounded = lost_statements + lost_qualifications
    answer.statements_dropped = len(lost_statements)

    # Surviving ONE claim is not the same as still having an answer.
    #
    # Asked "I want to leave my property to my children after I die",
    # the model made three claims, two were dropped, and the survivor was
    # an incidental point about adopted children's inheritance rights.
    # That was rendered as the answer, with sufficient still true - a
    # confident non-sequitur, which is harder for a reader to detect than
    # the overstatement this check was built to catch.
    #
    # So the verdict and the explanation were composed against the full
    # set of claims; if most of that set is gone, what remains is no
    # longer the answer the model formed. Refuse instead.
    if answer.sufficient and len(kept_statements) <= len(lost_statements):
        answer.sufficient = False
        answer.refused_by_grounding = True
        answer.verdict = "not_applicable"

        if not answer.missing:
            answer.missing = (
                "The retrieved sources do not support the claims needed "
                "to answer this."
            )

    return answer


# ---------------------------------------------------------
# RENDERING
# ---------------------------------------------------------

# The model is told, in the system prompt, what to say when the sources
# are insufficient — so it frequently says it, and `render` then prefixed
# the same sentence again. The user saw the refusal twice, which reads as
# a bug rather than as a considered decline.
REFUSAL_MARKERS = (
    "don't have enough reliable information",
    "do not have enough reliable information",
)


def _without_refusal(explanation: str) -> str:
    """Drop sentences that merely repeat the canned refusal."""

    if not explanation:
        return ""

    kept = [
        sentence
        for sentence in explanation.split(". ")
        if not any(
            marker in sentence.lower() for marker in REFUSAL_MARKERS
        )
    ]

    return ". ".join(part for part in kept if part.strip()).strip()


def render(answer: LegalAnswer) -> str:
    """
    Compose the user-facing text.

    The opening word of a polar answer is written here, not by the
    model, so the stated verdict and the explanation cannot disagree.
    """

    if not answer.sufficient:
        reason = _without_refusal(answer.missing)

        if reason:
            return f"{INSUFFICIENT_ANSWER}\n\n{reason}"

        return INSUFFICIENT_ANSWER

    # Composed from the assertions that survived grounding, so anything
    # the user reads is backed by a quote that was checked.
    explanation = " ".join(
        item.text for item in answer.statements if item.text
    ).strip()

    parts: list[str] = []

    opening = VERDICT_OPENING.get(answer.verdict)

    if answer.question_type == "polar" and opening:
        parts.append(f"{opening} {explanation}".strip())
    else:
        parts.append(explanation)

    if answer.qualifications:
        parts.append("Important qualifications:")
        parts.extend(f"- {item.text}" for item in answer.qualifications)

    return "\n\n".join(part for part in parts if part)


def cited_sources(
    answer: LegalAnswer,
    sources: list[dict],
) -> list[dict]:
    """The retrieved sources the answer actually relied on."""

    if not answer.cited_chunk_ids:
        return []

    order = {cid: index for index, cid in enumerate(answer.cited_chunk_ids)}

    matched = [
        source
        for source in sources
        if source.get("chunk_id") in order
    ]

    matched.sort(key=lambda source: order[source["chunk_id"]])

    return matched
