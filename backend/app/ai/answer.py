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

3. Fabricated citations become detectable for free. `cited_chunk_ids`
   is checked against the chunks actually retrieved, deterministically,
   with no second model call.

What this does NOT do is verify that a claim matches what its source
says. A free-text `explanation` can still overstate the law — saying
"at the time of arrest" where Article 49 says "promptly". That needs
entailment checking against the cited passage, and is tracked as the
Citation verification item in the project plan.
"""

import json
from dataclasses import dataclass, field


INSUFFICIENT_ANSWER = (
    "I don't have enough reliable information in my current legal "
    "sources to answer this confidently. Please consult a qualified "
    "advocate."
)


VERDICT_OPENING = {
    "yes": "Yes.",
    "no": "No.",
    "it_depends": "It depends.",
}


# ---------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------

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
        "explanation": {
            "type": "string",
            "description": (
                "The legal explanation. Must not open with 'Yes' or "
                "'No' - the verdict field carries that. When "
                "sufficient is false, state what is missing."
            ),
        },
        "qualifications": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Important limits, exceptions or conditions in the "
                "sources. Empty if none."
            ),
        },
        "cited_chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Exact Chunk ID values of the sources relied on, "
                "copied from the provided context. Never invent one."
            ),
        },
    },
    "required": [
        "sufficient",
        "question_type",
        "verdict",
        "explanation",
        "qualifications",
        "cited_chunk_ids",
    ],
}


# ---------------------------------------------------------
# ANSWER
# ---------------------------------------------------------

@dataclass
class LegalAnswer:
    sufficient: bool
    question_type: str
    verdict: str
    explanation: str
    qualifications: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)

    # Populated by verify_citations: ids the model produced that do not
    # correspond to any retrieved chunk.
    unverified_citations: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: str) -> "LegalAnswer":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"The LLM returned invalid JSON: {error}"
            ) from error

        missing = [
            key for key in ANSWER_SCHEMA["required"] if key not in data
        ]

        if missing:
            raise RuntimeError(
                "The LLM response is missing required fields: "
                + ", ".join(missing)
            )

        return cls(
            sufficient=bool(data["sufficient"]),
            question_type=data["question_type"],
            verdict=data["verdict"],
            explanation=data["explanation"].strip(),
            qualifications=list(data["qualifications"]),
            cited_chunk_ids=list(data["cited_chunk_ids"]),
        )


def verify_citations(
    answer: LegalAnswer,
    sources: list[dict],
) -> LegalAnswer:
    """
    Split cited ids into those that match a retrieved chunk and those
    that do not.

    A citation the model invented cannot be verified against anything,
    so it is removed from the answer rather than shown to a user as if
    it were a real authority.
    """

    known = {
        source.get("chunk_id")
        for source in sources
        if source.get("chunk_id")
    }

    verified = [cid for cid in answer.cited_chunk_ids if cid in known]
    unverified = [cid for cid in answer.cited_chunk_ids if cid not in known]

    answer.cited_chunk_ids = verified
    answer.unverified_citations = unverified

    return answer


# ---------------------------------------------------------
# RENDERING
# ---------------------------------------------------------

def render(answer: LegalAnswer) -> str:
    """
    Compose the user-facing text.

    The opening word of a polar answer is written here, not by the
    model, so the stated verdict and the explanation cannot disagree.
    """

    if not answer.sufficient:
        if answer.explanation:
            return f"{INSUFFICIENT_ANSWER}\n\n{answer.explanation}"

        return INSUFFICIENT_ANSWER

    parts: list[str] = []

    opening = VERDICT_OPENING.get(answer.verdict)

    if answer.question_type == "polar" and opening:
        parts.append(f"{opening} {answer.explanation}")
    else:
        parts.append(answer.explanation)

    if answer.qualifications:
        parts.append("Important qualifications:")
        parts.extend(f"- {item}" for item in answer.qualifications)

    return "\n\n".join(parts)


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
