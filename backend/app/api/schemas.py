"""
The HTTP contract.

Citation data comes from the chunk payload, never from the model's prose. The
backend already holds authoritative metadata for every source - document title,
citation, version, the date it is current as at - so the API returns that
directly rather than asking the LLM to format a sources list.
"""

from pydantic import BaseModel, Field

from backend.app.ai.answer import LegalAnswer
from backend.app.ai.intent import Intent


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="A question about Kenyan law.",
    )


class Source(BaseModel):
    """One legal passage, described from its stored payload."""

    chunk_id: str
    document: str | None = None
    document_type: str | None = None
    citation: str | None = None
    unit_title: str | None = None
    content: str | None = None

    # Temporal honesty: which edition this is, and the date the answer is
    # current as at. A legal answer without these is not checkable.
    version: str | None = None
    in_force: bool | None = None
    as_at: str | None = None

    source_name: str | None = None
    source_url: str | None = None

    rerank_score: float | None = None

    @classmethod
    def from_chunk(cls, chunk: dict) -> "Source":
        return cls(
            chunk_id=chunk.get("chunk_id", ""),
            document=chunk.get("title"),
            document_type=chunk.get("document_type"),
            citation=chunk.get("citation"),
            unit_title=chunk.get("unit_title"),
            content=chunk.get("content"),
            version=chunk.get("version"),
            in_force=chunk.get("in_force"),
            as_at=chunk.get("as_at"),
            source_name=chunk.get("source_name"),
            source_url=chunk.get("source_url"),
            rerank_score=chunk.get("rerank_score"),
        )


class AskResponse(BaseModel):
    """
    One reply, whichever branch produced it.

    The legal fields are absent on a conversational reply rather than filled
    with placeholder values - a greeting has no verdict.
    """

    intent: Intent
    answer: str

    sufficient: bool | None = None
    question_type: str | None = None
    verdict: str | None = None
    explanation: str | None = None
    qualifications: list[str] = Field(default_factory=list)

    # `sources` is what the answer actually relied on. `considered` is
    # everything that reached the LLM. Keeping them apart is deliberate: a
    # refusal must not present every reranked passage as though it supported
    # an answer that was never given.
    sources: list[Source] = Field(default_factory=list)
    considered: list[Source] = Field(default_factory=list)

    # Citations the model produced that matched no retrieved chunk. Removed
    # from the answer before it got here; surfaced so the caller knows it
    # happened rather than discovering it in a log.
    unverified_citations: list[str] = Field(default_factory=list)

    @classmethod
    def conversational(cls, intent: Intent, answer: str) -> "AskResponse":
        return cls(intent=intent, answer=answer)

    @classmethod
    def legal(cls, outcome: dict) -> "AskResponse":
        structured: LegalAnswer = outcome["structured"]

        return cls(
            intent=Intent.LEGAL,
            answer=outcome["answer"],
            sufficient=structured.sufficient,
            question_type=structured.question_type,
            verdict=structured.verdict,
            explanation=structured.explanation,
            qualifications=list(structured.qualifications),
            sources=[
                Source.from_chunk(c) for c in outcome.get("sources", [])
            ],
            considered=[
                Source.from_chunk(c) for c in outcome.get("considered", [])
            ],
            unverified_citations=list(structured.unverified_citations),
        )


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    collection: str
    vectors: int | None = None
    detail: str | None = None
