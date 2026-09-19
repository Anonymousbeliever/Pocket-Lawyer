"""
The shapes the evaluation harness works with.

A Question is an expectation about the system. A QuestionResult is what
actually happened. Everything else - scoring, reporting, baseline
comparison - operates on these two.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend.app.core.config import DOCUMENTS_DIR


QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"


@dataclass
class Question:
    """One expectation about the system's behaviour."""

    id: str
    question: str
    answerable: bool
    expect_any_of: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class QuestionResult:
    """What the pipeline actually did with one question."""

    id: str
    answerable: bool

    # None where the metric is not observable:
    #  - for questions the corpus cannot answer, there is no expected
    #    authority to find
    #  - retrieval_hit is None in tier 2, because LegalRAG.answer()
    #    exposes only the reranked sources, not the raw retrieved set.
    #    Tier 1 measures retrieval properly and for free.
    retrieval_hit: bool | None = None
    rerank_hit: bool | None = None
    rerank_top1: bool | None = None

    # What actually came back, so a failure can be diagnosed without
    # re-running anything by hand. Each field holds exactly one stage -
    # conflating them once produced three wrong diagnoses.
    retrieved_top: list[str] = field(default_factory=list)
    reranked_top: list[str] = field(default_factory=list)
    cited_top: list[str] = field(default_factory=list)

    # Tier 2
    sufficient: bool | None = None

    # Whether the answer actually leaned on an expected authority, as
    # opposed to merely having been shown one.
    cited_expected: bool | None = None

    @property
    def refusal_correct(self) -> bool | None:
        if self.sufficient is None:
            return None

        return self.sufficient == self.answerable

    @property
    def false_refusal(self) -> bool:
        """The corpus answers it, but the system declined."""

        return self.answerable and self.sufficient is False

    @property
    def false_answer(self) -> bool:
        """The corpus cannot answer it, but the system answered anyway."""

        return not self.answerable and self.sufficient is True

    @property
    def wrong_authority(self) -> bool:
        """
        Answered confidently while citing none of the expected sources.

        Distinct from `false_answer`, which only sees questions the
        corpus cannot answer at all. This is an in-scope question
        answered from the wrong law — the failure the Criminal Procedure
        Code surfaced, where "Can police arrest me without telling me
        why?" was answered from s.29 (Arrest by police officer without
        warrant) and s.2 (Interpretation) rather than Article 49. Every
        citation was real and passed verification; none of them
        addressed what was asked.

        Counted, not treated as a failure. `expect_any_of` lists the
        authorities we know of, not every authority that could be
        legitimate — the Criminal Procedure Code's own s.123 turned out
        to be a valid answer to the bail question nobody had listed. So
        this flags answers worth reading, rather than asserting they are
        wrong.
        """

        return (
            self.answerable
            and self.sufficient is True
            and self.cited_expected is False
        )

    def passed(self, tier: int) -> bool:
        """
        Whether this question is currently in a good state.

        Tier 0 runs no reranker, so it can only ask whether the expected
        authority came back at all. That is a narrower question than tier
        1's and a genuinely useful one: the Criminal Procedure Code's worst
        regression was `arrest-bail`, where Article 49 fell outside the
        top 30 entirely. No cross-encoder was needed to see it.

        Tier 1 only judges answerable questions: did the expected
        authority survive to the point where the LLM would see it?
        `rerank_top1` is tracked as a quality signal, not a pass
        condition - several articles can be legitimately relevant.

        Tier 2 adds the refusal judgement, which is the only thing
        that can be checked for a question the corpus cannot answer.

        Because the three mean different things, a run must not be compared
        against a baseline from another tier - see `_incomparable_reasons`.
        """

        if tier < 1:
            if not self.answerable:
                return True

            return bool(self.retrieval_hit)

        if tier < 2:
            if not self.answerable:
                return True

            return bool(self.rerank_hit)

        if not self.answerable:
            return self.sufficient is False

        return bool(self.rerank_hit) and self.sufficient is True


# ---------------------------------------------------------
# LOADING AND VALIDATION
# ---------------------------------------------------------

def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    if not path.exists():
        raise FileNotFoundError(f"Question set not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []

    if not isinstance(raw, list):
        raise ValueError("questions.yaml must contain a list.")

    questions = [
        Question(
            id=entry["id"],
            question=entry["question"],
            answerable=bool(entry["answerable"]),
            expect_any_of=list(entry.get("expect_any_of") or []),
            note=entry.get("note", ""),
        )
        for entry in raw
    ]

    errors = validate_questions(questions)

    if errors:
        raise ValueError(
            "Invalid question set:\n  " + "\n  ".join(errors)
        )

    return questions


def validate_questions(questions: list[Question]) -> list[str]:
    errors: list[str] = []

    if not questions:
        return ["Question set is empty."]

    ids = [q.id for q in questions]

    duplicates = sorted({i for i in ids if ids.count(i) > 1})

    if duplicates:
        errors.append("Duplicate question ids: " + ", ".join(duplicates))

    for question in questions:
        if not question.question.strip():
            errors.append(f"{question.id}: question text is empty")

        if question.answerable and not question.expect_any_of:
            errors.append(
                f"{question.id}: answerable questions need at least one "
                "expected chunk id"
            )

        if not question.answerable and question.expect_any_of:
            errors.append(
                f"{question.id}: a question the corpus cannot answer "
                "must not list expected chunks"
            )

    return errors


def corpus_chunk_ids(documents_dir: Path = DOCUMENTS_DIR) -> set[str]:
    """Every chunk id currently produced by the ingestion pipeline."""

    ids: set[str] = set()

    for path in documents_dir.glob("*/chunks.json"):
        chunks = json.loads(path.read_text(encoding="utf-8"))
        ids.update(chunk["chunk_id"] for chunk in chunks)

    return ids


def validate_against_corpus(
    questions: list[Question],
    known_ids: set[str],
) -> list[str]:
    """
    Catch expected chunk ids that do not exist.

    A typo here would otherwise show up as a mysterious retrieval
    failure, sending you looking for a bug in the retriever.
    """

    if not known_ids:
        return []

    errors = []

    for question in questions:
        unknown = [
            chunk_id
            for chunk_id in question.expect_any_of
            if chunk_id not in known_ids
        ]

        if unknown:
            errors.append(
                f"{question.id}: expected chunk id not in the corpus: "
                + ", ".join(unknown)
            )

    return errors
