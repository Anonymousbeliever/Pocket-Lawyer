"""
Request routing.

Two layers are under test here, and only the first one belongs in this file.

The ARITHMETIC - exact-match precedence, nearest centroid, the margin, and the
fallbacks - is tested with a stub embedder, so this file runs in seconds with
no models, no Qdrant and no API key, like the rest of the suite.

Whether BGE-M3 genuinely places "morning" near "hello" and "who can arrest me"
near the legal centroid is a question about the real model. That is measured by
the CLI harness, `python -m backend.app.ai.intent`, which is also where
CONVERSATIONAL_MARGIN comes from.

The rule both layers protect is **fail toward LEGAL**. A greeting sent to the
pipeline costs seconds and cents. A legal question sent to a canned reply is
the product failing at its job, so the asymmetric cases matter most.
"""

import numpy as np
import pytest

from backend.app.ai.intent import (
    CONVERSATIONAL,
    EXACT,
    Intent,
    IntentClassifier,
    exact_match,
    normalise,
    reply_for,
)


# ---------------------------------------------------------
# A STUB EMBEDDER
#
# Four orthogonal axes, one per intent. Every exemplar of an intent maps to
# that intent's axis, so its centroid is the axis itself and similarity is
# exactly controllable. This tests the arithmetic, not the language model.
# ---------------------------------------------------------

AXES = {
    Intent.GREETING: 0,
    Intent.IDENTITY: 1,
    Intent.GRATITUDE: 2,
    Intent.LEGAL: 3,
}

EXEMPLARS = {
    intent: [f"{intent}-example-{n}" for n in range(3)]
    for intent in AXES
}


def vector_for(intent: Intent, legal_share: float = 0.0) -> list[float]:
    """
    A vector on `intent`'s axis, optionally pulled toward the legal axis.

    `legal_share` is what makes the margin testable: 0.0 is unambiguous, and
    raising it walks the input toward the boundary.
    """

    vector = np.zeros(4, dtype=np.float32)
    vector[AXES[intent]] = 1.0 - legal_share
    vector[AXES[Intent.LEGAL]] += legal_share

    return (vector / np.linalg.norm(vector)).tolist()


def stub_embed(text: str) -> list[float]:
    for intent, phrases in EXEMPLARS.items():
        if text in phrases:
            return vector_for(intent)

    return vector_for(Intent.LEGAL)


@pytest.fixture
def classifier() -> IntentClassifier:
    return IntentClassifier(
        embed=stub_embed,
        exemplars=EXEMPLARS,
        margin=0.10,
    )


# ---------------------------------------------------------
# THE DEMO FAILURES
# ---------------------------------------------------------

def test_who_are_you_is_not_a_legal_question(classifier):
    """
    Asked live in front of a partner, this was embedded, searched against
    Kenyan law, and answered "I don't have enough reliable information in my
    current legal sources."
    """

    assert classifier.classify("Who are you") is Intent.IDENTITY


def test_what_is_your_name(classifier):
    """
    Found live. The regex covered "what's your name" and missed this, costing
    a retrieval, a 30-passage rerank and a paid LLM call to answer "I don't
    know my name" - having surfaced CPC s.32, "Refusal to give name and
    residence".
    """

    assert classifier.classify("What is your name") is Intent.IDENTITY


def test_morning(classifier):
    """Found live. The regex required "good morning"."""

    assert classifier.classify("morning") is Intent.GREETING


# ---------------------------------------------------------
# EXACT-MATCH FAST PATH
# ---------------------------------------------------------

def test_exact_match_needs_no_vector(classifier):
    """The point of the fast path: common inputs skip the embedding."""

    assert classifier.classify("hi", vector=None) is Intent.GREETING
    assert classifier.classify("thanks", vector=None) is Intent.GRATITUDE


def test_the_fast_path_does_not_pay_for_an_embedding():
    """
    "hi" must not cost a forward pass. The classifier embeds lazily, only
    for what the exact path does not recognise.
    """

    calls: list[str] = []

    def counting_embed(text: str) -> list[float]:
        calls.append(text)
        return stub_embed(text)

    classifier = IntentClassifier(counting_embed, EXEMPLARS, margin=0.10)

    built = len(calls)

    classifier.classify("hi")
    assert len(calls) == built

    classifier.classify("some phrase nobody enumerated")
    assert len(calls) == built + 1


def test_exact_match_wins_over_the_centroids(classifier):
    """Cheaper and certain, so it is consulted first."""

    legal_looking = vector_for(Intent.LEGAL)

    assert classifier.classify("hello", legal_looking) is Intent.GREETING


def test_exact_match_returns_none_rather_than_legal():
    """
    "Not recognised" must mean "ask the centroids", not "it is legal" - the
    fallback moved one level down when semantic routing arrived.
    """

    assert exact_match("morning") is Intent.GREETING
    assert exact_match("what is bail") is None
    assert exact_match("") is None


def test_the_fast_path_stays_small():
    """
    It is an optimisation, not the classifier. Growing it is how the previous
    version failed - guard the intent, not the exact number.
    """

    assert len(EXACT) <= 20


def test_case_and_punctuation_do_not_matter(classifier):
    assert classifier.classify("  HELLO!!  ") is Intent.GREETING
    assert classifier.classify("Who are you???") is Intent.IDENTITY


def test_normalise_strips_case_whitespace_and_trailing_punctuation():
    assert normalise("  Who   are  YOU?? ") == "who are you"


# ---------------------------------------------------------
# NEAREST CENTROID
# ---------------------------------------------------------

def test_unrecognised_conversational_input_reaches_its_centroid(classifier):
    """What the regexes could never do: route a phrase nobody enumerated."""

    for intent in CONVERSATIONAL:
        result = classifier.classify(
            "a phrase in no pattern and no exemplar",
            vector_for(intent),
        )

        assert result is intent


def test_legal_vectors_stay_legal(classifier):
    assert classifier.classify("anything at all", vector_for(Intent.LEGAL)) is (
        Intent.LEGAL
    )


def test_similarities_cover_every_intent(classifier):
    scores = classifier.similarities(vector_for(Intent.GREETING))

    assert set(scores) == set(AXES)
    assert scores[Intent.GREETING] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------
# THE MARGIN — THE SAFETY PROPERTY
# ---------------------------------------------------------

def test_a_near_tie_resolves_to_legal(classifier):
    """
    The asymmetry made concrete. An input pulled most of the way toward the
    legal axis is not confidently conversational, so it goes to the law.
    """

    borderline = vector_for(Intent.GREETING, legal_share=0.48)

    assert classifier.classify("ambiguous", borderline) is Intent.LEGAL


def test_a_clear_win_is_honoured(classifier):
    clear = vector_for(Intent.GREETING, legal_share=0.05)

    assert classifier.classify("clearly chat", clear) is Intent.GREETING


def test_a_wider_margin_routes_more_to_legal():
    """Raising the margin can only move inputs toward LEGAL, never away."""

    vector = vector_for(Intent.GREETING, legal_share=0.35)

    lenient = IntentClassifier(stub_embed, EXEMPLARS, margin=0.05)
    strict = IntentClassifier(stub_embed, EXEMPLARS, margin=0.95)

    assert lenient.classify("borderline", vector) is Intent.GREETING
    assert strict.classify("borderline", vector) is Intent.LEGAL


# ---------------------------------------------------------
# DEGRADING SAFELY
# ---------------------------------------------------------

def test_an_embedding_failure_resolves_to_legal():
    """
    An embedding failure must cost money, not correctness: unrecognised input
    goes to the pipeline rather than being guessed at.
    """

    def broken_embed(text: str) -> list[float]:
        if text in {p for phrases in EXEMPLARS.values() for p in phrases}:
            return stub_embed(text)
        raise RuntimeError("model unavailable")

    classifier = IntentClassifier(broken_embed, EXEMPLARS, margin=0.10)

    assert classifier.classify("habari yako") is Intent.LEGAL
    assert classifier.classify("what is bail") is Intent.LEGAL

    # The fast path still works - it never needed the model.
    assert classifier.classify("hi") is Intent.GREETING


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_empty_input_is_not_small_talk(classifier, question):
    """
    An empty question is the API's to reject with a 422, not the router's to
    chat at - so the fast path declines it and it falls through to the law.
    """

    assert exact_match(question) is None
    assert classifier.classify(question, vector_for(Intent.LEGAL)) is Intent.LEGAL


# ---------------------------------------------------------
# REPLIES
# ---------------------------------------------------------

def test_every_conversational_intent_has_a_reply():
    for intent in CONVERSATIONAL:
        assert reply_for(intent)


def test_legal_intent_has_no_canned_reply():
    """The pipeline answers legal questions, not a lookup table."""

    assert reply_for(Intent.LEGAL) is None


def test_identity_reply_sets_scope_without_lecturing():
    """
    One light clause, not a disclaimer paragraph. The heavy version belongs in
    INSUFFICIENT_ANSWER, where the system is genuinely unable to help.
    """

    reply = reply_for(Intent.IDENTITY)

    assert "advocate" in reply.lower()
    assert reply.lower().count("advocate") == 1


def test_greeting_and_gratitude_carry_no_disclaimer():
    """Repeating it on "hello" only makes the assistant feel defensive."""

    for intent in (Intent.GREETING, Intent.GRATITUDE):
        assert "advocate" not in reply_for(intent).lower()
