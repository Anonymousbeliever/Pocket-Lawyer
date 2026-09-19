"""
The HTTP layer.

`get_rag` is overridden with a stub throughout, so these run with no models, no
Qdrant and no API key - the same constraint the rest of the suite honours.

The assertion that matters most is the negative one: a conversational request
must never touch the pipeline. Counting calls on the stub is what proves the
bypass exists rather than merely appearing to.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.ai.answer import LegalAnswer
from backend.app.ai.intent import Intent, exact_match
from backend.app.api.routes import get_classifier, get_rag
from backend.app.main import app


ARTICLE_49 = "constitution-of-kenya-2010@v2010-chapter-four-article-49"
CPC_29 = "criminal-procedure-code@v2023-12-11-part-iii-section-29"


def chunk(chunk_id: str, **overrides) -> dict:
    base = {
        "chunk_id": chunk_id,
        "title": "Constitution of Kenya, 2010",
        "document_type": "constitution",
        "citation": "Chapter Four — Article 49",
        "unit_title": "Rights of arrested persons",
        "content": "An arrested person has the right to remain silent.",
        "version": "2010",
        "in_force": True,
        "as_at": "2026-09-12",
        "source_name": "Kenya Law",
        "source_url": None,
        "rerank_score": 0.99,
    }
    base.update(overrides)
    return base


class StubCount:
    def __init__(self, count: int):
        self.count = count


class StubQdrant:
    def count(self, collection_name: str) -> StubCount:
        return StubCount(591)


class StubRetriever:
    """Only what /health reaches for."""

    def __init__(self):
        self.client = StubQdrant()


class StubClassifier:
    """
    The real routing arithmetic is covered in test_intent.py with controlled
    vectors. Here the only question is whether the route wires it up - so this
    uses the genuine exact-match path and sends everything else to the law.
    """

    def classify(self, text: str, vector=None) -> Intent:
        return exact_match(text) or Intent.LEGAL


class StubRAG:
    """Records what it was asked, so the bypass can be asserted."""

    def __init__(self, outcome: dict | None = None):
        self.calls: list[str] = []
        self.outcome = outcome or self.grounded()
        self.retriever = StubRetriever()

    def answer(self, question: str) -> dict:
        self.calls.append(question)
        return self.outcome

    @staticmethod
    def grounded() -> dict:
        return {
            "answer": "No. An arrested person has the right to remain silent.",
            "structured": LegalAnswer(
                sufficient=True,
                question_type="polar",
                verdict="no",
                explanation="An arrested person has the right to remain silent.",
                qualifications=["Subject to the twenty-four hour limit."],
                cited_chunk_ids=[ARTICLE_49],
            ),
            "sources": [chunk(ARTICLE_49)],
            "considered": [chunk(ARTICLE_49), chunk(CPC_29)],
        }

    @staticmethod
    def refusal() -> dict:
        return {
            "answer": "I don't have enough reliable information...",
            "structured": LegalAnswer(
                sufficient=False,
                question_type="open",
                verdict="not_applicable",
                explanation="The sources do not cover the Penal Code.",
                qualifications=[],
                cited_chunk_ids=[],
            ),
            "sources": [],
            "considered": [chunk(CPC_29)],
        }


@pytest.fixture
def stub():
    return StubRAG()


@pytest.fixture
def client(stub):
    # Deliberately NOT `with TestClient(app)`. The context-manager form runs
    # the lifespan, which builds a real LegalRAG - models, Qdrant and an API
    # key. These tests must need none of those.
    app.dependency_overrides[get_rag] = lambda: stub
    app.dependency_overrides[get_classifier] = lambda: StubClassifier()

    # /health reads app.state directly rather than through the dependency,
    # because it has to be able to report "not loaded yet".
    app.state.rag = stub

    yield TestClient(app)

    app.dependency_overrides.clear()
    app.state.rag = None


# ---------------------------------------------------------
# THE BYPASS
# ---------------------------------------------------------

def test_identity_request_never_reaches_the_pipeline(client, stub):
    """The whole point of the router."""

    response = client.post("/ask", json={"question": "Who are you"})

    assert response.status_code == 200
    assert response.json()["intent"] == "identity"
    assert stub.calls == []


def test_greeting_never_reaches_the_pipeline(client, stub):
    response = client.post("/ask", json={"question": "hello"})

    assert response.json()["intent"] == "greeting"
    assert stub.calls == []


def test_a_missing_classifier_sends_everything_to_the_law(stub):
    """
    Routing is an optimisation, not a precondition. Losing the classifier
    must cost money, never correctness - so even a greeting goes to the
    pipeline rather than the service refusing to answer.
    """

    app.dependency_overrides[get_rag] = lambda: stub
    app.dependency_overrides[get_classifier] = lambda: None
    app.state.rag = stub

    body = TestClient(app).post("/ask", json={"question": "hello"}).json()

    assert body["intent"] == "legal"
    assert stub.calls == ["hello"]

    app.dependency_overrides.clear()
    app.state.rag = None


def test_conversational_reply_carries_no_legal_fields(client):
    """A greeting has no verdict, so it must not pretend to."""

    body = client.post("/ask", json={"question": "hi"}).json()

    assert body["answer"]
    assert body["sufficient"] is None
    assert body["verdict"] is None
    assert body["sources"] == []


# ---------------------------------------------------------
# THE LEGAL PATH
# ---------------------------------------------------------

def test_legal_question_goes_through_the_pipeline_once(client, stub):
    question = "What are my rights if I am arrested?"

    body = client.post("/ask", json={"question": question}).json()

    assert stub.calls == [question]
    assert body["intent"] == "legal"
    assert body["sufficient"] is True
    assert body["verdict"] == "no"
    assert body["qualifications"] == ["Subject to the twenty-four hour limit."]


def test_sources_are_structured_from_the_payload(client):
    """
    Citation data comes from the chunk, never from the model's prose - the
    backend holds the authoritative metadata.
    """

    body = client.post(
        "/ask", json={"question": "What are my rights if I am arrested?"}
    ).json()

    source = body["sources"][0]

    assert source["chunk_id"] == ARTICLE_49
    assert source["citation"] == "Chapter Four — Article 49"
    assert source["document"] == "Constitution of Kenya, 2010"
    assert source["version"] == "2010"
    assert source["in_force"] is True
    assert source["as_at"] == "2026-09-12"


def test_cited_and_considered_stay_separate(client):
    """
    A refusal must not present everything the reranker passed to the LLM as
    though it supported an answer that was never given.
    """

    app.dependency_overrides[get_rag] = lambda: StubRAG(StubRAG.refusal())

    body = client.post(
        "/ask", json={"question": "What is robbery under Kenyan law?"}
    ).json()

    assert body["sufficient"] is False
    assert body["sources"] == []
    assert len(body["considered"]) == 1


def test_unverified_citations_are_surfaced(client):
    """Stripped before the user sees them, but reported rather than buried."""

    outcome = StubRAG.grounded()
    outcome["structured"].unverified_citations = ["invented-section-99"]

    app.dependency_overrides[get_rag] = lambda: StubRAG(outcome)

    body = client.post("/ask", json={"question": "What are my rights?"}).json()

    assert body["unverified_citations"] == ["invented-section-99"]


# ---------------------------------------------------------
# VALIDATION AND STATUS
# ---------------------------------------------------------

@pytest.mark.parametrize("question", ["", "   "])
def test_empty_questions_are_rejected(client, question, stub):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert stub.calls == []


def test_root_keeps_its_original_shape(client):
    """Pre-existing contract; adding a service must not break it."""

    body = client.get("/").json()

    assert body == {
        "name": "Pocket Lawyer",
        "status": "online",
        "version": "0.1.0",
    }


def test_health_reports_readiness(client):
    body = client.get("/health").json()

    assert body["status"] == "ready"
    assert body["models_loaded"] is True
    assert body["vectors"] == 591


def test_health_reports_when_models_are_not_loaded():
    """Startup takes ~40 seconds; the endpoint must say so rather than 500."""

    app.state.rag = None

    body = TestClient(app).get("/health").json()

    assert body["status"] == "starting"
    assert body["models_loaded"] is False


def test_ask_returns_503_before_models_are_loaded():
    app.dependency_overrides.clear()
    app.state.rag = None

    response = TestClient(app).post("/ask", json={"question": "hello"})

    assert response.status_code == 503
