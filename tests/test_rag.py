"""
How the pipeline wires in query rewriting.

Every model is a fake that records what it was handed - no Qdrant, no
cross-encoder, no API key. What matters is what flows where: the rewrites
reach retrieval and the reranker, never the answering model, and with no
rewrites every call is exactly what it was before rewriting existed.
"""

from types import SimpleNamespace

from backend.app.ai import rag as rag_module
from backend.app.ai.answer import LegalAnswer
from backend.app.ai.rag import LegalRAG, resolve_rewriter
from backend.app.ai.rewrite import QueryRewriter, rerank_query


QUESTION = "The police took my phone. Will I get it back?"
CHUNK = {"chunk_id": "s177", "document_id": "cpc", "content": "text"}
REWRITES = ["seizure of property by police", "return of seized property"]


def pipeline(rewrites=None, retrieved=True):
    """
    A LegalRAG with fakes for every model. `rewrites=None` means no
    rewriter at all; a list is what the rewriter returns.
    """

    p = object.__new__(LegalRAG)
    p.calls = {}

    def retrieve(**request):
        p.calls["retrieve"] = request
        return [dict(CHUNK)] if retrieved else []

    def rerank(**request):
        p.calls["rerank"] = request
        return [dict(CHUNK)]

    def generate(**request):
        p.calls["generate"] = request
        return LegalAnswer(
            sufficient=False,
            question_type="open",
            verdict="not_applicable",
            missing="The sources are silent.",
        )

    p.retriever = SimpleNamespace(retrieve=retrieve)
    p.reranker = SimpleNamespace(rerank=rerank)
    p.llm = SimpleNamespace(generate=generate)
    p.rewriter = (
        None
        if rewrites is None
        else SimpleNamespace(rewrite=lambda question: list(rewrites))
    )

    return p


# ---------------------------------------------------------
# WITHOUT REWRITING: UNCHANGED
# ---------------------------------------------------------

def test_without_a_rewriter_every_call_is_what_it_always_was():
    p = pipeline()

    outcome = p.answer(QUESTION)

    assert p.calls["retrieve"]["extra_queries"] is None
    assert p.calls["rerank"]["query"] == QUESTION
    assert outcome["rewrites"] == []


def test_a_failed_rewrite_is_the_pipeline_without_rewriting():
    """The rewriter degrades to [] - and [] must change nothing."""

    failed = pipeline(rewrites=[])
    failed.answer(QUESTION)

    off = pipeline()
    off.answer(QUESTION)

    assert failed.calls == off.calls


# ---------------------------------------------------------
# WITH REWRITING
# ---------------------------------------------------------

def test_rewrites_reach_retrieval_and_the_reranker():
    p = pipeline(rewrites=REWRITES)

    outcome = p.answer(QUESTION)

    assert p.calls["retrieve"]["extra_queries"] == REWRITES
    assert p.calls["rerank"]["query"] == rerank_query(QUESTION, REWRITES)
    assert outcome["rewrites"] == REWRITES


def test_the_answering_model_never_sees_the_rewrites():
    """
    Rewrites decide which law is FOUND. The answer is to the person's own
    words, from real retrieved text - so a poor rewrite can cost a
    retrieval, never a citation.
    """

    p = pipeline(rewrites=REWRITES)

    p.answer(QUESTION)

    assert p.calls["generate"]["question"] == QUESTION


def test_the_rewrites_are_reported_even_when_nothing_is_retrieved():
    outcome = pipeline(rewrites=REWRITES, retrieved=False).answer(QUESTION)

    assert outcome["rewrites"] == REWRITES


# ---------------------------------------------------------
# WHICH REWRITER
# ---------------------------------------------------------

def test_a_given_rewriter_is_used_as_given():
    """How the evaluation hands tier 2 the rewrites tier 1 measured."""

    given = object()

    assert resolve_rewriter(given) is given


def test_with_rewriting_off_there_is_no_rewriter(monkeypatch):
    monkeypatch.setattr(rag_module, "QUERY_REWRITE", False)

    assert resolve_rewriter() is None


def test_the_serving_rewriter_never_writes_questions_to_disk(monkeypatch):
    monkeypatch.setattr(rag_module, "QUERY_REWRITE", True)
    # No real client: constructing one must not depend on a key here.
    monkeypatch.setattr("backend.app.ai.rewrite.OPENAI_API_KEY", None)

    rewriter = resolve_rewriter()

    assert isinstance(rewriter, QueryRewriter)
    assert rewriter.cache_dir is None
