"""
The request the retriever sends to Qdrant.

No Qdrant and no model: the client and encoders are fakes that record what
they were asked. What matters is the request itself - above all that a
retrieval WITHOUT rewrites is exactly the request it was before rewriting
existed, because every baseline was recorded with that request.
"""

from types import SimpleNamespace

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from backend.app.ai.retriever import LegalRetriever
from backend.app.core.config import (
    DENSE_VECTOR_NAME,
    REWRITE_TOP_K,
    SPARSE_TOP_K,
    SPARSE_VECTOR_NAME,
)


IN_FORCE = Filter(
    must=[FieldCondition(key="in_force", match=MatchValue(value=True))]
)


def vector(text: str) -> list[float]:
    """A distinct, recognisable 'embedding' per text."""

    return [float(len(text)), 1.0]


def fake_retriever(hybrid: bool = True, sparse_empty: bool = False):
    retriever = object.__new__(LegalRetriever)
    retriever.collection_name = "pocket_lawyer_legal"
    retriever.calls = []

    retriever.embedding_model = SimpleNamespace(
        encode=lambda text, normalize_embeddings: SimpleNamespace(
            tolist=lambda: vector(text)
        )
    )

    retriever.lexical = (
        SimpleNamespace(
            encode_one=lambda text: SimpleNamespace(
                is_empty=sparse_empty, indices=[3, 7], values=[0.5, 0.25]
            )
        )
        if hybrid
        else None
    )

    def query_points(**request):
        retriever.calls.append(request)
        return SimpleNamespace(points=[])

    retriever.client = SimpleNamespace(query_points=query_points)

    return retriever


QUESTION = "Can the police search my house?"


# ---------------------------------------------------------
# WITHOUT REWRITES: THE REQUEST IT ALWAYS WAS
# ---------------------------------------------------------

def test_hybrid_retrieval_without_rewrites_is_unchanged():
    retriever = fake_retriever()

    retriever.retrieve(QUESTION, top_k=30)

    assert retriever.calls == [
        {
            "collection_name": "pocket_lawyer_legal",
            "prefetch": [
                Prefetch(
                    query=vector(QUESTION),
                    using=DENSE_VECTOR_NAME,
                    filter=IN_FORCE,
                    limit=30,
                ),
                Prefetch(
                    query=SparseVector(indices=[3, 7], values=[0.5, 0.25]),
                    using=SPARSE_VECTOR_NAME,
                    filter=IN_FORCE,
                    limit=SPARSE_TOP_K,
                ),
            ],
            "query": FusionQuery(fusion=Fusion.RRF),
            "limit": 30,
            "with_payload": True,
        }
    ]


@pytest.mark.parametrize(
    "hybrid, sparse_empty", [(False, False), (True, True)]
)
def test_dense_only_retrieval_without_rewrites_is_unchanged(
    hybrid, sparse_empty
):
    retriever = fake_retriever(hybrid=hybrid, sparse_empty=sparse_empty)

    retriever.retrieve(QUESTION, top_k=30)

    assert retriever.calls == [
        {
            "collection_name": "pocket_lawyer_legal",
            "query": vector(QUESTION),
            "using": DENSE_VECTOR_NAME,
            "query_filter": IN_FORCE,
            "limit": 30,
            "with_payload": True,
        }
    ]


@pytest.mark.parametrize("empty", [None, [], ["", "   "]])
def test_no_usable_rewrites_means_no_rewriting(empty):
    with_none = fake_retriever()
    with_none.retrieve(QUESTION, top_k=30)

    with_empty = fake_retriever()
    with_empty.retrieve(QUESTION, top_k=30, extra_queries=empty)

    assert with_empty.calls == with_none.calls


# ---------------------------------------------------------
# WITH REWRITES
# ---------------------------------------------------------

REWRITES = ["entry and search without warrant", "privacy of the home"]


def rewrite_branches(top_k: int = 30) -> list[Prefetch]:
    return [
        Prefetch(
            query=vector(text),
            using=DENSE_VECTOR_NAME,
            filter=IN_FORCE,
            limit=min(REWRITE_TOP_K, top_k),
        )
        for text in REWRITES
    ]


def test_each_rewrite_adds_one_dense_branch_to_the_same_fusion():
    """
    The question's own dense and lexical branches stay first and
    untouched; rewrites are appended, dense only, at their own budget.
    """

    retriever = fake_retriever()

    retriever.retrieve(QUESTION, top_k=30, extra_queries=REWRITES)

    (request,) = retriever.calls

    plain = fake_retriever()
    plain.retrieve(QUESTION, top_k=30)

    assert request["prefetch"][:2] == plain.calls[0]["prefetch"]
    assert request["prefetch"][2:] == rewrite_branches()
    assert request["query"] == FusionQuery(fusion=Fusion.RRF)
    assert request["limit"] == 30


@pytest.mark.parametrize(
    "hybrid, sparse_empty", [(False, False), (True, True)]
)
def test_dense_only_retrieval_fuses_the_question_with_its_rewrites(
    hybrid, sparse_empty
):
    retriever = fake_retriever(hybrid=hybrid, sparse_empty=sparse_empty)

    retriever.retrieve(QUESTION, top_k=30, extra_queries=REWRITES)

    (request,) = retriever.calls

    assert request["prefetch"] == [
        Prefetch(
            query=vector(QUESTION),
            using=DENSE_VECTOR_NAME,
            filter=IN_FORCE,
            limit=30,
        ),
        *rewrite_branches(),
    ]
    assert request["query"] == FusionQuery(fusion=Fusion.RRF)


# ---------------------------------------------------------
# A FIXED ORDER FOR EXACT TIES
# ---------------------------------------------------------

def point(chunk_id: str, score: float):
    return SimpleNamespace(payload={"chunk_id": chunk_id}, score=score)


def retriever_returning(points):
    retriever = fake_retriever()
    retriever.client = SimpleNamespace(
        query_points=lambda **request: SimpleNamespace(points=list(points))
    )
    return retriever


def test_tied_candidates_always_come_back_in_the_same_order():
    """
    Qdrant's fusion produces exact ties - 0.5 for first place in any list -
    and returned them in a different order on two runs of the same day.
    """

    one_way = [point("b", 0.5), point("a", 0.5), point("c", 0.333)]
    other_way = [point("a", 0.5), point("b", 0.5), point("c", 0.333)]

    first = retriever_returning(one_way).retrieve(QUESTION, top_k=30)
    second = retriever_returning(other_way).retrieve(QUESTION, top_k=30)

    assert [c["chunk_id"] for c in first] == ["a", "b", "c"]
    assert [c["chunk_id"] for c in second] == ["a", "b", "c"]


def test_only_ties_move():
    """A higher score is never placed below a lower one by the tie-break."""

    results = retriever_returning(
        [point("z", 0.9), point("a", 0.5), point("m", 0.2)]
    ).retrieve(QUESTION, top_k=30)

    assert [c["chunk_id"] for c in results] == ["z", "a", "m"]


def test_a_rewrite_budget_never_exceeds_the_question_budget():
    retriever = fake_retriever()

    retriever.retrieve(QUESTION, top_k=2, extra_queries=REWRITES)

    (request,) = retriever.calls

    assert all(branch.limit <= 2 for branch in request["prefetch"][2:])
