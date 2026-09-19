"""
HTTP endpoints.

This is where the branch point lives that the system previously had nowhere:
conversational input is answered directly, and only legal questions reach the
RAG pipeline. `LegalRAG` itself is untouched and unaware of any of this.
"""

import threading

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.app.ai.intent import Intent, IntentClassifier, reply_for
from backend.app.ai.rag import LegalRAG
from backend.app.api.schemas import AskRequest, AskResponse, HealthResponse
from backend.app.core.config import COLLECTION_NAME


router = APIRouter()


# Serialises the heavy path. A sync endpoint is run in Starlette's threadpool,
# which defaults to 40 workers - and 40 concurrent BGE inferences on ~2.2 GB of
# in-process models would thrash the machine.
#
# This reflects TODAY'S inference stack, not a ceiling on Pocket Lawyer. The
# call is blocking because the models run in-process on CPU. Move them to a GPU,
# put them behind an inference service the API talks to over the network, or
# simply run more worker processes, and this goes away: delete the lock and
# change `def ask` to `async def ask`. The pipeline underneath knows nothing
# about it either way.
_inference_lock = threading.Lock()


def get_rag(request: Request) -> LegalRAG:
    """
    The single RAG instance built at startup.

    Also the seam tests override, so the API can be exercised with a stub and
    no models, no Qdrant and no API key.
    """

    rag = getattr(request.app.state, "rag", None)

    if rag is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The legal engine is not loaded. If the service has just "
                "started, wait for model loading to finish."
            ),
        )

    return rag


def get_classifier(request: Request) -> IntentClassifier | None:
    """
    The intent classifier built at startup, or None if it failed to build.

    None is tolerated on purpose: routing is an optimisation over the
    pipeline, not a precondition for it. Losing the classifier should make
    the service spend money it did not need to, never make it stop answering
    legal questions.
    """

    return getattr(request.app.state, "classifier", None)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    rag = getattr(request.app.state, "rag", None)

    if rag is None:
        return HealthResponse(
            status="starting",
            models_loaded=False,
            collection=COLLECTION_NAME,
            detail="Models are not loaded yet.",
        )

    # Cheap round trip to Qdrant. Reporting "ready" without checking would make
    # this endpoint useless for the failure that actually happens.
    try:
        count = rag.retriever.client.count(
            collection_name=COLLECTION_NAME,
        ).count

    except Exception as error:
        return HealthResponse(
            status="degraded",
            models_loaded=True,
            collection=COLLECTION_NAME,
            detail=f"Qdrant is not reachable: {error}",
        )

    return HealthResponse(
        status="ready",
        models_loaded=True,
        collection=COLLECTION_NAME,
        vectors=count,
    )


# Deliberately `def`, not `async def`. The pipeline is CPU-bound plus a
# blocking OpenAI call; an async endpoint would hold the event loop for the
# whole request. Starlette runs a sync endpoint in a threadpool instead.
@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    rag: LegalRAG = Depends(get_rag),
    classifier: IntentClassifier | None = Depends(get_classifier),
) -> AskResponse:
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    intent = _route(question, rag, classifier)

    # Conversational input never reaches retrieval, and never takes the
    # inference lock - so a greeting is answered instantly even while a legal
    # question is being reranked.
    if intent is not Intent.LEGAL:
        return AskResponse.conversational(intent, reply_for(intent))

    with _inference_lock:
        outcome = rag.answer(question)

    return AskResponse.legal(outcome)


def _route(
    question: str,
    rag: LegalRAG,
    classifier: IntentClassifier | None,
) -> Intent:
    """
    Decide where this question goes.

    Semantic routing needs the query vector, and `LegalRetriever.embed_query`
    sits three layers below the branch point inside `rag.answer()`. Rather
    than plumb a precomputed vector down through `rag.py` and `retriever.py`
    - the two files keeping the evaluation harness trustworthy - the
    classifier was given its own handle on the embedder at startup and embeds
    above the branch itself.

    The pipeline then embeds the question a second time. That is deliberate:
    one short query is an order of magnitude cheaper than the thirty-passage
    rerank this exists to avoid, and it costs no edits to the engine. Measure
    before optimising it away.

    Every failure resolves toward LEGAL.
    """

    if classifier is None:
        return Intent.LEGAL

    try:
        return classifier.classify(question)

    except Exception:
        return Intent.LEGAL
