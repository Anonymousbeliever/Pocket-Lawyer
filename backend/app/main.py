"""
The Pocket Lawyer API.

    python -m uvicorn backend.app.main:app --reload

Models load once here, at startup, and are reused for every request. The CLI
harnesses build them per process, which costs ~40 seconds and ~2.5 GB each time
- fine for a one-off lookup, unusable for a service.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.ai.intent import IntentClassifier
from backend.app.ai.rag import LegalRAG
from backend.app.api.routes import router
from backend.app.core.config import COLLECTION_NAME, QDRANT_URL


STARTUP_HELP = f"""
Pocket Lawyer could not start.

The legal engine needs two things before it can serve requests:

  1. Qdrant, holding the '{COLLECTION_NAME}' collection, at {QDRANT_URL}
     Start it with:  docker compose up -d

  2. OPENAI_API_KEY set in .env

Original error: {{error}}
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once. Every request reuses this instance.
    try:
        app.state.rag = LegalRAG()

    except Exception as error:
        # Without this, a stopped container surfaces as ~100 lines of httpx
        # and qdrant_client traceback that never names the actual problem.
        # Handled here rather than inside the retriever, which stays as it is.
        raise RuntimeError(STARTUP_HELP.format(error=error)) from error

    # Routing centroids, built from the embedding model the retriever just
    # loaded. Roughly 70 short strings, embedded once.
    #
    # A failure here is NOT fatal. Routing is an optimisation over the
    # pipeline, not a precondition for it: without a classifier every request
    # goes to the law, which costs money and looks clumsy on a greeting but
    # answers legal questions correctly. Refusing to start would trade a
    # working service for a tidy one.
    try:
        print()
        print("Building intent centroids...")

        app.state.classifier = IntentClassifier(
            embed=app.state.rag.retriever.embed_query,
        )

        print("[PASS] Intent classifier ready")

    except Exception as error:
        app.state.classifier = None

        print(f"[WARN] Intent routing unavailable: {error}")
        print("       Every request will go through the legal pipeline.")

    yield

    app.state.rag = None
    app.state.classifier = None


app = FastAPI(
    title="Pocket Lawyer API",
    version="0.1.0",
    description=(
        "AI legal information for Kenyan law, answered from the text of "
        "authoritative sources with citations."
    ),
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
def root():
    return{
        "name": "Pocket Lawyer",
        "status": "online",
        "version": "0.1.0"
    }
