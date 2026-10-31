"""HTTP API and the three pages.

    GET  /                 chat UI                     GET  /browse   read the regulation
    GET  /debug            retrieval inspector
    POST /chat             {"message", "history"} -> Server-Sent Events:
                           query?, retrieved, token*, sources, done
    GET  /livez            process is up (container healthcheck)
    GET  /health           Ollama/OpenAI + models + Qdrant + collection ready? 200 or 503
    POST /ingest           start an ingest in the background
    GET  /ingest           status of the last ingest
    GET  /search?q=...     retrieval results with the score of each branch, for tuning
    GET  /toc              table of contents of every indexed document
    GET  /article?...      full text of one article / appendix
    GET  /source/{doc_id}  the original PDF, so "trang 14" can be opened at page 14
    POST /feedback         thumbs up/down on an answer
    GET  /metrics          answer rate, latency, the questions nothing could answer
    GET  /config           the retrieval settings in force (no secrets)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.library import Library
from app.api.telemetry import Telemetry
from app.config import get_settings
from app.ingest.run import ingest
from app.rag.chat import ChatService
from app.rag.embeddings import OllamaEmbedder
from app.rag.llm import LLMClient
from app.rag.retriever import Retriever
from app.rag.store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("api")

settings = get_settings()
store = QdrantStore(settings.qdrant_url, settings.collection_name)
embedder = OllamaEmbedder(settings)
llm = LLMClient(settings)
retriever = Retriever(settings, embedder, store, llm=llm)
telemetry = Telemetry(Path(settings.log_dir), settings.log_queries)
library = Library(Path(settings.processed_dir), Path(settings.raw_dir))
chat_service = ChatService(settings, retriever, llm, on_turn=telemetry.queries.append)
log.info("chat model: %s (%s), rerank: %s", settings.chat_model,
         "openai" if settings.use_openai else "ollama",
         settings.rerank_mode if settings.rerank_enabled else "off")

STATIC = Path(__file__).resolve().parent.parent / "static"
_ingest_lock = threading.Lock()
_ingest_state: dict = {"status": "idle"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Payload indexes for the exact-article lookup and the neighbour expansion. Harmless if
    # they already exist; skipped quietly when Qdrant is not up yet (ingest creates them too).
    try:
        if store.exists():
            store.ensure_indexes()
    except Exception as e:  # noqa: BLE001
        log.warning("could not ensure payload indexes: %s", e)
    yield


app = FastAPI(title="academic-regs-rag", lifespan=lifespan)
# Stylesheet shared by the three pages. Everything else stays inline so the app keeps working
# with no network at all.
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# --- health ---


@app.get("/livez")
def livez() -> dict:
    return {"ok": True}


def _ollama_models() -> set[str]:
    resp = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5)
    resp.raise_for_status()
    names = {m["name"] for m in resp.json().get("models", [])}
    return names | {n.removesuffix(":latest") for n in names}


@app.get("/health")
def health() -> JSONResponse:
    checks: dict = {"llm_provider": "openai" if settings.use_openai else "ollama"}
    if settings.use_openai:
        # Only the key's presence is checked: /health is polled, and start.sh already verified it.
        checks["llm_model"] = settings.chat_model if settings.openai_api_key else "missing: OPENAI_API_KEY"
    try:
        models = _ollama_models()
        checks["ollama"] = "ok"
        wanted = [("embed_model", settings.embed_model)]
        if not settings.use_openai:
            wanted.insert(0, ("llm_model", settings.llm_model))
        for key, name in wanted:
            checks[key] = name if name in models else f"missing: ollama pull {name}"
    except Exception as e:  # noqa: BLE001 - report any connection problem
        checks["ollama"] = f"unreachable at {settings.ollama_url}: {e}"
    try:
        checks["qdrant"] = "ok"
        checks["points"] = store.count()
    except Exception as e:  # noqa: BLE001
        checks["qdrant"] = f"unreachable at {settings.qdrant_url}: {e}"
    checks["ingest"] = _ingest_state.get("status")
    checks["query_cache"] = {"hits": embedder.cache_hits, "misses": embedder.cache_misses}
    ready = (
        checks["ollama"] == "ok"
        and checks.get("llm_model") == settings.chat_model
        and checks.get("embed_model") == settings.embed_model
        and checks["qdrant"] == "ok"
        and checks.get("points", 0) > 0
    )
    return JSONResponse({"ready": ready, **checks}, status_code=200 if ready else 503)


@app.get("/config")
def config() -> dict:
    """Retrieval settings in force. Handy when tuning; contains no credentials."""
    return {
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "hybrid_search": settings.hybrid_search,
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_final": settings.top_k_final,
        "rerank": settings.rerank_mode if settings.rerank_enabled else "off",
        "min_relevance_score": settings.min_relevance_score,
        "glossary_expansion": settings.glossary_expansion,
        "article_lookup": settings.article_lookup,
        "context_expansion": settings.context_expansion,
        "context_neighbours": settings.context_neighbours,
        "max_context_tokens": settings.max_context_tokens,
        "chunk_max_tokens": settings.chunk_max_tokens,
    }


# --- ingest ---


def _run_ingest(force: bool) -> None:
    _ingest_state.update(status="running", started=time.time(), error=None)
    try:
        _ingest_state.update(status="done", result=ingest(settings, force=force,
                                                           embedder=embedder, store=store))
    except Exception as e:
        log.exception("ingest failed")
        _ingest_state.update(status="failed", error=str(e))
    finally:
        _ingest_lock.release()


@app.post("/ingest", status_code=202)
def start_ingest(force: bool = False) -> dict:
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "ingest already running")
    try:
        threading.Thread(target=_run_ingest, args=(force,), daemon=True).start()
    except Exception:
        _ingest_lock.release()  # never leave the lock held by a thread that never started
        raise
    return {"status": "started"}


@app.get("/ingest")
def ingest_status() -> dict:
    return _ingest_state


# --- retrieval ---


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 5, final: bool = False) -> dict:
    """`final=true` runs the full pipeline (gate, rerank, widen); otherwise raw candidates."""
    result = retriever.relevant(q) if final else retriever.retrieve(q)
    hits = result.hits[:k] if not final else result.hits
    return {
        "query": result.query,
        "expanded": result.expansion.added if result.expansion else [],
        "refs": [str(r) for r in result.refs],
        "candidates": result.candidates,
        "took_ms": result.took_ms,
        "gated": result.gated,
        "hits": [h.view() | {"text": str(h.payload.get("text", ""))[:500]} for h in hits],
    }


# --- reading the document ---


@app.get("/toc")
def toc() -> list[dict]:
    return library.toc()


@app.get("/article")
def article(doc_id: str, article: str) -> dict:
    found = library.article(doc_id, article)
    if found is None:
        raise HTTPException(404, f"{article} not found in {doc_id}")
    return found


@app.get("/source/{doc_id}")
def source(doc_id: str) -> FileResponse:
    """The original PDF. `inline` so the browser viewer can open it at #page=N."""
    path = library.source_pdf(doc_id)
    if path is None:
        raise HTTPException(404, "no source file for this document")
    # `filename=` rather than a hand-built Content-Disposition: the name is Vietnamese, HTTP
    # headers are latin-1, and Starlette knows to emit `filename*=utf-8''...` for that. Writing
    # the header by hand raised UnicodeEncodeError and the browser got a 500 instead of a PDF.
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        content_disposition_type="inline")


# --- chat ---


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = []


class Feedback(BaseModel):
    vote: str = Field(pattern="^(up|down)$")
    question: str = Field(max_length=2000)
    answer: str = Field(default="", max_length=8000)
    note: str = Field(default="", max_length=1000)
    sources: list[str] = []


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    def events():
        try:
            for event, data in chat_service.answer(req.message, [t.model_dump() for t in req.history]):
                yield _sse(event, data)
        except Exception as e:  # surface the failure in the stream
            log.exception("chat failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback", status_code=202)
def feedback(body: Feedback) -> dict:
    telemetry.feedback.append(body.model_dump())
    return {"ok": True}


@app.get("/metrics")
def metrics() -> dict:
    return telemetry.summary() | {
        "query_cache": {"hits": embedder.cache_hits, "misses": embedder.cache_misses},
        "points": store.count() if store.exists() else 0,
    }


# --- pages ---


def _page(name: str) -> FileResponse:
    return FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})


@app.get("/")
def index() -> FileResponse:
    return _page("index.html")


@app.get("/browse")
def browse() -> FileResponse:
    return _page("browse.html")


@app.get("/debug")
def debug() -> FileResponse:
    return _page("debug.html")
