"""HTTP API and the chat page.

    GET  /                 chat UI (app/static/index.html)
    POST /chat             {"message", "history"} -> Server-Sent Events: query?, token*, sources, done
    GET  /livez            process is up (container healthcheck)
    GET  /health           Ollama + models + Qdrant + collection ready? 200 or 503
    POST /ingest           start an ingest in the background
    GET  /ingest           status of the last ingest
    GET  /search?q=...     hybrid retrieval results, for checking the index
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

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
app = FastAPI(title="academic-regs-rag")
store = QdrantStore(settings.qdrant_url, settings.collection_name)
embedder = OllamaEmbedder(settings)
retriever = Retriever(settings, embedder, store)
chat_service = ChatService(settings, retriever, LLMClient(settings))

STATIC = Path(__file__).resolve().parent.parent / "static"
_ingest_lock = threading.Lock()
_ingest_state: dict = {"status": "idle"}


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
    checks: dict = {}
    try:
        models = _ollama_models()
        checks["ollama"] = "ok"
        for key, name in (("llm_model", settings.llm_model), ("embed_model", settings.embed_model)):
            checks[key] = name if name in models else f"missing: ollama pull {name}"
    except Exception as e:  # noqa: BLE001 - report any connection problem
        checks["ollama"] = f"unreachable at {settings.ollama_url}: {e}"
    try:
        checks["qdrant"] = "ok"
        checks["points"] = store.count()
    except Exception as e:  # noqa: BLE001
        checks["qdrant"] = f"unreachable at {settings.qdrant_url}: {e}"
    checks["ingest"] = _ingest_state.get("status")
    ready = (
        checks["ollama"] == "ok"
        and checks.get("llm_model") == settings.llm_model
        and checks.get("embed_model") == settings.embed_model
        and checks["qdrant"] == "ok"
        and checks.get("points", 0) > 0
    )
    return JSONResponse({"ready": ready, **checks}, status_code=200 if ready else 503)


def _run_ingest(force: bool) -> None:
    _ingest_state.update(status="running", started=time.time(), error=None)
    try:
        _ingest_state.update(status="done", result=ingest(settings, force=force,
                                                           embedder=embedder, store=store))
    except Exception as e:  # noqa: BLE001
        log.exception("ingest failed")
        _ingest_state.update(status="failed", error=str(e))
    finally:
        _ingest_lock.release()


@app.post("/ingest", status_code=202)
def start_ingest(force: bool = False) -> dict:
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "ingest already running")
    threading.Thread(target=_run_ingest, args=(force,), daemon=True).start()
    return {"status": "started"}


@app.get("/ingest")
def ingest_status() -> dict:
    return _ingest_state


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 5) -> list[dict]:
    hits = retriever.search(q, limit=max(k, settings.top_k_retrieve))[:k]
    return [
        {
            "score": round(h.score, 4),
            "cosine": round(h.dense_score, 4),
            "article": h.payload.get("article"),
            "sections": h.payload.get("sections"),
            "page": h.payload.get("page"),
            "source_refs": h.payload.get("source_refs"),
            "text": h.payload.get("text", "")[:500],
        }
        for h in hits
    ]


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = []


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    def events():
        try:
            for event, data in chat_service.answer(req.message, [t.model_dump() for t in req.history]):
                yield _sse(event, data)
        except Exception as e:  # noqa: BLE001 - surface the failure in the stream
            log.exception("chat failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
