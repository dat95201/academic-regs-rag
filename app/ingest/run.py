"""Ingest: data/raw -> data/processed/*.md -> chunks -> embeddings -> Qdrant.

Idempotent. Point ids are deterministic and a manifest records the fingerprint of each
document already in Qdrant, so a second run with no changes does nothing:

    python -m app.ingest.run            # incremental
    python -m app.ingest.run --force    # re-embed everything
    python -m app.ingest.run --dry-run  # only show what would change

A document is re-embedded when its Markdown or the chunking settings change; documents whose
Markdown disappeared are deleted from Qdrant. Changing EMBED_MODEL or HYBRID_SEARCH recreates
the collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from qdrant_client import models

from app.config import Settings, get_settings
from app.ingest import chunk as chunking
from app.ingest.parse import prepare
from app.rag.embeddings import OllamaEmbedder, bm25_document, corpus_avg_len
from app.rag.store import DENSE, SPARSE, QdrantStore

log = logging.getLogger("ingest")

# Bump when chunk.py changes in a way that changes the chunks.
CHUNKER_VERSION = "2"


def fingerprint(md: Path, settings: Settings, avg_len: float) -> str:
    h = hashlib.sha256(md.read_bytes())
    # avg_len enters the BM25 weights, so it belongs in the fingerprint - but bucketed, or
    # adding one document would re-embed every other document for a drift of a few syllables.
    h.update(f"|{CHUNKER_VERSION}|{settings.chunk_max_tokens}|{settings.chunk_overlap_tokens}"
             f"|{settings.embed_model}|{settings.hybrid_search}|{round(avg_len / 50)}".encode())
    return h.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}


def to_points(chunks: list[chunking.Chunk], vectors: list[list[float]], settings: Settings,
              avg_len: float):
    for c, vector in zip(chunks, vectors, strict=True):
        vec: dict = {DENSE: vector}
        if settings.hybrid_search:
            indices, values = bm25_document(c.embed_text, avg_len)
            vec[SPARSE] = models.SparseVector(indices=indices, values=values)
        payload = asdict(c) | {"embed_model": settings.embed_model}
        yield models.PointStruct(id=c.chunk_id, vector=vec, payload=payload)


def ingest(settings: Settings | None = None, force: bool = False, dry_run: bool = False,
           embedder: OllamaEmbedder | None = None, store: QdrantStore | None = None) -> dict:
    settings = settings or get_settings()
    started = time.monotonic()
    files = prepare(Path(settings.raw_dir), Path(settings.processed_dir))
    embedder = embedder or OllamaEmbedder(settings)
    store = store or QdrantStore(settings.qdrant_url, settings.collection_name)

    manifest_path = Path(settings.ingest_manifest)
    manifest = load_manifest(manifest_path)
    if dry_run:
        fresh = False
    else:
        fresh = store.ensure(embedder.dimension(), settings.hybrid_search, settings.embed_model)
    if fresh or force or store.count() == 0:
        manifest = {}  # the collection does not hold what the manifest describes

    # Chunk everything first: the BM25 document weights need the average chunk length of the
    # whole corpus, not of whichever document happens to be processed first.
    chunked = [(md, chunking.chunk_file(md, settings.chunk_max_tokens,
                                        settings.chunk_overlap_tokens)) for md in files]
    avg_len = corpus_avg_len([c.embed_text for _, cs in chunked for c in cs])
    log.info("%d document(s), %d chunks, average length %.0f syllables",
             len(chunked), sum(len(cs) for _, cs in chunked), avg_len)

    stats = {"added_or_updated": [], "unchanged": [], "removed": [], "chunks": 0,
             "avg_chunk_len": round(avg_len)}
    seen: set[str] = set()
    for md, chunks in chunked:
        if not chunks:
            log.warning("%s: no chunks", md.name)
            continue
        doc_id = chunks[0].doc_id
        seen.add(doc_id)
        fp = fingerprint(md, settings, avg_len)
        if manifest.get(doc_id) == fp:
            stats["unchanged"].append(doc_id)
            continue
        log.info("%s: %d chunks, embedding with %s ...", doc_id, len(chunks), settings.embed_model)
        stats["added_or_updated"].append(doc_id)
        stats["chunks"] += len(chunks)
        if dry_run:
            continue
        vectors = embedder.embed([c.embed_text for c in chunks])
        store.delete_doc(doc_id)  # drops chunks that no longer exist after an edit
        store.upsert(to_points(chunks, vectors, settings, avg_len))
        manifest[doc_id] = fp
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for doc_id in sorted(set(manifest) - seen):
        log.info("%s: source removed, deleting its chunks", doc_id)
        stats["removed"].append(doc_id)
        if not dry_run:
            store.delete_doc(doc_id)
            del manifest[doc_id]
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    stats["points"] = store.count()
    stats["seconds"] = round(time.monotonic() - started, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-embed every document")
    parser.add_argument("--dry-run", action="store_true", help="show what would change")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    stats = ingest(force=args.force, dry_run=args.dry_run)
    print(
        f"Ingest xong trong {stats['seconds']}s: "
        f"{len(stats['added_or_updated'])} văn bản cập nhật ({stats['chunks']} chunk), "
        f"{len(stats['unchanged'])} không đổi, {len(stats['removed'])} bị xoá. "
        f"Qdrant hiện có {stats['points']} point."
    )


if __name__ == "__main__":
    main()
