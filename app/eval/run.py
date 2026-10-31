"""Measure retrieval against eval/questions.toml.

Opinions about retrieval are cheap; this makes them falsifiable. Every question names the
article that actually answers it, and the harness reports how often that article is found.

    python -m app.eval.run                 # current settings
    python -m app.eval.run --compare       # each technique on its own, against a baseline
    python -m app.eval.run --sweep         # trade-off curve for RELATIVE_CUTOFF
    python -m app.eval.run --json out.json # machine-readable, for a before/after diff

Metrics:
    recall@retrieve  an expected article is somewhere in the candidate list
    hit@final        it survives the gate, the rerank and the cut to TOP_K_FINAL
    MRR              1/rank of the first expected article in the final list

The chat model is never called: this measures retrieval, which is where answers are won or
lost. It needs Ollama (for embeddings) and Qdrant, so run it inside the stack.
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.rag.embeddings import OllamaEmbedder
from app.rag.retriever import Retriever
from app.rag.store import QdrantStore

# Next to the module, so it is copied into the image and found whatever the cwd.
QUESTIONS = Path(__file__).resolve().parent / "questions.toml"


@dataclass
class Case:
    ask: str
    expect: list[str]
    note: str = ""


def load(path: Path) -> list[Case]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [Case(q["ask"], list(q["expect"]), q.get("note", "")) for q in data["question"]]


def evaluate(retriever: Retriever, cases: list[Case]) -> dict:
    rows, started = [], time.monotonic()
    for case in cases:
        candidates = retriever.retrieve(case.ask)
        final = retriever.relevant(case.ask)
        in_candidates = any(h.payload.get("article") in case.expect for h in candidates.hits)
        ranks = [i for i, h in enumerate(final.hits, 1) if h.payload.get("article") in case.expect]
        rows.append({
            "ask": case.ask,
            "expect": case.expect,
            "recall": in_candidates,
            "hit": bool(ranks),
            "rank": ranks[0] if ranks else None,
            "got": [h.payload.get("article") for h in final.hits],
            "expanded": final.expansion.added if final.expansion else [],
            "ms": final.took_ms,
        })
    n = len(rows) or 1
    return {
        "questions": len(rows),
        "recall_at_retrieve": round(sum(r["recall"] for r in rows) / n, 3),
        "hit_at_final": round(sum(r["hit"] for r in rows) / n, 3),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
        "median_ms": sorted(r["ms"] for r in rows)[len(rows) // 2] if rows else 0,
        "avg_sources": round(sum(len(r["got"]) for r in rows) / n, 2),
        "wall_s": round(time.monotonic() - started, 1),
        "rows": rows,
    }


def build(settings: Settings) -> Retriever:
    return Retriever(settings, OllamaEmbedder(settings),
                     QdrantStore(settings.qdrant_url, settings.collection_name))


VARIANTS: dict[str, dict] = {
    "baseline (dense only)": dict(hybrid_search=False, rerank_enabled=False,
                                  glossary_expansion=False, article_lookup=False,
                                  context_expansion=False),
    "+ hybrid BM25": dict(rerank_enabled=False, glossary_expansion=False,
                          article_lookup=False, context_expansion=False),
    "+ glossary": dict(rerank_enabled=False, article_lookup=False, context_expansion=False),
    "+ article lookup": dict(rerank_enabled=False, context_expansion=False),
    "+ rerank (current)": dict(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=QUESTIONS)
    parser.add_argument("--compare", action="store_true", help="add one technique at a time")
    parser.add_argument("--sweep", action="store_true",
                        help="RELATIVE_CUTOFF vs quality: how few sources you can show "
                             "before the right article starts falling out")
    parser.add_argument("--json", type=Path, help="write the full result here")
    parser.add_argument("--verbose", action="store_true", help="list every miss")
    args = parser.parse_args()

    cases = load(args.file)
    base = get_settings().model_dump()

    if args.sweep:
        # This parameter cannot be tuned against the test stub embedder: its cosines spread
        # far wider than bge-m3's, so the same ratio means something different. Sweep it here,
        # on the real model, and keep the largest value that costs no hit@k.
        print(f"{'cutoff':>7}  recall  hit@k    MRR   nguồn TB")
        for cut in (0.0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            r = evaluate(build(Settings(**(base | {"relative_cutoff": cut}))), cases)
            mark = "  <- đang dùng" if abs(cut - base["relative_cutoff"]) < 1e-9 else ""
            print(f"{cut:>7.2f}  {r['recall_at_retrieve']:6.3f}  {r['hit_at_final']:5.3f}"
                  f"  {r['mrr']:5.3f}  {r['avg_sources']:7.2f}{mark}")
        return
    runs: dict[str, dict] = {}
    for name, overrides in (VARIANTS if args.compare else {"current": {}}).items():
        settings = Settings(**(base | overrides))
        runs[name] = evaluate(build(settings), cases)

    width = max(len(n) for n in runs)
    print(f"{'':{width}}  recall  hit@k    MRR   ms  nguồn TB")
    for name, result in runs.items():
        print(f"{name:{width}}  {result['recall_at_retrieve']:6.3f}  {result['hit_at_final']:5.3f}"
              f"  {result['mrr']:5.3f}  {result['median_ms']:4d}  {result['avg_sources']:7.2f}")

    last = list(runs.values())[-1]
    misses = [r for r in last["rows"] if not r["hit"]]
    if misses:
        print(f"\n{len(misses)}/{last['questions']} không vào được top {len(last['rows'][0]['got'])}:")
        for r in misses if args.verbose else misses[:10]:
            print(f"  - {r['ask']}\n    cần {r['expect']}, nhận {r['got']}")
    if args.json:
        args.json.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
