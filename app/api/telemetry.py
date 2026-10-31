"""What was asked, how long it took, and whether the corpus could answer it.

One JSON line per turn in data/logs/queries.jsonl and one per thumbs-up/down in
feedback.jsonl. This is the cheapest evaluation signal a deployed RAG system has: the
questions that return nothing are exactly the gaps in the corpus or in the glossary, and
`summary()` puts them on /metrics instead of leaving them in a file nobody opens.

Writes are best-effort: a failure here never breaks an answer.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("telemetry")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


class JsonLog:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        if not self.enabled:
            return
        record = {"ts": round(time.time(), 3), **record}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("cannot write %s", self.path)

    def read(self, limit: int = 2000) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


class Telemetry:
    def __init__(self, directory: Path, enabled: bool = True):
        self.queries = JsonLog(directory / "queries.jsonl", enabled)
        self.feedback = JsonLog(directory / "feedback.jsonl", enabled)

    def summary(self, limit: int = 2000) -> dict:
        turns = self.queries.read(limit)
        votes = self.feedback.read(limit)
        answered = [t for t in turns if t.get("found")]
        latencies = [float(t["latency_ms"]) for t in turns if t.get("latency_ms")]
        retrieval = [float(t["retrieval_ms"]) for t in turns if t.get("retrieval_ms")]
        unanswered: dict[str, int] = {}
        for t in turns:
            if not t.get("found"):
                q = str(t.get("question", "")).strip()
                if q:
                    unanswered[q] = unanswered.get(q, 0) + 1
        flagged = [t for t in turns if t.get("unsupported_citations")]
        return {
            "turns": len(turns),
            "answered": len(answered),
            "answer_rate": round(len(answered) / len(turns), 3) if turns else None,
            "latency_ms": {"p50": round(_percentile(latencies, 50)),
                           "p95": round(_percentile(latencies, 95))},
            "retrieval_ms": {"p50": round(_percentile(retrieval, 50)),
                             "p95": round(_percentile(retrieval, 95))},
            "unsupported_citation_turns": len(flagged),
            "votes": {"up": sum(v.get("vote") == "up" for v in votes),
                      "down": sum(v.get("vote") == "down" for v in votes)},
            "top_unanswered": sorted(unanswered.items(), key=lambda kv: -kv[1])[:20],
        }
