"""Chat model client over the OpenAI-compatible API (Ollama `/v1`, vLLM, ...)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from app.config import Settings

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class ThinkFilter:
    """Drops <think>...</think> blocks from a token stream (Qwen3 emits an empty one even
    with /no_think). Holds back text that might be the start of a tag."""

    def __init__(self) -> None:
        self.buffer = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.buffer += text
        out = []
        while self.buffer:
            if self.inside:
                end = self.buffer.find(THINK_CLOSE)
                if end < 0:
                    self.buffer = self.buffer[-len(THINK_CLOSE):]
                    break
                self.buffer = self.buffer[end + len(THINK_CLOSE):].lstrip()
                self.inside = False
                continue
            start = self.buffer.find(THINK_OPEN)
            if start >= 0:
                out.append(self.buffer[:start])
                self.buffer = self.buffer[start + len(THINK_OPEN):]
                self.inside = True
                continue
            # keep a possible partial "<think" at the end for the next chunk
            keep = next((i for i in range(1, len(THINK_OPEN)) if self.buffer.endswith(THINK_OPEN[:i])), 0)
            out.append(self.buffer[: len(self.buffer) - keep])
            self.buffer = self.buffer[len(self.buffer) - keep:]
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self.buffer = ("" if self.inside else self.buffer), ""
        return rest


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.Client(
            base_url=settings.llm_url,
            timeout=httpx.Timeout(settings.ollama_timeout_s, connect=10),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    def _payload(self, messages: list[dict], stream: bool, temperature: float | None) -> dict:
        messages = [dict(m) for m in messages]
        if self.settings.llm_disable_thinking and messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += " /no_think"
        return {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "stream": stream,
        }

    def stream(self, messages: list[dict], temperature: float | None = None) -> Iterator[str]:
        think = ThinkFilter()
        payload = self._payload(messages, True, temperature)
        with self.client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:300]}")
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                text = think.feed(delta.get("content") or "")
                if text:
                    yield text
        tail = think.flush()
        if tail:
            yield tail

    def complete(self, messages: list[dict], temperature: float | None = None) -> str:
        resp = self.client.post("/chat/completions", json=self._payload(messages, False, temperature))
        resp.raise_for_status()
        think = ThinkFilter()
        content = resp.json()["choices"][0]["message"].get("content") or ""
        return (think.feed(content) + think.flush()).strip()
