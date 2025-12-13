from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


def now_s() -> float:
    return time.perf_counter()


@dataclass(frozen=True)
class NIMConfig:
    """
    Lightweight client config for localhost NIM endpoints.

    We intentionally keep this flexible because NIM deployments can expose
    OpenAI-compatible routes and/or NIM-specific routes depending on the container.
    """

    base_url: str = os.environ.get("NIM_BASE_URL", "http://localhost:8000")
    timeout_s: float = 60.0

    # Model ids as expected by the NIM server
    embed_model: str = os.environ.get("NIM_EMBED_MODEL", "nvidia/llama-3.1-nemotron-embedding")
    rerank_model: str = os.environ.get("NIM_RERANK_MODEL", "nvidia/llama-3.2-nemoretriever-500m-rerank-v2")
    gen_model: str = os.environ.get("NIM_GEN_MODEL", "qwen/qwen-2.5-7b-instruct")

    # Endpoints (customizable)
    embeddings_path: str = os.environ.get("NIM_EMBED_PATH", "/v1/embeddings")
    chat_path: str = os.environ.get("NIM_CHAT_PATH", "/v1/chat/completions")
    rerank_path: str = os.environ.get("NIM_RERANK_PATH", "/v1/rerank")

    # Optional auth
    api_key: str | None = os.environ.get("NIM_API_KEY") or None


class NIMClient:
    def __init__(self, cfg: NIMConfig):
        self.cfg = cfg

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            # Most OpenAI-compatible servers accept this.
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _url(self, path: str) -> str:
        return self.cfg.base_url.rstrip("/") + "/" + path.lstrip("/")

    def embed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        """
        Returns embeddings and latency (seconds).
        Expects an OpenAI-compatible embeddings response:
          { data: [ { embedding: [...] }, ... ] }
        """
        payload = {"model": self.cfg.embed_model, "input": texts}
        t0 = now_s()
        r = requests.post(self._url(self.cfg.embeddings_path), headers=self._headers(), json=payload, timeout=self.cfg.timeout_s)
        dt = now_s() - t0
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        embs = [item.get("embedding") for item in data]
        if any(e is None for e in embs):
            raise ValueError(f"Unexpected embeddings response shape: keys={list(j.keys())}")
        return embs, dt

    def embed_many(self, texts: list[str], batch_size: int = 64) -> tuple[list[list[float]], float]:
        """
        Batch helper around embed() to avoid request size limits.
        Returns (embeddings, total_latency_seconds).
        """
        all_embs: list[list[float]] = []
        total = 0.0
        bs = max(1, int(batch_size))
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            embs, dt = self.embed(chunk)
            all_embs.extend(embs)
            total += dt
        return all_embs, total

    def rerank(self, query: str, documents: list[str], top_n: int) -> tuple[list[int], float]:
        """
        Returns reranked indices into documents and latency.

        There is no universal OpenAI rerank spec, so we support the common NIM-ish shape:
          POST /v1/rerank { model, query, documents, top_n }
        Response expected:
          { results: [ { index: int, relevance_score: float }, ... ] }
        """
        payload = {"model": self.cfg.rerank_model, "query": query, "documents": documents, "top_n": int(top_n)}
        t0 = now_s()
        r = requests.post(self._url(self.cfg.rerank_path), headers=self._headers(), json=payload, timeout=self.cfg.timeout_s)
        dt = now_s() - t0
        r.raise_for_status()
        j = r.json()
        results = j.get("results") or j.get("data") or []
        idxs = []
        for item in results:
            if "index" in item:
                idxs.append(int(item["index"]))
            elif "document_index" in item:
                idxs.append(int(item["document_index"]))
        if not idxs:
            raise ValueError(f"Unexpected rerank response: keys={list(j.keys())}")
        return idxs, dt

    def chat(self, prompt: str, max_tokens: int = 200, temperature: float = 0.2) -> tuple[str, float]:
        """
        Returns text and latency.
        Expects OpenAI-compatible chat completions:
          { choices: [ { message: { content: ... } } ] }
        """
        payload = {
            "model": self.cfg.gen_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        t0 = now_s()
        r = requests.post(self._url(self.cfg.chat_path), headers=self._headers(), json=payload, timeout=self.cfg.timeout_s)
        dt = now_s() - t0
        r.raise_for_status()
        j = r.json()
        choices = j.get("choices") or []
        if not choices:
            raise ValueError(f"Unexpected chat response: keys={list(j.keys())}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content is None:
            # Some servers return {text: ...}
            content = choices[0].get("text")
        return str(content or "").strip(), dt


