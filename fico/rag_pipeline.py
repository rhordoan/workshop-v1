from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np


Mode = Literal["local", "nim"]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    title: str | None = None
    tenant_id: str | None = None
    doc_type: str | None = None
    tags: list[str] | None = None


@dataclass(frozen=True)
class RAGConfig:
    mode: Mode
    top_k: int = 10
    rerank_top_k: int = 5
    max_context_chars: int = 6000
    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 150
    # Generation
    max_new_tokens: int = 180
    temperature: float = 0.2
    # If you want tenant-aware prompts, set this
    tenant_id: str | None = None


@dataclass
class RAGResult:
    answer: str
    contexts: list[dict[str, Any]]
    timings_s: dict[str, float]
    debug: dict[str, Any]


def now_s() -> float:
    return time.perf_counter()


class Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def time(self, key: str, fn: Callable[[], Any]) -> Any:
        t0 = now_s()
        out = fn()
        self.timings[key] = self.timings.get(key, 0.0) + (now_s() - t0)
        return out


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(50, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks_from_df(df) -> list[Chunk]:
    """
    Expects df columns produced by our generator:
      - doc_id, title, body_redacted/body, tenant_id, doc_type, tags
    """
    chunks: list[Chunk] = []
    for _, row in df.iterrows():
        doc_id = str(row.get("doc_id"))
        title = row.get("title")
        tenant_id = row.get("tenant_id")
        doc_type = row.get("doc_type")
        tags = row.get("tags")

        body = row.get("body_redacted") or row.get("body") or ""
        parts = chunk_text(str(body), chunk_size=900, overlap=150)
        for j, part in enumerate(parts):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::c{j:03d}",
                    text=part,
                    title=str(title) if title is not None else None,
                    tenant_id=str(tenant_id) if tenant_id is not None else None,
                    doc_type=str(doc_type) if doc_type is not None else None,
                    tags=list(tags) if isinstance(tags, list) else None,
                )
            )
    return chunks


def normalize_rows(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def _cosine_topk(E_norm: np.ndarray, q_norm: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = E_norm @ q_norm
    k = min(int(k), sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


class VectorIndex:
    """
    Lightweight cosine-similarity index; uses FAISS if available, otherwise numpy.
    This keeps the workshop runnable even if faiss installation is blocked.
    """

    def __init__(self, E_norm: np.ndarray, use_faiss: bool = True):
        self.E_norm = E_norm.astype(np.float32)
        self.use_faiss = use_faiss
        self._faiss = None
        self._index = None

        if use_faiss:
            try:
                import faiss  # type: ignore

                self._faiss = faiss
                d = self.E_norm.shape[1]
                index = faiss.IndexFlatIP(d)  # cosine if vectors are normalized
                index.add(self.E_norm)
                self._index = index
            except Exception:
                self.use_faiss = False

    def search(self, q_norm: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q_norm = q_norm.astype(np.float32).reshape(1, -1)
        k = min(int(k), self.E_norm.shape[0])
        if self.use_faiss and self._index is not None:
            D, I = self._index.search(q_norm, k)
            return I[0], D[0]
        idx, sims = _cosine_topk(self.E_norm, q_norm[0], k)
        return idx, sims


def make_prompt(question: str, contexts: list[Chunk]) -> str:
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        header = f"[{i}] doc_id={c.doc_id} chunk_id={c.chunk_id}"
        if c.title:
            header += f" title={c.title}"
        ctx_lines.append(header + "\n" + c.text)

    ctx = "\n\n".join(ctx_lines)
    return (
        "You are a careful assistant. Answer the user's question using ONLY the provided context.\n"
        "If the context is insufficient, say you don't know.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "ANSWER:"
    )


def default_cache_dir() -> str:
    return os.path.join(os.path.dirname(__file__), ".module_c_cache")


def cache_paths(cache_dir: str, name: str) -> dict[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, name)
    return {
        "chunks_jsonl": base + "_chunks.jsonl",
        "embeddings_npy": base + "_chunk_embeddings.npy",
        "index_meta_json": base + "_index_meta.json",
    }


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out







