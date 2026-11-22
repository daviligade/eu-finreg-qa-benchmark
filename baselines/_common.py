"""
Shared utilities for the B0 and B1 baseline pipelines.

Kept deliberately minimal: the baselines are supposed to be *simple* — any
complexity belongs to the v1/v2 systems. The point of a baseline is that
readers can trace what it does in one sitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    instrument: str
    text: str
    page: Optional[int] = None


def _read_pdf_text(pdf_path: str) -> List[tuple]:
    """Return a list of (page_number, text) tuples for the whole PDF.

    Falls back across pdfplumber -> pypdf so the script runs even if one
    library is missing. Quietly returns an empty list if both fail.
    """
    pages: List[tuple] = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages.append((i, txt))
        if pages:
            return pages
    except Exception as e:  # noqa: BLE001
        log.debug("pdfplumber failed on %s: %s", pdf_path, e)

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append((i, txt))
    except Exception as e:  # noqa: BLE001
        log.warning("pypdf failed on %s: %s", pdf_path, e)

    return pages


def chunk_pdf(
    pdf_path: str,
    instrument_id: str,
    *,
    chunk_words: int = 220,
    overlap_words: int = 40,
    start_chunk_id: int = 1,
) -> List[Chunk]:
    """Split a PDF into overlapping word-window chunks.

    This is the chunking that B0/B1 use. It's deliberately naive compared to
    v1/v2 (which work at AKN article granularity).
    """
    chunks: List[Chunk] = []
    cid = start_chunk_id
    for page_no, page_text in _read_pdf_text(pdf_path):
        # Collapse whitespace; keep line breaks as spaces.
        norm = re.sub(r"\s+", " ", page_text).strip()
        words = norm.split()
        if not words:
            continue
        step = max(1, chunk_words - overlap_words)
        for start in range(0, len(words), step):
            window = words[start:start + chunk_words]
            if len(window) < 40:  # skip tiny trailing fragments
                continue
            chunks.append(Chunk(
                chunk_id=cid,
                instrument=instrument_id,
                text=" ".join(window),
                page=page_no,
            ))
            cid += 1
    return chunks


# ---------------------------------------------------------------------------
# Tiny embedding cache (local file, keyed by sha256(text|model))
# ---------------------------------------------------------------------------

class EmbedCache:
    """Naive JSONL-backed cache — good enough for a few thousand chunks.

    Not thread-safe; baselines run single-threaded by design.
    """

    def __init__(self, path: str, model: str):
        self.path = Path(path)
        self.model = model
        self._mem: dict[str, list] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self._mem[row["k"]] = row["v"]
                    except Exception:
                        continue

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def get(self, text: str) -> Optional[list]:
        return self._mem.get(self._key(text))

    def set(self, text: str, vec: list) -> None:
        k = self._key(text)
        self._mem[k] = vec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"k": k, "v": vec}) + "\n")


# ---------------------------------------------------------------------------
# Ollama client helpers (embed + chat)
# ---------------------------------------------------------------------------

def _ollama():
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "The `ollama` Python package is required. Install with "
            "`pip install ollama`."
        ) from e
    return ollama


def embed_texts(
    texts: Sequence[str],
    *,
    model: str,
    host: str = "http://localhost:11434",
    cache: Optional[EmbedCache] = None,
    batch: int = 16,
) -> np.ndarray:
    """Embed a list of texts with an Ollama embedding model.

    Returns an (n, d) float32 array, L2-normalized for cosine = dot product.
    """
    ollama = _ollama()
    client = ollama.Client(host=host)

    out: List[list] = []
    to_query: List[tuple] = []  # (index, text)
    for i, t in enumerate(texts):
        cached = cache.get(t) if cache else None
        if cached is not None:
            out.append(cached)
        else:
            out.append(None)  # placeholder
            to_query.append((i, t))

    for i, text in to_query:
        try:
            resp = client.embeddings(model=model, prompt=text)
        except Exception as e:
            raise RuntimeError(f"Ollama embeddings failed: {e}") from e
        vec = list(resp["embedding"])
        out[i] = vec
        if cache:
            cache.set(text, vec)

    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def chat_complete(
    *,
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    system: Optional[str] = None,
    temperature: float = 0.0,
    num_ctx: int = 8192,
    timeout_s: int = 300,
) -> str:
    """Single-turn chat completion via Ollama. Returns the assistant text.

    Not streamed — we want the full string for downstream parsing. The
    `timeout_s` parameter is advisory: the Ollama client honors it at the
    HTTP layer.
    """
    ollama = _ollama()
    client = ollama.Client(host=host, timeout=timeout_s)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat(
        model=model,
        messages=messages,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "seed": 13,
        },
    )
    return resp["message"]["content"]


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------

def retry(fn, *args, max_tries: int = 3, backoff_s: float = 2.0, **kwargs):
    """Retry `fn` with exponential backoff. Used to paper over transient
    Ollama hiccups (model loading, busy, etc.). We keep this tiny on purpose
    because noisy retry logic is where latency bugs hide.
    """
    last: Optional[Exception] = None
    for i in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("retry %d/%d: %s", i + 1, max_tries, e)
            time.sleep(backoff_s * (2 ** i))
    raise RuntimeError(f"All {max_tries} retries failed: {last}")
