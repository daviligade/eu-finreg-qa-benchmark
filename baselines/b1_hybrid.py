"""
B1 — Hybrid RAG baseline (dense + BM25 with RRF fusion, no AKN).

Design
------
- Two retrievers: dense (Ollama embeddings, cosine) + sparse (BM25Okapi).
- Fused with Reciprocal Rank Fusion (RRF), k_rrf=60 (standard).
- Chunks are plain PDF word windows (same as B0) — NO article awareness,
  NO instrument-aware diversity, NO skeleton validation.
- Single prompt template; no retry loop.

Why this baseline exists
------------------------
B0 isolates "dense retrieval". B1 isolates "hybrid retrieval *without*
structural knowledge". If v2 > B1, the delta is attributable to the
structural / AKN / intent-routing machinery — NOT to hybrid retrieval itself.
This is the B1 that the ablation table in the protocol targets.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from baselines._common import (
    Chunk,
    EmbedCache,
    chat_complete,
    chunk_pdf,
    embed_texts,
    retry,
)

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a regulatory assistant. Answer the user's question using ONLY the passages below.

Rules:
- Cite every claim with [Ck] where k is the passage number.
- If the passages do not contain the information needed, reply exactly: "I cannot answer from the provided material."
- Do not invent article numbers or facts.

Passages:
{passages}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# BM25 — use rank_bm25 when available, else a tiny deterministic fallback.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def _tok(s: str) -> List[str]:
    return [w.lower() for w in _TOKEN_RE.findall(s)]


class _FallbackBM25:
    """Minimal BM25Okapi reimplementation. Used only if rank_bm25 is missing.

    Kept here so the baseline has one fewer hard dependency; the math is
    the standard k1=1.5, b=0.75 formulation.
    """
    def __init__(self, docs_tok: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tok)
        self.doc_len = np.array([len(d) for d in docs_tok], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        self._df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        for d in docs_tok:
            tf: dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1

    def get_scores(self, q_tok: List[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for t in q_tok:
            df = self._df.get(t, 0)
            if df == 0:
                continue
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for i, tf in enumerate(self._tf):
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(1e-6, self.avgdl))
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


def _make_bm25(docs_tok: List[List[str]]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(docs_tok)
    except ImportError:
        log.info("rank_bm25 not installed — using fallback BM25")
        return _FallbackBM25(docs_tok)


# ---------------------------------------------------------------------------
# Config + Pipeline
# ---------------------------------------------------------------------------

@dataclass
class B1Config:
    answerer_model: str
    embed_model: str
    ollama_host: str = "http://localhost:11434"
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 6
    k_rrf: int = 60
    chunk_words: int = 220
    overlap_words: int = 40
    temperature: float = 0.0
    num_ctx: int = 8192
    timeout_s: int = 300
    cache_path: str = ".cache/b1_embed.jsonl"


class Pipeline:
    def __init__(self, cfg: B1Config, corpus: List[dict]):
        self.cfg = cfg
        self._build_index(corpus)

    # ------------------------------------------------------------------
    def _build_index(self, corpus: List[dict]) -> None:
        all_chunks: List[Chunk] = []
        cid = 1
        for src in corpus:
            chunks = chunk_pdf(
                src["pdf"],
                instrument_id=src["id"],
                chunk_words=self.cfg.chunk_words,
                overlap_words=self.cfg.overlap_words,
                start_chunk_id=cid,
            )
            all_chunks.extend(chunks)
            cid += len(chunks)
        if not all_chunks:
            raise RuntimeError("B1: no chunks produced from corpus.")
        log.info("B1 index: %d chunks across %d instruments", len(all_chunks), len(corpus))

        cache = EmbedCache(self.cfg.cache_path, self.cfg.embed_model)
        self._chunks = all_chunks
        self._embeds = embed_texts(
            [c.text for c in all_chunks],
            model=self.cfg.embed_model,
            host=self.cfg.ollama_host,
            cache=cache,
        )
        self._bm25 = _make_bm25([_tok(c.text) for c in all_chunks])

    # ------------------------------------------------------------------
    def _dense_topk(self, question: str) -> List[int]:
        cache = EmbedCache(self.cfg.cache_path, self.cfg.embed_model)
        q_vec = embed_texts([question], model=self.cfg.embed_model,
                             host=self.cfg.ollama_host, cache=cache)[0]
        sims = self._embeds @ q_vec
        return list(np.argsort(-sims)[: self.cfg.top_k_dense])

    def _bm25_topk(self, question: str) -> List[int]:
        scores = np.asarray(self._bm25.get_scores(_tok(question)), dtype=float)
        return list(np.argsort(-scores)[: self.cfg.top_k_bm25])

    def _rrf_fuse(self, dense_ids: List[int], sparse_ids: List[int]) -> List[int]:
        """Reciprocal Rank Fusion: score = sum_r 1 / (k + rank_r).
        Higher is better. Returns top_k_final indices."""
        fused: dict[int, float] = {}
        for r, idx in enumerate(dense_ids, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.cfg.k_rrf + r)
        for r, idx in enumerate(sparse_ids, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.cfg.k_rrf + r)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        return [idx for idx, _ in ordered[: self.cfg.top_k_final]]

    # ------------------------------------------------------------------
    def _format_passages(self, chunks: List[Chunk]) -> str:
        return "\n\n".join(
            f"[C{i+1}] (source: {c.instrument}, page {c.page}) {c.text}"
            for i, c in enumerate(chunks)
        )

    # ------------------------------------------------------------------
    def answer(self, question: str, intent: Optional[str] = None) -> dict:
        t0 = time.time()
        dense_ids = self._dense_topk(question)
        sparse_ids = self._bm25_topk(question)
        final_ids = self._rrf_fuse(dense_ids, sparse_ids)
        chunks = [self._chunks[i] for i in final_ids]
        passages = self._format_passages(chunks)
        prompt = PROMPT_TEMPLATE.format(passages=passages, question=question)
        text = retry(
            chat_complete,
            model=self.cfg.answerer_model,
            prompt=prompt,
            host=self.cfg.ollama_host,
            temperature=self.cfg.temperature,
            num_ctx=self.cfg.num_ctx,
            timeout_s=self.cfg.timeout_s,
        )
        refused = "cannot answer from the provided material" in text.lower()
        citations = [
            {
                "citation_id": i + 1,
                "instrument": c.instrument,
                "article": "",
                "paragraph": None,
                "point": None,
                "passage": c.text,
            }
            for i, c in enumerate(chunks)
        ]
        return {
            "answer_text": text,
            "refused": refused,
            "citations": citations,
            "retrieved_contexts": [c.text for c in chunks],
            "latency_s": time.time() - t0,
            "retries": 0,
        }


def build(manifest: dict, corpus: List[dict]) -> Pipeline:
    cfg = B1Config(
        answerer_model=manifest["models"]["answerer"]["name"],
        embed_model=manifest["models"]["embed"]["name"],
        ollama_host=manifest["models"]["answerer"]["host"],
        temperature=manifest["models"]["answerer"].get("temperature", 0.0),
        num_ctx=manifest["models"]["answerer"].get("num_ctx", 8192),
        timeout_s=manifest["models"]["answerer"].get("timeout_s", 300),
        cache_path=manifest["paths"].get("cache_dir", ".cache") + "/b1_embed.jsonl",
    )
    return Pipeline(cfg, corpus)
