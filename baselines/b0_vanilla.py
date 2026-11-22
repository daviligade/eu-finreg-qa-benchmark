"""
B0 — Vanilla RAG baseline.

Design
------
- Dense retrieval only (cosine similarity over Ollama embeddings).
- No BM25, no RRF, no instrument awareness, no AKN.
- A single prompt template with NO retry loop.
- No refusal logic beyond passing the retrieved passages through.


Interface
---------
Conforms to the `Pipeline` protocol expected by evaluation/harness.py:

    class Pipeline:
        def __init__(self, config: dict, corpus: list[dict]): ...
        def answer(self, question: str, intent: str | None = None) -> dict
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
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


@dataclass
class B0Config:
    answerer_model: str
    embed_model: str
    ollama_host: str = "http://localhost:11434"
    top_k: int = 6
    chunk_words: int = 220
    overlap_words: int = 40
    temperature: float = 0.0
    num_ctx: int = 8192
    timeout_s: int = 300
    cache_path: str = ".cache/b0_embed.jsonl"


class Pipeline:
    """Vanilla dense-only RAG."""

    def __init__(self, cfg: B0Config, corpus: List[dict]):
        """`corpus` is the list of {id, pdf, akn} entries from manifest.yaml."""
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
            raise RuntimeError("B0: no chunks produced from corpus — check PDF paths.")
        log.info("B0 index: %d chunks across %d instruments", len(all_chunks), len(corpus))

        cache = EmbedCache(self.cfg.cache_path, self.cfg.embed_model)
        self._chunks = all_chunks
        self._embeds = embed_texts(
            [c.text for c in all_chunks],
            model=self.cfg.embed_model,
            host=self.cfg.ollama_host,
            cache=cache,
        )

    # ------------------------------------------------------------------
    def _retrieve(self, question: str) -> List[Chunk]:
        cache = EmbedCache(self.cfg.cache_path, self.cfg.embed_model)
        q_vec = embed_texts([question], model=self.cfg.embed_model,
                             host=self.cfg.ollama_host, cache=cache)[0]
        sims = self._embeds @ q_vec
        # Top-k indices (descending)
        top = np.argsort(-sims)[: self.cfg.top_k]
        return [self._chunks[i] for i in top]

    # ------------------------------------------------------------------
    def _format_passages(self, chunks: List[Chunk]) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            lines.append(f"[C{i}] (source: {c.instrument}, page {c.page}) {c.text}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    def answer(self, question: str, intent: Optional[str] = None) -> dict:
        t0 = time.time()
        chunks = self._retrieve(question)
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
                "article": "",         # B0 has no article info
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


# ---------------------------------------------------------------------------
# Factory used by harness
# ---------------------------------------------------------------------------

def build(manifest: dict, corpus: List[dict]) -> Pipeline:
    cfg = B0Config(
        answerer_model=manifest["models"]["answerer"]["name"],
        embed_model=manifest["models"]["embed"]["name"],
        ollama_host=manifest["models"]["answerer"]["host"],
        temperature=manifest["models"]["answerer"].get("temperature", 0.0),
        num_ctx=manifest["models"]["answerer"].get("num_ctx", 8192),
        timeout_s=manifest["models"]["answerer"].get("timeout_s", 300),
        cache_path=manifest["paths"].get("cache_dir", ".cache") + "/b0_embed.jsonl",
    )
    return Pipeline(cfg, corpus)
