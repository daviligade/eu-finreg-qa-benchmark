"""Baseline RAG pipelines for evaluation.

B0 — Vanilla RAG (dense-only, no structure awareness)
B1 — Hybrid RAG (dense + BM25 + RRF, but treats the PDF as a flat document;
     no AKN, no article-level pinpointing, no instrument-aware retrieval)

Both baselines share the same Pipeline interface defined in
evaluation/harness.py so they drop directly into the evaluation loop.
"""
