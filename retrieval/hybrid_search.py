"""
retrieval/hybrid_search.py — Hybrid BM25 + FAISS retrieval with page-indexed results

Two complementary retrieval modes:
  1. Dense vector search (FAISS) — semantic similarity via sentence-transformers
  2. Sparse BM25 search (vectorless) — keyword/term frequency matching
  3. Combined via Reciprocal Rank Fusion (RRF) for robust hybrid scoring
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import BM25_WEIGHT, EMBEDDING_WEIGHT, TOP_K_RETRIEVAL, EMBEDDING_MODEL

log = logging.getLogger(__name__)

# Module-level caches for loaded indexes
_model_cache: Optional[SentenceTransformer] = None
_index_cache: dict = {}   # subfolder_path → {"faiss", "bm25", "chunks", "page_index"}


def get_embedding_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache


def load_subfolder_index(subfolder: Path) -> dict | None:
    """Load FAISS + BM25 + chunks for a subfolder (cached)."""
    key = str(subfolder)
    if key in _index_cache:
        return _index_cache[key]

    index_dir = subfolder / "index"
    faiss_path = index_dir / "faiss.index"
    bm25_path = index_dir / "bm25.pkl"
    chunks_path = index_dir / "chunks.json"

    if not (faiss_path.exists() and bm25_path.exists() and chunks_path.exists()):
        return None

    try:
        faiss_index = faiss.read_index(str(faiss_path))
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        with open(chunks_path, encoding="utf-8") as f:
            registry = json.load(f)

        _index_cache[key] = {
            "faiss": faiss_index,
            "bm25": bm25,
            "chunks": registry["chunks"],
            "page_index": registry.get("page_index", {}),
            "subfolder": str(subfolder),
        }
        return _index_cache[key]
    except Exception as e:
        log.error(f"Failed to load index for {subfolder}: {e}")
        return None


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


def vector_search(query: str, index_data: dict, top_k: int) -> list[dict]:
    """Dense semantic search via FAISS. Returns ranked chunks with scores."""
    model = get_embedding_model()
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    faiss_index: faiss.Index = index_data["faiss"]
    k = min(top_k, faiss_index.ntotal)
    if k == 0:
        return []

    scores, ids = faiss_index.search(query_vec, k)
    results = []
    for rank, (score, chunk_id) in enumerate(zip(scores[0], ids[0])):
        if chunk_id < 0:
            continue
        chunk = index_data["chunks"][chunk_id].copy()
        chunk["vector_score"] = float(score)
        chunk["vector_rank"] = rank
        chunk["source_type"] = "vector"
        results.append(chunk)
    return results


def bm25_search(query: str, index_data: dict, top_k: int) -> list[dict]:
    """Sparse BM25 (vectorless) keyword search. Returns ranked chunks with scores."""
    from rank_bm25 import BM25Okapi
    bm25: BM25Okapi = index_data["bm25"]
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] <= 0:
            continue
        chunk = index_data["chunks"][idx].copy()
        chunk["bm25_score"] = float(scores[idx])
        chunk["bm25_rank"] = rank
        chunk["source_type"] = "bm25"
        results.append(chunk)
    return results


def get_page_chunks(doc_name: str, page_num: int, index_data: dict) -> list[dict]:
    """
    Page indexing: directly fetch all chunks from a specific doc page.
    Used for reference traversal and precise citation lookup.
    """
    key = f"{doc_name}::p{page_num}"
    chunk_ids = index_data["page_index"].get(key, [])
    return [index_data["chunks"][cid] for cid in chunk_ids if cid < len(index_data["chunks"])]


def hybrid_search(
    query: str,
    subfolders: list[Path],
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Hybrid retrieval across multiple subfolders using RRF fusion.
    
    For each subfolder:
      - Dense vector search (FAISS, semantic)
      - Sparse BM25 search (vectorless, keyword)
    Results are merged via Reciprocal Rank Fusion then deduplicated by chunk_id.
    
    Each result includes: text, doc, page, section, clause, act,
    vector_score, bm25_score, hybrid_score, source_subfolder.
    """
    all_results: dict[str, dict] = {}  # unique_key → chunk dict with scores

    for sf in subfolders:
        index_data = load_subfolder_index(sf)
        if index_data is None:
            continue

        # Dense vector retrieval
        vec_results = vector_search(query, index_data, top_k)
        # Sparse BM25 retrieval
        bm25_results = bm25_search(query, index_data, top_k)

        # RRF fusion — collect RRF scores per chunk
        rrf_scores: dict[int, float] = {}

        for rank, chunk in enumerate(vec_results):
            cid = chunk["chunk_id"]
            rrf_scores.setdefault(cid, 0.0)
            rrf_scores[cid] += EMBEDDING_WEIGHT * _rrf_score(rank)

        for rank, chunk in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            rrf_scores.setdefault(cid, 0.0)
            rrf_scores[cid] += BM25_WEIGHT * _rrf_score(rank)

        # Build merged result entries
        chunk_map = {c["chunk_id"]: c for c in vec_results + bm25_results}
        for cid, rrf in rrf_scores.items():
            chunk = chunk_map[cid].copy()
            chunk["hybrid_score"] = rrf
            chunk["source_subfolder"] = str(sf)
            # Global unique key: subfolder + chunk_id
            ukey = f"{sf}::{cid}"
            if ukey not in all_results or all_results[ukey]["hybrid_score"] < rrf:
                all_results[ukey] = chunk

    # Sort by hybrid_score descending
    ranked = sorted(all_results.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return ranked[:top_k]
