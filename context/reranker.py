"""
context/reranker.py — Cross-encoder re-ranking of retrieved chunks

Uses sentence-transformers cross-encoder to precisely score each
(query, chunk) pair. Much more accurate than first-stage retrieval
but slower — applied only to the top-K from hybrid search.
"""

import logging
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)
_reranker_cache = None


def get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        try:
            from sentence_transformers import CrossEncoder
            from config import RERANKER_MODEL
            log.info(f"Loading cross-encoder: {RERANKER_MODEL}")
            _reranker_cache = CrossEncoder(RERANKER_MODEL)
            log.info("Cross-encoder loaded.")
        except Exception as e:
            log.warning(f"Cross-encoder unavailable ({e}), using hybrid scores only.")
            _reranker_cache = None
    return _reranker_cache


def mmr_rerank(query: str, chunks: list[dict], top_k: int = 6, lambda_mult: float = 0.7) -> list[dict]:
    """
    Maximum Marginal Relevance (MMR) re-ranking.
    Combines Q-D relevance (from cross-encoder or hybrid) with D-D diversity.
    """
    if len(chunks) <= 1:
        return chunks

    from retrieval.hybrid_search import get_embedding_model
    from sklearn.metrics.pairwise import cosine_similarity

    # Extract relevance scores
    raw_scores = np.array([c["rerank_score"] for c in chunks])
    
    # Normalize relevance scores to [0, 1] range to be comparable with cosine similarity
    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max > score_min:
        relevance_scores = (raw_scores - score_min) / (score_max - score_min)
    else:
        relevance_scores = np.ones_like(raw_scores)
    
    # Calculate D-D similarity matrix using the embedding model
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    similarity_matrix = cosine_similarity(embeddings)

    selected_indices = []
    unselected_indices = list(range(len(chunks)))

    # Select the first chunk (max relevance)
    first_idx = np.argmax(relevance_scores)
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    # Iteratively select the rest
    while len(selected_indices) < top_k and unselected_indices:
        best_score = -np.inf
        best_idx = -1

        for idx in unselected_indices:
            # Relevance to Q
            rel_score = relevance_scores[idx]
            
            # Max similarity to already selected chunks
            sim_score = max([similarity_matrix[idx][s_idx] for s_idx in selected_indices])

            # MMR equation
            mmr_score = (lambda_mult * rel_score) - ((1 - lambda_mult) * sim_score)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    ranked_chunks = [chunks[i] for i in selected_indices]
    return ranked_chunks


def rerank(query: str, chunks: list[dict], top_k: int = 6, use_mmr: bool = True) -> list[dict]:
    """
    Re-rank chunks using cross-encoder scores.
    Optionally applies MMR to promote diversity among top results.
    """
    if not chunks:
        return []

    reranker = get_reranker()

    if reranker is None:
        # Fallback: use hybrid_score as base relevance
        for c in chunks:
            c["rerank_score"] = float(c.get("hybrid_score", 0))
    else:
        # Cross-encoder relevance
        pairs = [(query, c["text"]) for c in chunks]
        scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

    if use_mmr:
        # We process MMR on all retrieved chunks to pick top_k diverse
        reranked = mmr_rerank(query, chunks, top_k=top_k)
    else:
        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

    for rank, c in enumerate(reranked):
        c["rerank_rank"] = rank

    log.debug(f"Re-ranked {len(chunks)} → returning top {len(reranked)} (MMR={use_mmr})")
    return reranked
