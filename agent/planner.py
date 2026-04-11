"""
agent/planner.py — Multi-step agentic RAG orchestration

Orchestrates the full query pipeline:
  Step 1: Query expansion (rule + LLM)
  Step 2: Metadata filtering (select relevant subfolders)
  Step 3: Hybrid search (vector + BM25) across subfolders
  Step 4: Context compression + deduplication
  Step 5: Re-ranking (cross-encoder)
  Step 6: Reference traversal (linked sections)
  Step 7: LLM reasoning with chain-of-thought
  Step 8: Citation verification
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from config import TOP_K_RETRIEVAL, TOP_K_RERANK, KB_PATH

log = logging.getLogger(__name__)


async def run_query(
    question: str,
    filters: dict | None = None,
    use_llm_expansion: bool = True,
    attached_image_b64: str | None = None,
) -> dict[str, Any]:
    """
    Full agentic RAG pipeline. Returns structured response with answer,
    reasoning steps, verified citations, source chunks, and timing.
    """
    filters = filters or {}
    timing: dict[str, float] = {}
    trace: list[str] = []   # human-readable reasoning trace for UI

    # ── Step 1: Query Expansion ───────────────────────────────────────────────
    t0 = time.perf_counter()
    from retrieval.query_expansion import expand_query
    expansion = await expand_query(question, use_llm=use_llm_expansion)
    search_query = expansion["final_expanded"]
    timing["query_expansion_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"🔍 **Query expanded** — added legal terms: `{expansion['rule_expanded'][len(question):].strip()[:120]}`")

    # ── Step 2: Metadata Filtering ────────────────────────────────────────────
    t0 = time.perf_counter()
    from retrieval.metadata_filter import filter_subfolders
    subfolders: list[Path] = filter_subfolders(filters, kb_path=KB_PATH, require_indexed=True)
    timing["metadata_filter_s"] = round(time.perf_counter() - t0, 3)

    if not subfolders:
        return {
            "answer": "No indexed documents were found matching the given filters. Please run the ingestion pipeline first (`python ingest.py --all`).",
            "reasoning_steps": [],
            "verified_citations": [],
            "source_chunks": [],
            "subfolders_searched": [],
            "timing": timing,
            "trace": trace,
        }

    trace.append(f"📁 **Subfolders selected**: {[str(sf.name) for sf in subfolders]}")

    # ── Step 3: Hybrid Retrieval (vector + BM25) ──────────────────────────────
    t0 = time.perf_counter()
    from retrieval.hybrid_search import hybrid_search
    raw_chunks = hybrid_search(search_query, subfolders, top_k=TOP_K_RETRIEVAL)
    timing["hybrid_search_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"⚡ **Hybrid search** — retrieved {len(raw_chunks)} chunks (vector + BM25)")

    if not raw_chunks:
        return {
            "answer": "No relevant content found for your query in the loaded knowledge base. Try rephrasing or broadening your question.",
            "reasoning_steps": [],
            "verified_citations": [],
            "source_chunks": [],
            "subfolders_searched": [str(sf) for sf in subfolders],
            "timing": timing,
            "trace": trace,
        }

    # ── Step 4: Context Compression ───────────────────────────────────────────
    t0 = time.perf_counter()
    from context.compressor import compress
    compressed = compress(raw_chunks)
    timing["compression_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"🗜️ **Compression** — {len(raw_chunks)} → {len(compressed)} chunks (deduped + cleaned)")

    # ── Step 5: Re-ranking ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    from context.reranker import rerank
    reranked = rerank(question, compressed, top_k=TOP_K_RERANK)
    timing["reranking_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"📊 **Re-ranked** — top {len(reranked)} chunks selected for reasoning")

    # ── Step 6: Reference Traversal ───────────────────────────────────────────
    t0 = time.perf_counter()
    from context.reference_traversal import traverse_references
    final_chunks = traverse_references(reranked, subfolders)
    timing["traversal_s"] = round(time.perf_counter() - t0, 3)
    added = len(final_chunks) - len(reranked)
    if added:
        trace.append(f"🔗 **Reference traversal** — followed {added} linked section(s)")

    # ── Step 7: Build Context for LLM ────────────────────────────────────────
    context_parts = []
    for i, chunk in enumerate(final_chunks, start=1):
        doc_info = f"[{chunk.get('doc', 'Unknown')} | Page {chunk.get('page', '?')}]"
        section_info = ""
        if chunk.get("act"):
            section_info += f" [{chunk['act']}"
            if chunk.get("section"):
                section_info += f", Section {chunk['section']}"
            section_info += "]"
        ref_tag = " [Referenced]" if chunk.get("is_reference") else ""
        context_parts.append(
            f"--- Chunk {i}{section_info}{ref_tag} {doc_info} ---\n{chunk['text']}"
        )
    context_str = "\n\n".join(context_parts)

    # ── Step 8: Gemini Reasoning ──────────────────────────────────────────────
    t0 = time.perf_counter()
    from llm.gemini_client import generate_text
    from llm.prompts import LEGAL_REASONING_PROMPT
    prompt = LEGAL_REASONING_PROMPT.format(context=context_str, question=question)
    raw_answer = await generate_text(prompt, max_tokens=2048, attached_image_b64=attached_image_b64)
    timing["llm_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"🤖 **Gemini reasoning** complete ({timing['llm_s']}s)")

    # ── Step 9: Citation Verification ────────────────────────────────────────
    from agent.citation import verify_and_annotate
    result = verify_and_annotate(raw_answer, final_chunks)

    # Parse reasoning steps from the structured LLM output
    reasoning_steps = _parse_reasoning_steps(raw_answer)

    return {
        "answer": raw_answer,
        "reasoning_steps": reasoning_steps,
        "verified_citations": result["verified_citations"],
        "citation_confidence": result["citation_confidence"],
        "unverified_sections": result["unverified_sections"],
        "source_chunks": [
            {
                "text": c["text"][:300],
                "doc": c.get("doc"),
                "page": c.get("page"),
                "section": c.get("section"),
                "act": c.get("act"),
                "hybrid_score": c.get("hybrid_score"),
                "rerank_score": c.get("rerank_score"),
                "is_reference": c.get("is_reference", False),
                "source_subfolder": c.get("source_subfolder"),
            }
            for c in final_chunks
        ],
        "subfolders_searched": [str(sf) for sf in subfolders],
        "query_expansion": expansion,
        "timing": timing,
        "trace": trace,
    }


def _parse_reasoning_steps(answer: str) -> list[dict]:
    """Extract structured reasoning steps from LLM output."""
    import re
    steps = []
    pattern = re.compile(
        r"\*\*STEP\s+(\d+)\s+[—–-]\s*(.+?)\*\*\s*\n(.*?)(?=\*\*STEP|\*\*CITATIONS|$)",
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(answer):
        steps.append({
            "step": int(match.group(1)),
            "title": match.group(2).strip(),
            "content": match.group(3).strip(),
        })
    return steps
