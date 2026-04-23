"""
agent/planner.py — Multi-step agentic RAG orchestration

Orchestrates the full query pipeline with smart intent routing:
  Step 0: Intent classification (greeting / general / legal)
  Step 1: Query expansion (rule + LLM) — legal only
  Step 2: Metadata filtering — legal only
  Step 3: Hybrid search (vector + BM25) — legal only
  Step 4: Context compression + deduplication — legal only
  Step 5: Re-ranking (cross-encoder + MMR) — legal only
  Step 6: Reference traversal — legal only
  Step 7: LLM reasoning with chain-of-thought — legal only
  Step 8: Citation verification — legal only
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

from config import TOP_K_RETRIEVAL, TOP_K_RERANK, KB_PATH

log = logging.getLogger(__name__)


# ── Intent Classification ─────────────────────────────────────────────────────

async def classify_intent(query: str) -> str:
    """
    Classify user intent as 'greeting', 'legal', or 'general'.
    Uses a fast Gemini call for smart routing.
    Falls back to rule-based detection if Gemini fails.
    """
    # Quick rule-based check for obvious greetings (saves an API call)
    greetings = {
        "hi", "hello", "hey", "hola", "namaste", "greetings",
        "good morning", "good afternoon", "good evening",
        "who are you", "what can you do", "help", "buddy"
    }
    cleaned = re.sub(r'[^\w\s]', '', query.lower()).strip()
    words = cleaned.split()
    if len(words) <= 3 and any(g in cleaned for g in greetings):
        return "greeting"

    # Use Gemini for nuanced classification
    try:
        from llm.gemini_client import generate_text
        from llm.prompts import INTENT_CLASSIFICATION_PROMPT

        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        t0 = time.perf_counter()
        result = await generate_text(prompt, max_tokens=10)
        elapsed = time.perf_counter() - t0
        log.info(f"Intent classification: '{result.strip().lower()}' ({elapsed:.2f}s)")

        intent = result.strip().lower()
        if intent in ("greeting", "legal", "general"):
            return intent
        # If response doesn't match expected values, default to legal
        return "legal"
    except Exception as e:
        log.warning(f"Intent classification failed ({e}), defaulting to 'legal'")
        return "legal"


async def run_query(
    question: str,
    filters: dict | None = None,
    use_llm_expansion: bool = True,
    attached_image_b64: str | None = None,
    claim_value: float = 0,
    preferred_language: str = "English",
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Full agentic RAG pipeline with smart intent routing.
    Returns structured response with answer, intent type, and optional
    reasoning steps / citations (only for legal queries).
    """
    filters = filters or {}
    timing: dict[str, float] = {}
    trace: list[str] = []   # human-readable reasoning trace for UI

    # ── Step 0: Intent Classification ─────────────────────────────────────────
    t0 = time.perf_counter()
    intent = await classify_intent(question)
    timing["intent_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"🧠 **Intent detected**: `{intent}`")

    # ── Handle GREETING intent ────────────────────────────────────────────────
    if intent == "greeting":
        from llm.gemini_client import generate_text
        from retrieval.legal_logic import get_legal_check_context
        legal_context = get_legal_check_context(claim_value, preferred_language)

        t0 = time.perf_counter()
        raw_answer = await generate_text(f"GREETING: {question}\n\n{legal_context}", max_tokens=512)
        timing["llm_s"] = round(time.perf_counter() - t0, 3)

        # Store in memory
        if session_id:
            from agent.memory import add_message
            add_message(session_id, "user", question)
            add_message(session_id, "model", raw_answer)

        return {
            "answer": raw_answer,
            "intent": "greeting",
            "reasoning_steps": [],
            "verified_citations": [],
            "citation_confidence": 1.0,
            "unverified_sections": [],
            "source_chunks": [],
            "subfolders_searched": [],
            "query_expansion": None,
            "timing": timing,
            "trace": trace,
        }

    # ── Handle GENERAL intent (no RAG, just Gemini chat) ──────────────────────
    if intent == "general":
        from llm.gemini_client import generate_text, generate_with_history

        t0 = time.perf_counter()
        if session_id:
            from agent.memory import get_history, add_message
            history = get_history(session_id)
            # Build messages list for multi-turn
            messages = list(history) + [{"role": "user", "parts": [question]}]
            raw_answer = generate_with_history(messages, max_tokens=1024)
            add_message(session_id, "user", question)
            add_message(session_id, "model", raw_answer)
        else:
            raw_answer = await generate_text(question, max_tokens=1024)

        timing["llm_s"] = round(time.perf_counter() - t0, 3)

        return {
            "answer": raw_answer,
            "intent": "general",
            "reasoning_steps": [],
            "verified_citations": [],
            "citation_confidence": 1.0,
            "unverified_sections": [],
            "source_chunks": [],
            "subfolders_searched": [],
            "query_expansion": None,
            "timing": timing,
            "trace": trace,
        }

    # ── LEGAL intent — Full Agentic RAG Pipeline ──────────────────────────────

    # Enrich query with conversational memory if available
    enriched_question = question
    if session_id:
        from agent.memory import get_context_summary
        context_summary = get_context_summary(session_id)
        if context_summary:
            enriched_question = f"[Previous conversation context:\n{context_summary}]\n\nCurrent question: {question}"
            trace.append("💬 **Memory** — enriched query with conversation history")

    # ── Step 1: Query Expansion ───────────────────────────────────────────────
    t0 = time.perf_counter()
    from retrieval.query_expansion import expand_query
    expansion = await expand_query(enriched_question, use_llm=use_llm_expansion)
    search_query = expansion["final_expanded"]
    timing["query_expansion_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"🔍 **Query expanded** — added legal terms: `{expansion['rule_expanded'][len(enriched_question):].strip()[:120]}`")

    # ── Step 2: Metadata Filtering ────────────────────────────────────────────
    t0 = time.perf_counter()
    from retrieval.metadata_filter import filter_subfolders
    subfolders: list[Path] = filter_subfolders(filters, kb_path=KB_PATH, require_indexed=True)
    timing["metadata_filter_s"] = round(time.perf_counter() - t0, 3)

    if not subfolders:
        return {
            "answer": "No indexed documents were found matching the given filters. Please run the ingestion pipeline first (`python ingest.py --all`).",
            "intent": "legal",
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
            "intent": "legal",
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

    # ── Step 5: Re-ranking (with MMR diversity) ───────────────────────────────
    t0 = time.perf_counter()
    from context.reranker import rerank
    reranked = rerank(question, compressed, top_k=TOP_K_RERANK)
    timing["reranking_s"] = round(time.perf_counter() - t0, 3)
    trace.append(f"📊 **Re-ranked + MMR** — top {len(reranked)} diverse chunks selected")

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

    # Inject advanced legal logic context
    from retrieval.legal_logic import get_legal_check_context
    legal_context = get_legal_check_context(claim_value, preferred_language)
    context_str = "\n\n".join(context_parts) + "\n\n" + legal_context

    # ── Step 8: Gemini Reasoning (with memory if available) ───────────────────
    t0 = time.perf_counter()
    from llm.gemini_client import generate_text, generate_with_history
    from llm.prompts import LEGAL_REASONING_PROMPT
    prompt = LEGAL_REASONING_PROMPT.format(context=context_str, question=question)

    if session_id:
        from agent.memory import get_history, add_message
        history = get_history(session_id)
        messages = list(history) + [{"role": "user", "parts": [prompt]}]
        raw_answer = generate_with_history(messages, max_tokens=2048)
        add_message(session_id, "user", question)
        add_message(session_id, "model", raw_answer)
    else:
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
        "intent": "legal",
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
