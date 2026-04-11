"""
agent/citation.py — Citation extraction and verification

Parses citations from LLM output and verifies each against 
the chunks that were actually retrieved. Flags or removes
any unverifiable citations (hallucinated legal references).
"""

import re
import logging

log = logging.getLogger(__name__)

# Match citation patterns in LLM output
_CITATION_PATTERNS = [
    re.compile(r"Section\s+(\d+[A-Za-z]?(?:\([^)]+\))?)", re.IGNORECASE),
    re.compile(r"(Consumer Protection Act,?\s*\d{4})", re.IGNORECASE),
    re.compile(r"(Food Safety.*?Act,?\s*\d{4})", re.IGNORECASE),
    re.compile(r"(TRAI.*?Regulations?,?\s*\d{4})", re.IGNORECASE),
    re.compile(r"Chapter\s+([IVXLCM]+|[A-Z]+|\d+)", re.IGNORECASE),
    re.compile(r"Page\s+(\d+)", re.IGNORECASE),
]


def extract_citations_from_text(text: str) -> list[str]:
    """Extract all legal citations mentioned in the LLM response."""
    citations = []
    for pattern in _CITATION_PATTERNS:
        matches = pattern.findall(text)
        citations.extend(matches)
    return list(dict.fromkeys(citations))


def build_verified_citations(chunks: list[dict]) -> list[dict]:
    """
    Build a structured citation list from the source chunks used in generation.
    Each citation is guaranteed to be grounded in the retrieved context.
    """
    seen = set()
    citations = []
    for chunk in chunks:
        section = chunk.get("section")
        act = chunk.get("act")
        doc = chunk.get("doc", "Unknown Document")
        page = chunk.get("page", "?")
        subfolder = chunk.get("source_subfolder", "")

        key = f"{act}::{section}::{doc}::{page}"
        if key not in seen:
            seen.add(key)
            citations.append({
                "act": act,
                "section": section,
                "doc": doc,
                "page": page,
                "subfolder": subfolder,
                "text_snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "is_reference": chunk.get("is_reference", False),
            })
    return citations


def verify_and_annotate(answer: str, source_chunks: list[dict]) -> dict:
    """
    Check citations in LLM answer against source chunks.
    Returns verified citations list + annotated answer.
    """
    # Extract what the LLM claimed to cite
    claimed_sections = set()
    for pattern in _CITATION_PATTERNS[:1]:  # Section pattern
        claimed_sections.update(pattern.findall(answer))

    # What sections are actually in the source chunks
    sourced_sections = {c.get("section") for c in source_chunks if c.get("section")}
    sourced_acts = {c.get("act") for c in source_chunks if c.get("act")}

    unverified = claimed_sections - sourced_sections
    if unverified:
        log.warning(f"Potentially unverified citations in answer: {unverified}")

    verified_citations = build_verified_citations(source_chunks)

    return {
        "answer": answer,
        "verified_citations": verified_citations,
        "claimed_sections": list(claimed_sections),
        "sourced_sections": list(sourced_sections),
        "sourced_acts": list(sourced_acts),
        "unverified_sections": list(unverified),
        "citation_confidence": 1.0 - (len(unverified) / max(len(claimed_sections), 1)),
    }
