"""
context/compressor.py — Remove noise and compress retrieved chunks

Two-stage compression:
  1. Rule-based denoiser — strips legal boilerplate, repeated header/footer patterns
  2. Deduplication — removes near-duplicate chunks before sending to LLM
"""

import re
import logging
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

# Legal boilerplate patterns to strip from chunks
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*GOVERNMENT OF INDIA.*?\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Gazette of India.*?\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Ministry of.*?\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+.*?\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE),  # page n/m
    re.compile(r"\n{3,}", re.MULTILINE),  # 3+ blank lines → 2
]


def clean_text(text: str) -> str:
    """Remove boilerplate and normalise whitespace."""
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.pattern == r"\n{3,}":
            text = pattern.sub("\n\n", text)
        else:
            text = pattern.sub("", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    """Quick ratio for near-duplicate detection."""
    return SequenceMatcher(None, a[:300], b[:300]).ratio()


def deduplicate(chunks: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate chunks (same passage repeated from overlapping pages)."""
    seen: list[dict] = []
    for chunk in chunks:
        is_dup = any(similarity(chunk["text"], s["text"]) > threshold for s in seen)
        if not is_dup:
            seen.append(chunk)
    return seen


def compress(chunks: list[dict]) -> list[dict]:
    """
    Full compression pipeline:
      1. Clean boilerplate from each chunk
      2. Deduplicate near-identical passages
    Returns cleaned, deduplicated chunks ready for re-ranking.
    """
    cleaned = []
    for chunk in chunks:
        text = clean_text(chunk["text"])
        if len(text) < 40:  # skip empty-after-clean chunks
            continue
        updated = chunk.copy()
        updated["text"] = text
        cleaned.append(updated)

    deduped = deduplicate(cleaned)
    log.debug(f"Compressor: {len(chunks)} → {len(deduped)} chunks after cleaning+dedup")
    return deduped
