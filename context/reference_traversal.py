"""
context/reference_traversal.py — Follow "See Section X" links within retrieved chunks

When a chunk references another section/clause, this module fetches those
linked sections from the structured index so the LLM has complete context.
"""

import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Match references like "Section 38", "Section 82(a)", "Clause (v)", "sub-section (2)"
_REF_PATTERNS = [
    re.compile(r"[Ss]ection\s+(\d+[A-Za-z]?(?:\([^)]+\))?)", re.IGNORECASE),
    re.compile(r"[Cc]lause\s+\(?([A-Za-z0-9]+)\)?", re.IGNORECASE),
    re.compile(r"[Ss]ub-?section\s+\(?(\d+)\)?", re.IGNORECASE),
    re.compile(r"Chapter\s+([IVXLCM]+|[A-Z]+|\d+)", re.IGNORECASE),
]


def extract_references(text: str) -> list[str]:
    """Extract all legal cross-references from a chunk of text."""
    refs = []
    for pattern in _REF_PATTERNS:
        matches = pattern.findall(text)
        refs.extend(matches)
    return list(dict.fromkeys(refs))  # deduplicate preserving order


def traverse_references(
    chunks: list[dict],
    subfolders: list[Path],
    max_depth: int = 2,
    max_additions: int = 4,
) -> list[dict]:
    """
    Given a list of retrieved chunks, find cross-referenced sections
    and fetch them from the structured index.

    Args:
        chunks: Already retrieved+reranked chunks
        subfolders: Subfolder paths to search for linked sections
        max_depth: How deep to follow reference chains
        max_additions: Max number of additional chunks to add

    Returns:
        Original chunks + additional referenced chunks (deduplicated)
    """
    from retrieval.structured_index import load_legal_tree

    existing_ids = {(c.get("source_subfolder", ""), c["chunk_id"]) for c in chunks}
    added: list[dict] = []

    def _fetch(sections: list[str], depth: int):
        if depth > max_depth or len(added) >= max_additions:
            return
        for sf in subfolders:
            tree = load_legal_tree(sf)
            if not tree:
                continue
            for sec in sections:
                ref_chunks = tree.get_chunks_for(section=sec)
                for rc in ref_chunks:
                    uid = (str(sf), rc["chunk_id"])
                    if uid not in existing_ids and len(added) < max_additions:
                        rc_copy = rc.copy()
                        rc_copy["source_subfolder"] = str(sf)
                        rc_copy["traversal_depth"] = depth
                        rc_copy["is_reference"] = True
                        added.append(rc_copy)
                        existing_ids.add(uid)
                        # Recurse for deep chains
                        nested_refs = extract_references(rc["text"])
                        if nested_refs:
                            _fetch(nested_refs, depth + 1)

    # Step 1: collect all references from original chunks
    all_refs: list[str] = []
    for chunk in chunks:
        all_refs.extend(extract_references(chunk["text"]))
    all_refs = list(dict.fromkeys(all_refs))

    # Step 2: fetch referenced chunks
    _fetch(all_refs, depth=1)

    if added:
        log.debug(f"Reference traversal added {len(added)} linked chunk(s)")

    return chunks + added
