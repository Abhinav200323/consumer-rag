"""
retrieval/structured_index.py — Act → Section → Clause hierarchical index

Builds and navigates a legal tree structure from the chunk registry.
Enables structured navigation: given a matched Section, jump to parent Act
or sibling clauses. Used for reference traversal.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


class LegalTree:
    """
    In-memory hierarchical index:
      act → section → clause → [chunk_ids]

    Built from chunks.json per subfolder.
    """

    def __init__(self):
        # act_name → {section_id → {clause_id → [chunk_ids]}}
        self.tree: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # chunk_id → chunk dict (for lookup)
        self.chunks: list[dict] = []
        # page index: "doc::pN" → [chunk_ids]
        self.page_index: dict[str, list[int]] = {}

    def build_from_registry(self, registry: dict):
        self.chunks = registry["chunks"]
        self.page_index = registry.get("page_index", {})

        for chunk in self.chunks:
            act = chunk.get("act") or "Unknown Act"
            section = chunk.get("section") or "Unknown Section"
            clause = chunk.get("clause") or "root"
            cid = chunk["chunk_id"]
            self.tree[act][section][clause].append(cid)

    def get_act_names(self) -> list[str]:
        return list(self.tree.keys())

    def get_sections(self, act: str) -> list[str]:
        return list(self.tree.get(act, {}).keys())

    def get_clauses(self, act: str, section: str) -> list[str]:
        return list(self.tree.get(act, {}).get(section, {}).keys())

    def get_chunks_for(
        self, act: str | None = None, section: str | None = None, clause: str | None = None
    ) -> list[dict]:
        """
        Retrieve chunks filtered by act/section/clause hierarchy.
        Any None parameter acts as a wildcard.
        """
        results = []
        for a_name, sections in self.tree.items():
            if act and act.lower() not in a_name.lower():
                continue
            for s_id, clauses in sections.items():
                if section and section != s_id:
                    continue
                for c_id, chunk_ids in clauses.items():
                    if clause and clause != c_id:
                        continue
                    results.extend(self.chunks[cid] for cid in chunk_ids if cid < len(self.chunks))
        return results

    def get_page_chunks(self, doc: str, page: int) -> list[dict]:
        """Direct page-indexed lookup — fetch all chunks from a specific page."""
        key = f"{doc}::p{page}"
        ids = self.page_index.get(key, [])
        return [self.chunks[i] for i in ids if i < len(self.chunks)]


# Global tree registry: subfolder_path → LegalTree
_tree_cache: dict[str, LegalTree] = {}


def load_legal_tree(subfolder: Path) -> LegalTree | None:
    """Load (and cache) the LegalTree for a subfolder."""
    key = str(subfolder)
    if key in _tree_cache:
        return _tree_cache[key]

    chunks_path = subfolder / "index" / "chunks.json"
    if not chunks_path.exists():
        return None

    try:
        with open(chunks_path, encoding="utf-8") as f:
            registry = json.load(f)
        tree = LegalTree()
        tree.build_from_registry(registry)
        _tree_cache[key] = tree
        return tree
    except Exception as e:
        log.error(f"Failed to load legal tree for {subfolder}: {e}")
        return None


def build_global_tree(subfolders: list[Path]) -> LegalTree:
    """
    Merge trees from multiple subfolders into one global tree.
    Chunk IDs become globally scoped by appending subfolder prefix.
    """
    merged = LegalTree()
    offset = 0
    for sf in subfolders:
        tree = load_legal_tree(sf)
        if not tree:
            continue
        # Merge by rebuilding with offset IDs
        for chunk in tree.chunks:
            new_chunk = chunk.copy()
            new_chunk["chunk_id"] = chunk["chunk_id"] + offset
            new_chunk["source_subfolder"] = str(sf)
            merged.chunks.append(new_chunk)

        for act, sections in tree.tree.items():
            for sec, clauses in sections.items():
                for cl, ids in clauses.items():
                    merged.tree[act][sec][cl].extend(i + offset for i in ids)

        for page_key, ids in tree.page_index.items():
            merged.page_index[page_key] = [i + offset for i in ids]

        offset += len(tree.chunks)

    return merged
