"""
retrieval/metadata_filter.py — Filter knowledge base subfolders by metadata criteria

Reads all metadata.json files across the knowledge base tree and returns
paths to subfolders that match the requested filters.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from config import KB_PATH

log = logging.getLogger(__name__)


def load_all_metadata(kb_path: Path = KB_PATH) -> list[dict]:
    """Walk the knowledge base and load all subdomain-level metadata.json files."""
    entries = []
    for meta_file in sorted(kb_path.rglob("metadata.json")):
        # Only subdomain-level metadata (those that have a docs/ sibling dir)
        if (meta_file.parent / "docs").exists():
            try:
                with open(meta_file, encoding="utf-8") as f:
                    data = json.load(f)
                data["_path"] = str(meta_file.parent)
                entries.append(data)
            except Exception as e:
                log.warning(f"Could not load metadata at {meta_file}: {e}")
    return entries


def filter_subfolders(
    filters: dict,
    kb_path: Path = KB_PATH,
    require_indexed: bool = True,
) -> list[Path]:
    """
    Return list of subfolder paths matching the given filters.

    Filter keys (all optional):
      - jurisdiction: str  e.g. "India", "State"
      - doc_type: str  e.g. "Act", "Regulations"
      - domain: str  e.g. "consumer_protection"
      - subdomain: str
      - date_from: str  ISO date, filter docs with date_range.to >= date_from
      - date_to: str  ISO date, filter docs with date_range.from <= date_to
      - query_domains: list[str]  whitelist of domains to search

    If no filters provided or all entries pass, all indexed subfolders are returned.
    """
    all_meta = load_all_metadata(kb_path)
    matched: list[Path] = []

    for meta in all_meta:
        if require_indexed and not meta.get("indexed", False):
            continue

        # Session isolation for temporary documents
        rel_path = str(Path(meta["_path"]).relative_to(kb_path)).replace("\\", "/")
        if rel_path.startswith("temp/"):
            folder_session = Path(meta["_path"]).name
            if filters.get("session_id") != folder_session:
                continue

        # Jurisdiction filter
        if "jurisdiction" in filters and filters["jurisdiction"]:
            meta_jurisdictions = [j.lower() for j in meta.get("jurisdiction", [])]
            if filters["jurisdiction"].lower() not in meta_jurisdictions:
                continue

        # Document type filter
        if "doc_type" in filters and filters["doc_type"]:
            meta_types = [t.lower() for t in meta.get("doc_types", [])]
            if filters["doc_type"].lower() not in meta_types:
                continue

        # Domain filter
        if "domain" in filters and filters["domain"]:
            if meta.get("domain", "").lower() != filters["domain"].lower():
                continue

        # Subdomain filter
        if "subdomain" in filters and filters["subdomain"]:
            if meta.get("subdomain", "").lower() != filters["subdomain"].lower():
                continue

        # Date range filters
        date_range = meta.get("date_range", {})
        meta_from = date_range.get("from")
        meta_to = date_range.get("to")

        if "date_from" in filters and filters["date_from"] and meta_to:
            if meta_to < filters["date_from"]:
                continue

        if "date_to" in filters and filters["date_to"] and meta_from:
            if meta_from > filters["date_to"]:
                continue

        # Domain whitelist
        if "query_domains" in filters and filters["query_domains"]:
            if meta.get("domain") not in filters["query_domains"]:
                continue

        matched.append(Path(meta["_path"]))

    # If nothing matched filters (and filters were provided), fall back to all indexed
    if not matched and any(v for k, v in filters.items() if k != "session_id" and v):
        log.warning("No subfolders matched filters — falling back to all indexed subfolders")
        # Ensure we still respect session boundaries on fallback
        for m in all_meta:
            if not m.get("indexed", False): continue
            rp = str(Path(m["_path"]).relative_to(kb_path)).replace("\\", "/")
            if rp.startswith("temp/"):
                if filters.get("session_id") != Path(m["_path"]).name:
                    continue
            matched.append(Path(m["_path"]))

    return matched


def list_all_subfolders_metadata(kb_path: Path = KB_PATH) -> list[dict]:
    """Return metadata for all subfolders (indexed or not) for the UI browser."""
    return load_all_metadata(kb_path)
