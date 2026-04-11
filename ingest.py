"""
ingest.py — Document ingestion pipeline for Agentic RAG

Processes documents (PDF, DOCX, TXT) from a knowledge_base subfolder, builds:
  1. Dense vector index (FAISS) — sentence-transformer embeddings, page-indexed
  2. Sparse BM25 index — vectorless keyword-based retrieval
  3. chunks.json — full chunk registry with Act/Section/Clause/page metadata

Usage:
    python ingest.py --path knowledge_base/consumer_protection/general_provisions
    python ingest.py --all   # re-indexes all subfolders
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import KB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Document Parsers ──────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> list[dict]:
    """Return list of {text, page} dicts from a PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i})
        return pages
    except Exception as e:
        log.error(f"PDF parse error {path}: {e}")
        return []


def parse_docx(path: Path) -> list[dict]:
    """Return list of {text, page} dicts from DOCX (page approx by paragraph blocks)."""
    try:
        from docx import Document
        doc = Document(str(path))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        # Approximate page: every ~3000 chars ≈ 1 page
        chunks = []
        block_size = 3000
        for i, start in enumerate(range(0, len(full_text), block_size), start=1):
            text = full_text[start:start + block_size]
            if text.strip():
                chunks.append({"text": text, "page": i})
        return chunks
    except Exception as e:
        log.error(f"DOCX parse error {path}: {e}")
        return []


def parse_txt(path: Path) -> list[dict]:
    """Return list of {text, page} dicts from plain text (approx page by line count)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        pages = []
        lines_per_page = 50
        for i in range(0, len(lines), lines_per_page):
            chunk_lines = lines[i:i + lines_per_page]
            text = "\n".join(chunk_lines).strip()
            if text:
                page_num = (i // lines_per_page) + 1
                pages.append({"text": text, "page": page_num})
        return pages
    except Exception as e:
        log.error(f"TXT parse error {path}: {e}")
        return []


def parse_image(path: Path) -> list[dict]:
    """Return text extracted from an image via Gemini Vision."""
    try:
        from llm.gemini_client import extract_image_text_sync
        text = extract_image_text_sync(path)
        if text.strip():
            return [{"text": text, "page": 1}]
        return []
    except Exception as e:
        log.error(f"Image parse error {path}: {e}")
        return []


def load_document(path: Path) -> list[dict]:
    """Auto-detect format and return list of {text, page} dicts."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix == ".docx":
        return parse_docx(path)
    elif suffix in (".txt", ".md"):
        return parse_txt(path)
    elif suffix in (".png", ".jpg", ".jpeg"):
        return parse_image(path)
    else:
        log.warning(f"Unsupported file type: {path}")
        return []


# ── Legal Structure Extractor ─────────────────────────────────────────────────

_ACT_RE = re.compile(r"(Consumer Protection Act|Food Safety.*Act|TRAI|FSSAI|[A-Z][A-Za-z\s]+Act,?\s*\d{4})")
_SECTION_RE = re.compile(r"Section\s+(\d+[A-Za-z]*)", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"(?:Clause|Sub-section|sub-section)\s+\(?([a-zA-Z0-9]+)\)?", re.IGNORECASE)


def extract_legal_tags(text: str) -> dict:
    """Extract Act/Section/Clause references from chunk text."""
    acts = _ACT_RE.findall(text)
    sections = _SECTION_RE.findall(text)
    clauses = _CLAUSE_RE.findall(text)
    return {
        "act": acts[0] if acts else None,
        "section": sections[0] if sections else None,
        "clause": clauses[0] if clauses else None,
        "all_sections": list(dict.fromkeys(sections)),
        "all_clauses": list(dict.fromkeys(clauses)),
    }


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_page(page_text: str, page_num: int, doc_name: str,
               chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split page text into overlapping chunks.
    Each chunk carries: text, page, doc, char_start, char_end, legal tags.
    """
    text = page_text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            tags = extract_legal_tags(chunk_text)
            chunks.append({
                "text": chunk_text,
                "doc": doc_name,
                "page": page_num,
                "char_start": start,
                "char_end": end,
                **tags,
            })
        start = end - overlap if end < len(text) else end

    return chunks


# ── Index Builders ────────────────────────────────────────────────────────────

def build_vector_index(chunks: list[dict], model: SentenceTransformer) -> tuple:
    """
    Build FAISS dense vector index from chunks.
    Returns (faiss_index, embeddings_array) — page info is in chunks metadata.
    """
    log.info(f"  Encoding {len(chunks)} chunks with sentence-transformer...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    # Inner product index (works with normalized vectors as cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build BM25 sparse (vectorless) index from chunk texts.
    Tokenizes by lowercased words — no neural embeddings required.
    """
    log.info(f"  Building BM25 sparse index for {len(chunks)} chunks...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    log.info("  BM25 index built.")
    return bm25


# ── Main Indexer ──────────────────────────────────────────────────────────────

def index_subfolder(subfolder: Path, model: SentenceTransformer):
    """Full ingestion pipeline for a single knowledge_base subfolder."""
    docs_dir = subfolder / "docs"
    index_dir = subfolder / "index"
    metadata_path = subfolder / "metadata.json"

    if not docs_dir.exists():
        log.warning(f"No docs/ directory in {subfolder}, skipping.")
        return

    index_dir.mkdir(exist_ok=True)

    # Load existing metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Gather documents
    doc_files = list(docs_dir.glob("*"))
    supported = [p for p in doc_files if p.suffix.lower() in (".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg")]
    if not supported:
        log.warning(f"No supported documents found in {docs_dir}")
        return

    log.info(f"\n{'='*60}")
    log.info(f"Indexing: {subfolder}")
    log.info(f"Found {len(supported)} document(s)")

    # Parse + chunk all documents
    all_chunks: list[dict] = []
    page_index: dict[str, list[int]] = {}  # doc → list of chunk indices per page

    for doc_path in supported:
        log.info(f"  Parsing: {doc_path.name}")
        pages = load_document(doc_path)
        doc_chunks_start = len(all_chunks)

        for page_data in pages:
            page_chunks = chunk_page(
                page_data["text"],
                page_data["page"],
                doc_path.name,
            )
            for chunk in page_chunks:
                chunk["chunk_id"] = len(all_chunks)
                # Page index: doc → page → [chunk_ids]
                page_key = f"{doc_path.name}::p{chunk['page']}"
                page_index.setdefault(page_key, []).append(chunk["chunk_id"])
                all_chunks.append(chunk)

        doc_chunks_end = len(all_chunks)
        log.info(f"    → {doc_chunks_end - doc_chunks_start} chunks from {len(pages)} pages")

    if not all_chunks:
        log.warning("No chunks generated, skipping indexing.")
        return

    # ── 1. Vector (dense) index via FAISS ────────────────────────────────────
    faiss_index, _ = build_vector_index(all_chunks, model)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    log.info(f"  Saved: index/faiss.index")

    # ── 2. BM25 (sparse/vectorless) index ─────────────────────────────────────
    bm25 = build_bm25_index(all_chunks)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    log.info(f"  Saved: index/bm25.pkl")

    # ── 3. Chunk registry (with page index baked in) ──────────────────────────
    registry: dict[str, Any] = {
        "subdomain": subfolder.name,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
        "page_index": page_index,          # doc::page → [chunk_ids]
        "embedding_model": EMBEDDING_MODEL,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved: index/chunks.json ({len(all_chunks)} chunks)")

    # Update metadata
    metadata["indexed"] = True
    metadata["last_indexed"] = datetime.now(timezone.utc).isoformat()
    metadata["total_chunks"] = len(all_chunks)
    metadata["total_docs"] = len(supported)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"✓ Indexed {subfolder.name} ({len(supported)} docs, {len(all_chunks)} chunks)")


def find_all_subfolders(kb_path: Path) -> list[Path]:
    """Return all subdomain-level subfolders (those containing a docs/ dir)."""
    subfolders = []
    for p in sorted(kb_path.rglob("docs")):
        subfolders.append(p.parent)
    return subfolders


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Agentic RAG knowledge base.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", type=str, help="Path to a specific subfolder to index")
    group.add_argument("--all", action="store_true", help="Index all subfolders in the knowledge base")
    args = parser.parse_args()

    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    if args.all:
        subfolders = find_all_subfolders(KB_PATH)
        if not subfolders:
            log.error(f"No subfolders found in {KB_PATH}")
            sys.exit(1)
        log.info(f"Found {len(subfolders)} subfolder(s) to index")
        for sf in subfolders:
            index_subfolder(sf, model)
    else:
        sf = Path(args.path).resolve()
        if not sf.exists():
            log.error(f"Path does not exist: {sf}")
            sys.exit(1)
        index_subfolder(sf, model)

    log.info("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
