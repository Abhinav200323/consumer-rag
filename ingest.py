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
from config import KB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, GEMINI_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Gemini Metadata Extraction ────────────────────────────────────────────────

def extract_document_metadata(full_text: str, filename: str) -> dict:
    """
    Use Gemini to extract structured, easy-to-understand metadata from a document.
    Sends the first ~4000 chars to keep costs low and speed high.
    Returns a dict with title, summary, document_type, primary_acts, etc.
    Falls back to regex-based extraction if Gemini fails.
    """
    if not GEMINI_API_KEY:
        log.warning("No GEMINI_API_KEY — skipping AI metadata extraction, using regex fallback.")
        return _regex_fallback_metadata(full_text, filename)

    sample_text = full_text[:4000]

    for attempt in range(2):  # retry once on failure
        try:
            from llm.gemini_client import generate_text_sync
            from llm.prompts import METADATA_EXTRACTION_PROMPT

            prompt = METADATA_EXTRACTION_PROMPT.format(filename=filename, text=sample_text)
            raw = generate_text_sync(prompt, max_tokens=1024)

            metadata = _parse_json_response(raw)
            if metadata:
                metadata["extraction_method"] = "gemini"
                log.info(f"    ✨ Gemini metadata: {metadata.get('title', filename)}")
                return metadata

            if attempt == 0:
                log.warning(f"    ⚠️ Gemini JSON parse failed for {filename}, retrying...")
                continue

        except Exception as e:
            log.warning(f"    ⚠️ Metadata extraction attempt {attempt+1} failed for {filename}: {e}")

    # Final fallback: regex-based extraction from raw text
    log.info(f"    📝 Using regex fallback for {filename}")
    return _regex_fallback_metadata(full_text, filename)


def _parse_json_response(raw: str) -> dict | None:
    """Try multiple strategies to extract valid JSON from Gemini's response."""
    cleaned = raw.strip()

    # Strategy 1: Strip markdown code fences
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        else:
            cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Strategy 2: Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find JSON object in the response using brace matching
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i+1])
                    except json.JSONDecodeError:
                        break

    return None


def _regex_fallback_metadata(full_text: str, filename: str) -> dict:
    """Extract basic metadata from raw text using regex when Gemini fails."""
    text = full_text[:6000]

    # Extract Acts
    acts = list(dict.fromkeys(_ACT_RE.findall(text)))
    # Extract Sections
    sections = [f"Section {s}" for s in dict.fromkeys(_SECTION_RE.findall(text))]

    # Guess document type from filename and content
    fname_lower = filename.lower()
    if "act" in fname_lower:
        doc_type = "Act"
    elif "rule" in fname_lower:
        doc_type = "Rules"
    elif "notification" in fname_lower or "gazette" in text.lower()[:500]:
        doc_type = "Notification"
    elif "judgment" in fname_lower or "court" in fname_lower:
        doc_type = "Judgment"
    else:
        doc_type = "Act" if acts else "Document"

    # Generate a basic summary from first ~300 chars
    first_para = text[:300].replace("\n", " ").strip()
    summary = f"Document '{filename}' covering {', '.join(acts[:2]) if acts else 'Indian consumer law'}."

    return {
        "title": filename.replace(".pdf", "").replace(".docx", "").replace("_", " ").title(),
        "summary": summary,
        "document_type": doc_type,
        "primary_acts": acts[:5],
        "key_sections": sections[:10],
        "jurisdiction": "All India",
        "key_takeaways": [f"Covers {len(sections)} sections across {len(acts)} act(s)"] if sections else [],
        "effective_date": None,
        "extraction_method": "regex_fallback",
    }

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

    # Parse + chunk all documents + extract Gemini metadata per doc
    all_chunks: list[dict] = []
    page_index: dict[str, list[int]] = {}  # doc → list of chunk indices per page
    doc_metadata: dict[str, dict] = {}     # filename → Gemini-extracted metadata

    for doc_path in supported:
        log.info(f"  Parsing: {doc_path.name}")
        pages = load_document(doc_path)
        doc_chunks_start = len(all_chunks)

        # Collect full text for metadata extraction
        full_text = "\n".join(p["text"] for p in pages)

        # ── Gemini metadata extraction ────────────────────────────────────
        log.info(f"    🤖 Extracting metadata via Gemini...")
        doc_meta = extract_document_metadata(full_text, doc_path.name)
        doc_metadata[doc_path.name] = doc_meta

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

    # ── 3. Chunk registry (with page index + doc metadata baked in) ───────────
    registry: dict[str, Any] = {
        "subdomain": subfolder.name,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
        "page_index": page_index,          # doc::page → [chunk_ids]
        "doc_metadata": doc_metadata,      # filename → Gemini-extracted metadata
        "embedding_model": EMBEDDING_MODEL,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved: index/chunks.json ({len(all_chunks)} chunks, {len(doc_metadata)} docs with metadata)")

    # ── 4. Update metadata.json — ALWAYS populate structural + Gemini fields ──
    # Derive domain/subdomain from folder structure (e.g. knowledge_base/consumer_protection/general_provisions)
    try:
        rel = subfolder.relative_to(KB_PATH)
        parts = list(rel.parts)
        if len(parts) >= 2:
            metadata["domain"] = parts[0]       # e.g. "consumer_protection"
            metadata["subdomain"] = parts[1]    # e.g. "general_provisions"
        elif len(parts) == 1:
            metadata["domain"] = parts[0]
            metadata["subdomain"] = parts[0]
    except ValueError:
        metadata.setdefault("domain", subfolder.parent.name)
        metadata.setdefault("subdomain", subfolder.name)

    metadata["indexed"] = True
    metadata["last_indexed"] = datetime.now(timezone.utc).isoformat()
    metadata["total_chunks"] = len(all_chunks)
    metadata["total_docs"] = len(supported)

    # Always populate from Gemini extraction (overwrite stale data)
    all_acts = []
    all_sections = []
    all_takeaways = []
    all_doc_types = []
    all_jurisdictions = []
    all_summaries = []

    for dm in doc_metadata.values():
        all_acts.extend(dm.get("primary_acts", []))
        all_sections.extend(dm.get("key_sections", []))
        all_takeaways.extend(dm.get("key_takeaways", []))
        if dm.get("document_type"):
            all_doc_types.append(dm["document_type"])
        if dm.get("jurisdiction"):
            all_jurisdictions.append(dm["jurisdiction"])
        if dm.get("summary"):
            all_summaries.append(dm["summary"])

    # Always overwrite these fields with fresh Gemini data (deduped)
    if all_acts:
        metadata["acts"] = list(dict.fromkeys(all_acts))
    if all_sections:
        metadata["sections"] = list(dict.fromkeys(all_sections))
    if all_takeaways:
        metadata["key_takeaways"] = all_takeaways
    if all_doc_types:
        metadata["doc_types"] = list(dict.fromkeys(all_doc_types))
    if all_jurisdictions:
        metadata["jurisdiction"] = list(dict.fromkeys(all_jurisdictions))
    if all_summaries:
        # Combine all document summaries into one description
        metadata["description"] = " | ".join(all_summaries)

    # Store per-document metadata for quick lookup
    metadata["documents"] = {
        fname: {
            "title": dm.get("title", fname),
            "summary": dm.get("summary", ""),
            "document_type": dm.get("document_type", "Unknown"),
            "key_takeaways": dm.get("key_takeaways", []),
        }
        for fname, dm in doc_metadata.items()
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"✓ Indexed {subfolder.name} ({len(supported)} docs, {len(all_chunks)} chunks)")
    log.info(f"  📋 Metadata: domain={metadata.get('domain')}, acts={metadata.get('acts', [])}, jurisdiction={metadata.get('jurisdiction', [])}")


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
