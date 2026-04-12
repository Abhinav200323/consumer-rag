"""
app.py — FastAPI backend for the Agentic RAG for Consumer Law

Endpoints:
  POST /query          — Run the full agentic RAG pipeline
  POST /ingest         — Trigger ingestion for a subfolder
  GET  /knowledge-base — List all subfolders with metadata
  GET  /health         — Health check
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import KB_PATH, BACKEND_PORT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Consumer Law Agentic RAG API",
    description="Agentic RAG system for Indian Consumer Law — hybrid vector + BM25 search with Gemini reasoning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class QueryFilters(BaseModel):
    jurisdiction: Optional[str] = None
    doc_type: Optional[str] = None
    domain: Optional[str] = None
    subdomain: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    query_domains: Optional[list[str]] = None
    session_id: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    filters: QueryFilters = QueryFilters()
    use_llm_expansion: bool = True
    attached_image_b64: Optional[str] = None
    claim_value: float = 0
    preferred_language: str = "English"


class IngestRequest(BaseModel):
    subfolder_path: str   # Relative to KB_PATH, e.g. "consumer_protection/general_provisions"


class DraftRequest(BaseModel):
    document_type: str
    facts: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "kb_path": str(KB_PATH)}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Run the full 9-step agentic RAG pipeline:
    expand query → filter subfolders → hybrid search → compress →
    rerank → traverse references → Gemini reasoning → verify citations
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    from agent.planner import run_query
    try:
        result = await run_query(
            question=request.query,
            filters=request.filters.model_dump(exclude_none=True),
            use_llm_expansion=request.use_llm_expansion,
            attached_image_b64=request.attached_image_b64,
            claim_value=request.claim_value,
            preferred_language=request.preferred_language
        )
        return result
    except Exception as e:
        log.exception("Query pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Trigger the ingestion pipeline for a subfolder.
    Builds FAISS vector index + BM25 vectorless index + chunk registry.
    """
    subfolder = KB_PATH / request.subfolder_path
    if not subfolder.exists():
        raise HTTPException(status_code=404, detail=f"Subfolder not found: {subfolder}")

    try:
        # Run ingest.py as a subprocess so it doesn't block the event loop
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "ingest.py", "--path", str(subfolder),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed:\n{stderr.decode()}"
            )
        return {
            "status": "success",
            "subfolder": str(subfolder),
            "output": stdout.decode()[-1000:],  # last 1000 chars of output
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    subfolder_path: str = Form(...),
    admin_password: Optional[str] = Form(None),
):
    """
    Upload a document (PDF/DOCX/TXT) to a knowledge base subfolder
    and trigger re-indexing.
    """
    if not subfolder_path.startswith("temp/"):
        from config import ADMIN_PASSWORD
        if admin_password != ADMIN_PASSWORD:
            raise HTTPException(status_code=403, detail="Invalid admin password for permanent storage.")

    target_dir = KB_PATH / subfolder_path / "docs"
    target_dir.mkdir(parents=True, exist_ok=True)

    dest = target_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
    log.info(f"Uploaded {file.filename} to {dest}")

    # Trigger ingestion
    subfolder = KB_PATH / subfolder_path
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "ingest.py", "--path", str(subfolder),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Ingestion failed:\n{stderr.decode()}")

    return {
        "status": "success",
        "filename": file.filename,
        "subfolder": subfolder_path,
        "output": stdout.decode()[-500:],
    }


@app.post("/draft_document")
async def draft_document(request: DraftRequest):
    """
    Generate a legal document draft using Gemini.
    """
    from llm.gemini_client import generate_text
    from llm.prompts import DOCUMENT_DRAFTING_PROMPT
    
    if not request.facts.strip() or not request.document_type.strip():
        raise HTTPException(status_code=400, detail="Document type and facts cannot be empty")
        
    try:
        prompt = DOCUMENT_DRAFTING_PROMPT.format(
            document_type=request.document_type,
            facts=request.facts
        )
        draft = await generate_text(prompt, max_tokens=2048)
        return {"status": "success", "draft": draft}
    except Exception as e:
        log.exception("Draft generation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-base")
async def list_knowledge_base():
    """List all subfolders in the knowledge base with their metadata."""
    from retrieval.metadata_filter import list_all_subfolders_metadata
    entries = list_all_subfolders_metadata(KB_PATH)
    return {"entries": entries, "total": len(entries)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=BACKEND_PORT, reload=True)
