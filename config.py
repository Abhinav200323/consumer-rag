"""
config.py — Central configuration loaded from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

print(f"🚀 CONFIG: Loading model '{GEMINI_MODEL}'")
if GEMINI_API_KEY:
    print(f"🔑 CONFIG: API Key loaded (ends in ...{GEMINI_API_KEY[-4:]})")
else:
    print("⚠️ CONFIG: No API Key found in .env!")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
KB_PATH: Path = BASE_DIR / os.getenv("KB_PATH", "knowledge_base")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512          # characters per chunk
CHUNK_OVERLAP: int = 64        # overlap between chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_WEIGHT: float = 0.4       # α: weight for BM25 score in hybrid search
EMBEDDING_WEIGHT: float = 0.6  # (1-α): weight for embedding score
TOP_K_RETRIEVAL: int = 20      # chunks fetched before re-ranking
TOP_K_RERANK: int = 6          # chunks passed to LLM after re-ranking

# ── Server ────────────────────────────────────────────────────────────────────
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "secret")
