# Prompt Layer & Agentic Logic Architecture

This document outlines the LLM prompt structure and the multi-stage agentic logic used in the Indian Consumer Law RAG system.

## 1. LLM Prompt Layer
The system uses Google Gemini for reasoning, classification, and metadata generation. Each stage is guided by a specific prompt template located in `llm/prompts.py`.

### A. Query Execution Prompts
| Stage | Prompt Template | Purpose |
| :--- | :--- | :--- |
| **Intent Classification** | `INTENT_CLASSIFICATION_PROMPT` | Classifies query as `greeting`, `legal`, or `general` to route it through the correct pipeline. |
| **Query Expansion** | `QUERY_EXPANSION_PROMPT` | Expands natural language queries into formal legal terminology and section numbers. |
| **Legal Reasoning** | `LEGAL_REASONING_PROMPT` | The core "Legal Buddy" prompt that analyzes retrieved context and provides friendly, cited advice. |

### B. Ingestion & Pre-processing Prompts
| Action | Prompt Template | Purpose |
| :--- | :--- | :--- |
| **Metadata Extraction** | `METADATA_EXTRACTION_PROMPT` | Extracts title, summary, acts, and jurisdiction from raw legal documents during ingestion. |
| **Image OCR** | [Hardcoded in `gemini_client.py`] | Transcribes text from images (PNG/JPG) using Gemini's multi-modal capabilities. |

---

## 2. Agentic "Tools" & Reasoning Steps
The system performs several non-LLM operations (the "Agentic Layer") to ensure accuracy and groundedness.

### Step 1: Smart Intent Routing
Filters "Hello" or "How are you?" queries to the chat pipeline, saving RAG compute for actual legal questions.

### Step 2: Metadata Filtering
Uses pre-extracted document metadata to narrow the search space to relevant categories (e.g., if the user asks about "EMI", it prioritizes Financial Services filters).

### Step 3: Hybrid Search (Vector + BM25)
- **FAISS**: Performs semantic retrieval using `multi-qa-mpnet-base-dot-v1`.
- **BM25**: Performs keyword retrieval to ensure exact section numbers or rare terms are caught.

### Step 4: Context Compression & Deduplication
Strips "Boilerplate" (Government of India headers, page numbers) and removes near-duplicate chunks from overlapping PDF pages.

### Step 5: Cross-Encoder Re-ranking
Applies a specialized re-ranking model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to precisely score the top 20 candidates for relevance to the specific user question.

### Step 6: Reference Traversal
A graph-like step that follows internal citations. If a retrieved chunk says "subject to Section 12", the agent automatically retrieves Section 12 even if it wasn't in the initial search results.

### Step 7: Citation Grounding (Post-Verification)
After Gemini generates an answer, a rule-based engine verifies every cited Section and Act against the original source chunks to flag potential hallucinations.

---

## 3. Configuration & State Management
- **System Instruction**: Every call is wrapped in a `SYSTEM_PROMPT` that enforces the "Friendly Buddy" persona.
- **Temperature**: Set to `0.1` for legal reasoning to ensure deterministic, conservative outputs.
- **Memory**: Multi-turn context summary is injected into the query before expansion for session-aware chat.
