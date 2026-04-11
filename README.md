# Agentic RAG for Consumer Law 🇮🇳⚖️

A state-of-the-art **Agentic RAG** (Retrieval-Augmented Generation) system for Indian consumer law. It features an advanced 3-layer architecture (Hybrid Search + MMR Diversity + Agentic Reasoning) with full **multimodal support** and **session-scoped privacy**.

## ✨ Key Advanced Features

### 💬 Multimodal Conversational Chat
*   **Interactive Chatbox**: A modern Streamlit-based chat interface that tracks conversation history.
*   **Multimedia Support**: Drag-and-drop **Images (PNG, JPG)**, PDFs, and DOCX files directly into the chat.
*   **Native Vision OCR**: Automatically extracts text from uploaded images using Gemini Vision models and indexes them instantly into your local context.
*   **Multimodal Reasoning**: Gemini analyzes both the retrieved legal text and the raw image bytes simultaneously for pixel-perfect accuracy.

### 🔐 Two-Tier Scoped Ingestion
*   **Temporary (Session-Scoped)**: Personal documents uploaded to the chat are isolated to your specific browser session. They are indexed in a secured temporary directory and cannot be accessed by other users.
*   **Permanent (Admin Level)**: System-wide legal archives can be updated via the "Upload" tab, secured by an **Admin Password** (configured in `.env`).

### ⚖️ Agentic Intelligence
*   **MMR (Max Marginal Relevance)**: Retrieval logic that penalizes redundancy, ensuring the AI sees diverse legal viewpoints rather than repetitive chunks.
*   **Critical Timelines & Penalties**: Gemini is instructed to automatically extract legal deadlines and fines into a structured **Step 0 table** for immediate clarity.
*   **Reference Traversal**: If a retrieved act says *"Per Section 34..."*, the agent automatically fetches the referenced section to build complete context.

### 📝 Document Drafting & Gamification
*   **Legal Authoring**: Automatically generate formal legal notices, district forum complaints, and RTI applications based on your case facts.
*   **Learning Games**: Interactive quizzes on Indian Consumer Law (e.g., "Is it a Violation?") to increase public awareness.

---

## ⚡ Setup & Installation

### 1. Set up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env`
Copy `.env.example` to `.env` and set your keys:
```env
GEMINI_API_KEY=your_key_here
ADMIN_PASSWORD=your_admin_pass
```

---

## 📥 Document Ingestion

The system retrieves context from the `knowledge_base` folder.

### Run the Initial Ingestion
```bash
python ingest.py --all
```
This builds both **FAISS** (semantic) and **BM25** (keyword) indexes for all legal subfolders. 

---

## 🚀 Running the System

Start the backend and frontend in separate terminals:

### Terminal 1: Backend (FastAPI)
```bash
uvicorn app:app --reload --port 8000
```

### Terminal 2: Frontend (Streamlit)
```bash
streamlit run frontend/streamlit_app.py
```

---

## 🏗️ Architecture Breakdown

```
User Query + Image Attachment
    │
    ▼
┌─────────────────────────────────────┐
│  RETRIEVAL LAYER                    │
│  ├── Query Expansion (rule + LLM)   │
│  ├── Metadata Filter (jurisdiction) │
│  ├── Hybrid Search (Vector + BM25)  │
│  └── MMR Selection (Redundancy check)│
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│  CONTEXT LAYER                      │
│  ├── Image OCR (Gemini Vision)      │
│  ├── Session Isolation (Session ID) │
│  └── Reference Traversal (recursive)│
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│  REASONING LAYER (Agentic)          │
│  ├── Step 0: Timelines & Penalties  │
│  ├── Multi-step Chain of Thought    │
│  ├── Multimodal Perception          │
│  └── Grounded Citation Verifier     │
└──────────────────┬──────────────────┘
                   │
    ▼
Conversational Answer + Interactive Tools
```

