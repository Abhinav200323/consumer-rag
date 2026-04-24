"""
frontend/streamlit_app.py — Premium Chat UI for Consumer Law Agentic RAG

A polished, chat-first conversational interface designed for product demos.
Features:
  - Chat-first layout with st.chat_message / st.chat_input
  - Smart intent routing: legal answers show expandable reasoning, general answers are clean
  - Session memory for multi-turn conversations
  - Dark glassmorphism theme with animations
  - Sidebar: filters, KB status, tools (Draft, Upload)
"""

import requests
import streamlit as st
from pathlib import Path
import json
import uuid
import base64
import time as _time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumer Law AI — Legal Buddy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

API_BASE = "http://localhost:8000"

# ── Premium CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #f0f4f8;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --bg-glass: rgba(255, 255, 255, 0.8);
    --accent-primary: #6366f1; /* Friendly Indigo */
    --accent-secondary: #8b5cf6; /* Warm Purple */
    --accent-gold: #f59e0b;
    --accent-green: #10b981;
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    
    /* Dark mode overrides (Streamlit auto-adds dark mode class if user prefers) */
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #1e293b;
        --bg-glass: rgba(30, 41, 59, 0.75);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --border: #334155;
        --border-light: #1e293b;
    }
}

html, body, [class*="css"] {
    font-family: 'Nunito', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
}

/* ── Chat Container ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) ease-out;
}

[data-testid="chatAvatarIcon-user"] {
    background-color: var(--accent-primary) !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background-color: var(--accent-gold) !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 1rem;
    animation: slideDown 0.6s ease-out forwards;
}
.app-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.03em;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.app-header p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin: 0.5rem 0 0 0;
    font-weight: 600;
}

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.glass-card:hover {
    border-color: var(--accent-primary);
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.15);
}

/* ── Answer Bubbles ── */
.answer-bubble {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-green);
    border-radius: 18px;
    border-top-left-radius: 4px;
    padding: 1.4rem 1.8rem;
    line-height: 1.7;
    font-size: 1.05rem;
    color: var(--text-primary);
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    animation: fadeSlideIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.general-bubble {
    border-left: 4px solid var(--accent-primary);
}

/* ── Confidence Badge ── */
.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: var(--accent-green);
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-top: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    transition: transform 0.2s ease;
}
.confidence-badge:hover {
    transform: scale(1.05);
}

/* ── Metrics Row ── */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-chip {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
    text-align: center;
    min-width: 90px;
    transition: transform 0.2s ease;
}
.metric-chip:hover {
    transform: translateY(-2px);
    border-color: var(--accent-primary);
}
.metric-chip .val {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--accent-primary);
    display: block;
}
.metric-chip .lbl {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* ── Citation Tag ── */
.citation-tag {
    display: inline-block;
    background: rgba(99, 102, 241, 0.1);
    color: var(--accent-primary);
    border: 1px solid rgba(99, 102, 241, 0.2);
    font-size: 0.8rem;
    padding: 0.25rem 0.75rem;
    border-radius: 8px;
    margin: 0.2rem 0.15rem;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.2s ease;
    cursor: default;
}
.citation-tag:hover {
    background: rgba(99, 102, 241, 0.2);
    transform: translateY(-1px);
}

/* ── Trace Step ── */
.trace-step {
    background: var(--bg-primary);
    border-left: 4px solid var(--accent-secondary);
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 12px 12px 0;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
    font-weight: 600;
    transition: border-color 0.3s ease;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]:hover {
    border-color: var(--accent-primary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
    color: white !important;
    font-weight: 700 !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}
.stButton > button:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* ── Sample Question Chips (Secondary Buttons) ── */
[data-testid="column"] .stButton > button {
    background: var(--bg-card) !important;
    color: var(--accent-primary) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    white-space: normal !important;
    height: 100% !important;
}
[data-testid="column"] .stButton > button:hover {
    border-color: var(--accent-primary) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* ── Typing animation ── */
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.95) translateY(10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes bounce-pulse {
    0%, 100% { opacity: 0.4; transform: translateY(0); }
    50% { opacity: 1; transform: translateY(-4px); }
}
.typing-container {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 1rem 1.5rem;
    background: var(--bg-card);
    border-radius: 20px;
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}
.typing-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-primary);
    border-radius: 50%;
    animation: bounce-pulse 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Sidebar section headers ── */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent-primary);
    margin: 1.2rem 0 0.5rem 0;
}

hr { border-color: var(--border) !important; }

/* Clean up Streamlit UI */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Chat Input Styling */
[data-testid="stChatInput"] {
    border-radius: 24px !important;
    border: 2px solid var(--border) !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05) !important;
    background: var(--bg-card) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The query is taking longer than expected."}
    except Exception as e:
        return {"error": str(e)}


def check_backend() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def format_citation(c: dict) -> str:
    parts = []
    if c.get("act"):
        parts.append(c["act"])
    if c.get("section"):
        parts.append(f"§{c['section']}")
    if c.get("doc"):
        parts.append(c["doc"])
    if c.get("page"):
        parts.append(f"p.{c['page']}")
    return " · ".join(parts) if parts else "Unknown source"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Branding
    st.markdown("### ⚖️ Legal Buddy")
    st.caption("AI Consumer Law Assistant")
    st.divider()

    # Backend status
    backend_ok = check_backend()
    if backend_ok:
        st.success("● Backend connected", icon="🟢")
    else:
        st.error("● Backend offline — run `uvicorn app:app --reload`", icon="🔴")

    # ── Session Controls ──
    st.markdown('<div class="sidebar-section">💬 Session</div>', unsafe_allow_html=True)
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    # ── Search Filters ──
    st.markdown('<div class="sidebar-section">🔍 Filters</div>', unsafe_allow_html=True)
    jurisdiction = st.selectbox(
        "Jurisdiction",
        ["All", "India", "Central", "State", "District"],
        help="Filter by legal jurisdiction"
    )
    domain = st.selectbox(
        "Domain",
        ["All", "consumer_protection", "food_safety", "telecom"],
        help="Filter by law domain"
    )
    doc_type = st.selectbox(
        "Document Type",
        ["All", "Act", "Regulations", "Notification", "Directions", "Rules"],
    )
    use_llm_expansion = st.toggle("AI Query Expansion", value=True,
                                   help="Use Gemini to expand query into legal terms")

    st.divider()

    # ── Case Context ──
    st.markdown('<div class="sidebar-section">📋 Case Context</div>', unsafe_allow_html=True)
    claim_value = st.number_input(
        "Claim Value (₹)",
        min_value=0,
        value=0,
        step=10000,
        help="Helps determine the correct forum jurisdiction"
    )
    language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati", "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi"],
        index=0,
    )

    st.divider()

    # ── Tools ──
    st.markdown('<div class="sidebar-section">🛠️ Tools</div>', unsafe_allow_html=True)

    with st.expander("📝 Draft Legal Document"):
        draft_type = st.selectbox("Document Type", [
            "Legal Notice to Seller/Service Provider",
            "Consumer Complaint (District Commission)",
            "RTI Application",
            "Grievance Letter to Nodal Officer"
        ], key="draft_type_select")

        draft_facts = st.text_area(
            "Facts of your case",
            placeholder="E.g., I bought an AC on 10th March from ABC Electronics...",
            height=100,
            key="draft_facts_input"
        )

        if st.button("✍️ Generate Draft", use_container_width=True, key="draft_button"):
            if draft_facts.strip() and backend_ok:
                with st.spinner("Drafting..."):
                    res = api_post("/draft_document", {"document_type": draft_type, "facts": draft_facts})
                if "error" not in res:
                    st.session_state.messages.append({"role": "user", "content": f"Draft a {draft_type}"})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res["draft"],
                        "intent": "general"
                    })
                    st.rerun()
                else:
                    st.error(res["error"])

    with st.expander("📤 Upload Document"):
        uploaded = st.file_uploader(
            "Add to Knowledge Base",
            type=["pdf", "docx", "txt", "md", "png", "jpg"],
            key="doc_upload_sidebar"
        )
        subfolder_options = [
            "consumer_protection/general_provisions",
            "consumer_protection/product_liability",
            "consumer_protection/redressal_forums",
            "food_safety/fssai",
            "telecom/trai",
        ]
        target_subfolder = st.selectbox("Target", subfolder_options, key="upload_target")
        admin_pass = st.text_input("Admin Password", type="password", key="upload_pass")

        if st.button("⬆️ Upload & Index", use_container_width=True, key="upload_button"):
            if uploaded and admin_pass and backend_ok:
                with st.spinner(f"Uploading {uploaded.name}..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    data = {"subfolder_path": target_subfolder, "admin_password": admin_pass}
                    try:
                        r = requests.post(f"{API_BASE}/ingest/upload", files=files, data=data, timeout=180)
                        if r.status_code == 200:
                            st.success(f"✅ {uploaded.name} indexed!")
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(str(e))

    st.divider()

    # ── KB Status ──
    st.markdown('<div class="sidebar-section">📚 Knowledge Base</div>', unsafe_allow_html=True)
    if "kb_data" not in st.session_state:
        st.session_state["kb_data"] = api_get("/knowledge-base")
    kb_data = st.session_state.get("kb_data", {})
    entries = kb_data.get("entries", [])
    if entries:
        for entry in entries:
            idx = "✅" if entry.get("indexed") else "⬜"
            name = entry.get("subdomain", entry.get("domain", "?"))
            chunks = entry.get("total_chunks", 0) or 0
            st.caption(f"{idx} **{name}** — {chunks} chunks")
    else:
        st.caption("No KB data available")
    if st.button("Refresh", key="kb_refresh", use_container_width=True):
        st.session_state["kb_data"] = api_get("/knowledge-base")
        st.rerun()


# ── Main Chat Area ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <h1>⚖️ Legal Buddy</h1>
    <p>Your AI-powered Indian Consumer Law Assistant · Agentic RAG with Gemini</p>
</div>
""", unsafe_allow_html=True)

# Welcome message if no conversation yet
if not st.session_state.messages:
    st.markdown("""
    <div class="glass-card" style="text-align: center; max-width: 700px; margin: 0 auto 1.5rem auto;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">👋 <strong>Welcome!</strong> I'm your Legal Buddy.</p>
        <p style="color: var(--text-secondary); font-size: 0.88rem; margin: 0;">
            Ask me anything about Indian consumer law — refunds, complaints, compensation, legal notices,
            or just chat! I'll search through official legal documents and give you clear, actionable advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sample question chips
    sample_questions = [
        "Can I get a refund for a defective product?",
        "How to file a consumer complaint?",
        "Time limit for consumer complaints?",
        "Compensation for mental agony?",
    ]
    cols = st.columns(len(sample_questions))
    for i, (col, q) in enumerate(zip(cols, sample_questions)):
        if col.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["_pending_sample"] = q
            st.rerun()


# ── Render Chat History ────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚖️"
    with st.chat_message(msg["role"], avatar=avatar):
        intent = msg.get("intent", "legal")

        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Render answer in styled bubble
            bubble_class = "answer-bubble" if intent == "legal" else "answer-bubble general-bubble"
            st.markdown(f'<div class="{bubble_class}">\n\n{msg["content"]}\n\n</div>', unsafe_allow_html=True)

            # For legal responses: show confidence badge + optional reasoning expander
            if intent == "legal" and "result_data" in msg:
                res = msg["result_data"]
                conf = res.get("citation_confidence", 0)
                pct = int(conf * 100)
                st.markdown(f'<span class="confidence-badge">✓ {pct}% citation confidence</span>', unsafe_allow_html=True)

                # Expandable reasoning & sources (hidden by default)
                with st.expander("🔍 View reasoning & sources", expanded=False):
                    timing = res.get("timing", {})
                    citations = res.get("verified_citations", [])
                    steps = res.get("reasoning_steps", [])
                    chunks = res.get("source_chunks", [])
                    trace_items = res.get("trace", [])

                    # Metrics row
                    st.markdown(f"""
                    <div class="metrics-row">
                        <div class="metric-chip"><span class="val">{len(citations)}</span><span class="lbl">Citations</span></div>
                        <div class="metric-chip"><span class="val">{len(chunks)}</span><span class="lbl">Sources</span></div>
                        <div class="metric-chip"><span class="val">{sum(timing.values()):.1f}s</span><span class="lbl">Total Time</span></div>
                        <div class="metric-chip"><span class="val">{pct}%</span><span class="lbl">Confidence</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Pipeline trace
                    if trace_items:
                        st.markdown("**Pipeline Trace**")
                        for t in trace_items:
                            st.markdown(f'<div class="trace-step">{t}</div>', unsafe_allow_html=True)

                    # Citations
                    if citations:
                        st.markdown("**Verified Citations**")
                        for c in citations:
                            st.markdown(f'<span class="citation-tag">{format_citation(c)}</span>', unsafe_allow_html=True)

                    # Source chunks
                    if chunks:
                        st.markdown("**Source Chunks**")
                        for i_ch, ch in enumerate(chunks[:4], 1):
                            with st.container():
                                st.caption(f"Chunk {i_ch} · {ch.get('doc', '?')} · p.{ch.get('page', '?')}")
                                st.text(ch.get("text", "")[:200] + "...")


# ── Inline File Attacher ───────────────────────────────────────────────────────

uploaded_tmp = st.file_uploader(
    "📎 Attach a document or image to your message",
    type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
    key="chat_upload",
    label_visibility="collapsed",
)
if uploaded_tmp and st.session_state.get("last_uploaded") != uploaded_tmp.name:
    with st.spinner(f"Indexing {uploaded_tmp.name} into session context..."):
        files = {"file": (uploaded_tmp.name, uploaded_tmp.getvalue(), uploaded_tmp.type)}
        data = {"subfolder_path": f"temp/{st.session_state.session_id}"}
        try:
            requests.post(f"{API_BASE}/ingest/upload", files=files, data=data)
            st.session_state["last_uploaded"] = uploaded_tmp.name
            if uploaded_tmp.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                b64 = base64.b64encode(uploaded_tmp.getvalue()).decode()
                st.session_state["attached_image_b64"] = f"data:{uploaded_tmp.type};base64,{b64}"
            st.success(f"✅ Attached {uploaded_tmp.name}")
        except Exception as e:
            st.error(f"Failed: {e}")


# ── Chat Input Handler ────────────────────────────────────────────────────────

# Check for pending sample question
prompt = st.chat_input("Ask me anything about consumer law...")

if st.session_state.get("_pending_sample"):
    prompt = st.session_state.pop("_pending_sample")

if prompt:
    if not backend_ok:
        st.error("Backend is offline. Start it with: `uvicorn app:app --reload`")
    else:
        b64_to_send = st.session_state.pop("attached_image_b64", None)

        # Append & render user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            if b64_to_send:
                st.image(b64_to_send, width=200)

        # Generate response
        with st.chat_message("assistant", avatar="⚖️"):
            # Typing indicator
            typing_placeholder = st.empty()
            typing_placeholder.markdown(
                '<div class="typing-container"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>',
                unsafe_allow_html=True
            )

            # Build request
            filters = {"session_id": st.session_state.session_id}
            if jurisdiction != "All": filters["jurisdiction"] = jurisdiction
            if domain != "All": filters["domain"] = domain
            if doc_type != "All": filters["doc_type"] = doc_type

            result = api_post("/query", {
                "query": prompt,
                "filters": filters,
                "use_llm_expansion": use_llm_expansion,
                "attached_image_b64": b64_to_send,
                "claim_value": claim_value,
                "preferred_language": language
            })

            typing_placeholder.empty()

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                ans = result["answer"]
                intent = result.get("intent", "legal")
                bubble_class = "answer-bubble" if intent == "legal" else "answer-bubble general-bubble"

                st.markdown(f'<div class="{bubble_class}">\n\n{ans}\n\n</div>', unsafe_allow_html=True)

                # Show confidence badge for legal responses
                if intent == "legal":
                    conf = result.get("citation_confidence", 0)
                    pct = int(conf * 100)
                    st.markdown(f'<span class="confidence-badge">✓ {pct}% citation confidence</span>', unsafe_allow_html=True)

                # Store in session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans,
                    "intent": intent,
                    "result_data": result if intent == "legal" else None,
                })
                st.rerun()
