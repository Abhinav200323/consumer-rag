"""
frontend/streamlit_app.py — Streamlit UI for Consumer Law Agentic RAG

Features:
  - Query panel with Gemini-powered legal reasoning
  - Sidebar filters (jurisdiction, domain, doc type, date)
  - Reasoning steps viewer (expandable)
  - Source citations with page numbers
  - Reference traversal tree
  - Document browser & upload ingestion tab
"""

import requests
import streamlit as st
from pathlib import Path
import json
import uuid
import base64

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumer Law AI — Agentic RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

API_BASE = "http://localhost:8000"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2330;
    --accent-gold: #d4a017;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-purple: #bc8cff;
    --accent-orange: #f0883e;
    --text-primary: #e6edf3;
    --text-muted: #8b949e;
    --border: #30363d;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp { background-color: var(--bg-primary); }

/* Header */
.hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #d4a017 0%, #f0c040 40%, #58a6ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: var(--text-muted); font-size: 1rem; margin-bottom: 2rem;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-gold { border-left: 4px solid var(--accent-gold); }
.card-blue { border-left: 4px solid var(--accent-blue); }
.card-green { border-left: 4px solid var(--accent-green); }
.card-purple { border-left: 4px solid var(--accent-purple); }
.card-orange { border-left: 4px solid var(--accent-orange); }

/* Step badge */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #d4a017, #b8860b);
    color: #000; font-weight: 700; font-size: 0.75rem;
    padding: 0.15rem 0.5rem; border-radius: 6px; margin-right: 0.4rem;
}

/* Citation tag */
.citation-tag {
    display: inline-block;
    background: #1f3550; color: #58a6ff;
    border: 1px solid #2d4a6e;
    font-size: 0.75rem; padding: 0.15rem 0.5rem;
    border-radius: 4px; margin: 0.15rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric mini */
.metric-mini {
    background: var(--bg-secondary);
    border-radius: 8px; padding: 0.6rem 1rem;
    text-align: center; border: 1px solid var(--border);
}
.metric-mini .val { font-size: 1.4rem; font-weight: 700; color: var(--accent-gold); }
.metric-mini .lbl { font-size: 0.72rem; color: var(--text-muted); }

/* Answer box */
.answer-box {
    background: #0d1f12;
    border: 1px solid #1a4a26;
    border-radius: 10px; padding: 1.4rem;
    line-height: 1.7; color: var(--text-primary);
    font-size: 0.95rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] { background-color: var(--bg-secondary); }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
section[data-testid="stSidebar"] .stTextInput input {
    background-color: var(--bg-card);
    border-color: var(--border);
    color: var(--text-primary);
}

/* Trace step */
.trace-line {
    background: var(--bg-secondary);
    border-left: 3px solid var(--accent-blue);
    padding: 0.3rem 0.8rem;
    margin: 0.25rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #d4a017, #b8860b) !important;
    color: #000 !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2rem !important; width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

hr { border-color: var(--border); }
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
    return " | ".join(parts) if parts else "Unknown source"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ Consumer Law RAG")
    st.markdown("*Agentic AI Legal Assistant*")
    st.divider()

    backend_ok = check_backend()
    if backend_ok:
        st.success("🟢 Backend connected")
    else:
        st.error("🔴 Backend offline\n\nRun: `uvicorn app:app --reload`")

    st.markdown("### 🔍 Search Filters")
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
    use_llm_expansion = st.toggle("🔮 AI Query Expansion", value=True,
                                   help="Use Gemini to expand query into legal terms")

    st.divider()
    st.markdown("### 📋 Case Context")
    claim_value = st.number_input(
        "Claim Value (₹)", 
        min_value=0, 
        value=0, 
        step=10000,
        help="The total value of goods/services + compensation claimed. Helps determine jurisdiction."
    )
    
    language = st.selectbox(
        "Preferred Language",
        ["English", "Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati", "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi"],
        index=0,
        help="The language in which you want to receive the legal advice."
    )

    st.divider()
    st.markdown("### 📚 Knowledge Base")

    if st.button("🔄 Refresh KB Status"):
        st.session_state["kb_data"] = api_get("/knowledge-base")

    if "kb_data" not in st.session_state:
        st.session_state["kb_data"] = api_get("/knowledge-base")

    kb_data = st.session_state.get("kb_data", {})
    entries = kb_data.get("entries", [])

    if entries:
        for entry in entries:
            idx = "✅" if entry.get("indexed") else "⬜"
            st.markdown(f"{idx} **{entry.get('subdomain', entry.get('domain', '?'))}**")
            if entry.get("acts"):
                st.caption(f"  {', '.join(entry['acts'][:2])}")
    else:
        st.caption("No knowledge base data (backend offline?)")


# ── Main Tabs ──────────────────────────────────────────────────────────────────

st.markdown('<p class="hero-title">⚖️ Consumer Law AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Agentic RAG powered by hybrid vector + BM25 search and Gemini legal reasoning</p>', unsafe_allow_html=True)

tab_query, tab_draft, tab_games, tab_ingest, tab_about = st.tabs([
    "🔍 Ask a Legal Question", 
    "📝 Draft Document", 
    "🎮 Learning Games", 
    "📤 Upload Documents", 
    "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.markdown("**💡 Try a sample question:**")
    sample_questions = [
        "Can I get a refund for a defective product I bought online?",
        "How do I file a complaint against a company for unfair trade practices?",
        "What is the time limit to file a consumer complaint?",
        "Can I claim compensation for mental agony caused by service deficiency?",
    ]
    
    cols = st.columns(len(sample_questions))
    selected_sample = None
    for i, (col, q) in enumerate(zip(cols, sample_questions)):
        if col.button(f"Q{i+1}", help=q, key=f"sample_{i}"):
            selected_sample = q
            
    st.divider()
    
    # ── Chat history renderer ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant" and "result_data" in msg:
                res = msg["result_data"]
                timing = res.get("timing", {})
                citations = res.get("verified_citations", [])
                steps = res.get("reasoning_steps", [])
                chunks = res.get("source_chunks", [])
                conf = res.get("citation_confidence", 1.0)

                with st.expander("📋 View Reasoning & Legal Context", expanded=False):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1: st.markdown(f'<div class="metric-mini"><div class="val">{len(citations)}</div><div class="lbl">Citations</div></div>', unsafe_allow_html=True)
                    with m2: st.markdown(f'<div class="metric-mini"><div class="val">{len(steps)}</div><div class="lbl">Reasoning Steps</div></div>', unsafe_allow_html=True)
                    with m3: st.markdown(f'<div class="metric-mini"><div class="val">{len(chunks)}</div><div class="lbl">Source Chunks</div></div>', unsafe_allow_html=True)
                    with m4:
                        total_t = sum(timing.values())
                        st.markdown(f'<div class="metric-mini"><div class="val">{total_t:.1f}s</div><div class="lbl">Total Time</div></div>', unsafe_allow_html=True)
                    with m5:
                        pct = int(conf * 100)
                        st.markdown(f'<div class="metric-mini"><div class="val">{pct}%</div><div class="lbl">Confidence</div></div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    if steps:
                        st.markdown("#### 🧠 Step-by-Step Reasoning")
                        for step in steps:
                            st.markdown(f"**Step {step['step']} — {step['title']}**\n\n{step['content']}")
                            
                    st.divider()
                    col_cit, col_src = st.columns([1, 1])
                    with col_cit:
                        st.markdown("#### 📎 Verified Citations")
                        for c in citations:
                            st.info(f"**{format_citation(c)}**\n\n{c.get('text_snippet', '')}")
                    with col_src:
                        st.markdown("#### 📄 Source Chunks")
                        for i, ch in enumerate(chunks[:4], 1):
                            st.caption(f"Chunk {i} — {ch.get('doc', '?')} p.{ch.get('page', '?')}")
                            st.text(ch.get("text", "")[:200] + "...")

    # ── Inline Attacher ──
    uploaded_tmp = st.file_uploader("📎 Attach Document or Image to Chat (PDF, DOCX, TXT, PNG, JPG)", type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"], key="chat_upload")
    if uploaded_tmp and st.session_state.get("last_uploaded") != uploaded_tmp.name:
        with st.spinner(f"Indexing {uploaded_tmp.name} permanently into your secure session context..."):
             files = {"file": (uploaded_tmp.name, uploaded_tmp.getvalue(), uploaded_tmp.type)}
             data = {"subfolder_path": f"temp/{st.session_state.session_id}"}
             try:
                 requests.post(f"{API_BASE}/ingest/upload", files=files, data=data)
                 st.session_state["last_uploaded"] = uploaded_tmp.name
                 if uploaded_tmp.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                     b64 = base64.b64encode(uploaded_tmp.getvalue()).decode()
                     st.session_state["attached_image_b64"] = f"data:{uploaded_tmp.type};base64,{b64}"
                 st.success(f"Successfully attached {uploaded_tmp.name} to conversation.")
             except Exception as e:
                 st.error(f"Failed to attach: {e}")

    # ── Input Handler ──
    prompt = st.chat_input("Ask a legal question... (e.g. Can I get a refund on a defective product?)")
    
    if selected_sample:
        prompt = selected_sample

    if prompt:
        if not backend_ok:
            st.error("Backend is offline. Start it with: `uvicorn app:app --reload`")
        else:
            b64_to_send = st.session_state.pop("attached_image_b64", None)
            
            # Append & render user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                if b64_to_send:
                    st.image(b64_to_send, width=200)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Searching knowledge base and structuring response..."):
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

                if "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    ans = result["answer"]
                    st.markdown(f'<div class="answer-box"> \n{ans} \n </div>', unsafe_allow_html=True)
                    # Cache message natively
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f'<div class="answer-box"> \n{ans} \n </div>', 
                        "result_data": result
                    })
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRAFTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_draft:
    st.markdown("## 📝 Legal Document Drafting")
    st.markdown("Generate formal Indian legal drafts automatically from your case facts.")
    
    doc_type = st.selectbox("Select Document Type", [
        "Legal Notice to Seller/Service Provider",
        "Consumer Complaint (District Commission)",
        "RTI Application", 
        "Grievance Letter to Nodal Officer"
    ])
    
    facts = st.text_area(
        "Enter the facts of your case", 
        placeholder="E.g., I bought an AC on 10th March from ABC Electronics. It stopped cooling on 12th March. The dealer refused to replace or repair it despite warranty...",
        height=150
    )
    
    if st.button("✍️ Generate Legal Draft", type="primary"):
        if not facts.strip():
            st.warning("Please provide the facts of your case to generate a draft.")
        elif not backend_ok:
            st.error("Backend is not running.")
        else:
            with st.spinner("⚖️ Applying Indian Consumer Law and drafting your document..."):
                res = api_post("/draft_document", {"document_type": doc_type, "facts": facts})
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success("Draft generated successfully! Note: Always review with a legal professional.")
                    st.markdown("### Your Draft Document")
                    st.info("Copy the text below to your editor.")
                    st.markdown(res["draft"])
                    
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GAMES
# ══════════════════════════════════════════════════════════════════════════════
with tab_games:
    st.markdown("## 🎮 Consumer Law Learning Games")
    st.markdown("Test your knowledge of Indian Consumer Law with these quick interactive exercises!")
    
    st.divider()
    
    st.markdown("### Game 1: Is it a Violation?")
    st.markdown("Read the scenario and decide if it violates the Indian Consumer Protection Act.")
    
    # Simple state handling for Game 1
    if "g1_answered" not in st.session_state:
        st.session_state.g1_answered = False
        
    scenario = "A popular restaurant automatically adds a 'Service Charge' of 10% to your bill without asking you. They insist you must pay it."
    st.info(f"**Scenario:** {scenario}")
    
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("⚖️ Yes, it's a violation!"):
            st.session_state.g1_answered = "yes"
    with col_no:
        if st.button("🤷 No, it's legal."):
            st.session_state.g1_answered = "no"
            
    if st.session_state.g1_answered == "yes":
        st.success("**Correct!** The CCPA (Central Consumer Protection Authority) guidelines clearly state that levying service charge automatically or by default is an 'unfair trade practice'.")
    elif st.session_state.g1_answered == "no":
        st.error("**Incorrect.** It is a violation! According to CCPA guidelines, forcing a consumer to pay a service charge is considered an 'unfair trade practice' under the Consumer Protection Act, 2019.")

    st.divider()
    
    st.markdown("### Game 2: Time Limit Trivia")
    st.markdown("Deadlines are critical in law. Can you guess the correct timeline?")
    
    if "g2_answered" not in st.session_state:
        st.session_state.g2_answered = False
        
    q = "**What is the maximum time limit to file a complaint in the District Consumer Forum from the date the issue occurred?**"
    ans = st.radio(q, ["1 Year", "2 Years", "3 Years", "No Time Limit"])
    
    if st.button("Check Answer 🎯"):
        st.session_state.g2_answered = ans
        
    if st.session_state.g2_answered:
        if st.session_state.g2_answered == "2 Years":
            st.success("**Correct!** Under Section 69 of the Consumer Protection Act, 2019, you must file a complaint within **2 years** from the date the cause of action arises.")
        else:
            st.error(f"**Incorrect.** You chose {st.session_state.g2_answered}, but the correct answer is **2 Years** (Section 69 of the Act).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("## 📤 Upload & Index Documents")
    st.markdown("Upload legal documents (PDF, DOCX, TXT) to the knowledge base. The system will automatically parse, chunk, build vector + BM25 indexes with page tracking.")

    col_up, col_trig = st.columns([1, 1])

    with col_up:
        st.markdown("### Upload a Document (Knowledge Base Registry)")
        
        uploaded = st.file_uploader(
            "Choose file (PDF, DOCX, TXT, PNG)",
            type=["pdf", "docx", "txt", "md", "png", "jpg"],
            key="doc_upload_admin"
        )
        
        subfolder_options = [
            "consumer_protection/general_provisions",
            "consumer_protection/product_liability",
            "consumer_protection/redressal_forums",
            "food_safety/fssai",
            "telecom/trai",
        ]
        target_subfolder = st.selectbox("Target subfolder", subfolder_options)
        admin_pass = st.text_input("Admin Password", type="password")

        if st.button("⬆️ Upload & Index globally"):
            if not uploaded:
                st.warning("Please select a file first.")
            elif not admin_pass:
                st.warning("Admin password lies strictly required for permanent registry.")
            elif not backend_ok:
                st.error("Backend is not running.")
            else:
                with st.spinner(f"Uploading and indexing {uploaded.name}..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    data = {"subfolder_path": target_subfolder}
                    if admin_pass:
                        data["admin_password"] = admin_pass
                        
                    try:
                        r = requests.post(
                            f"{API_BASE}/ingest/upload",
                            files=files, data=data, timeout=180
                        )
                        if r.status_code == 200:
                            st.success(f"✅ {uploaded.name} uploaded and indexed securely!")
                            st.session_state["kb_data"] = api_get("/knowledge-base")
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(str(e))

    with col_trig:
        st.markdown("### Re-index Existing Subfolder")
        st.markdown("Trigger full re-indexing of a subfolder that already has documents.")
        reindex_target = st.selectbox(
            "Select subfolder to re-index",
            subfolder_options,
            key="reindex_select"
        )
        if st.button("🔄 Re-index Subfolder"):
            if not backend_ok:
                st.error("Backend is not running.")
            else:
                with st.spinner(f"Indexing {reindex_target}..."):
                    result = api_post("/ingest", {"subfolder_path": reindex_target})
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ Indexed {reindex_target}")
                    st.code(result.get("output", ""), language="bash")
                    st.session_state["kb_data"] = api_get("/knowledge-base")

    st.divider()
    st.markdown("### 📊 Knowledge Base Status")
    kb_data = st.session_state.get("kb_data", {})
    entries = kb_data.get("entries", [])
    if entries:
        rows = []
        for e in entries:
            rows.append({
                "Domain": e.get("domain", ""),
                "Subdomain": e.get("subdomain", ""),
                "Indexed": "✅" if e.get("indexed") else "⬜",
                "Chunks": e.get("total_chunks", 0) or 0,
                "Docs": e.get("total_docs", 0) or 0,
                "Last Indexed": e.get("last_indexed", "Never")[:19] if e.get("last_indexed") else "Never",
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Start the backend and refresh to see knowledge base status.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## ℹ️ About this System")
    st.markdown("""
    ### 🏗️ Architecture — 3-Layer Agentic RAG

    #### 🗂️ Layer 1 — Retrieval
    | Component | Description |
    |---|---|
    | **Hybrid Search** | Dense FAISS vector + sparse BM25 fused via RRF |
    | **Query Expansion** | Legal synonym mapping + Gemini LLM expansion |
    | **Metadata Filtering** | Jurisdiction / domain / doc-type subfolder selection |
    | **Page Indexing** | Every chunk tagged with page number for exact citation |
    | **Structured Index** | Act → Section → Clause navigation tree |

    #### 🧩 Layer 2 — Context
    | Component | Description |
    |---|---|
    | **Compression** | Boilerplate removal + near-duplicate deduplication |
    | **Re-ranking** | Cross-encoder (query, chunk) scoring |
    | **Reference Traversal** | Follows "See Section X" links recursively |

    #### 🤖 Layer 3 — Reasoning (Agentic)
    | Component | Description |
    |---|---|
    | **Planner** | 9-step orchestration pipeline |
    | **Gemini Reasoning** | Structured CoT: Facts → Law → Analysis → Conclusion |
    | **Citation Verification** | Cross-checks every cited section against retrieved context |

    ---
    ### 🚀 Quick Start
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Set your Gemini API key
    cp .env.example .env
    # Edit .env and add your GEMINI_API_KEY

    # 3. Index the knowledge base
    python ingest.py --all

    # 4. Start the backend (Terminal 1)
    uvicorn app:app --reload --port 8000

    # 5. Start the frontend (Terminal 2)
    streamlit run frontend/streamlit_app.py --server.port 8501
    ```
    """)
