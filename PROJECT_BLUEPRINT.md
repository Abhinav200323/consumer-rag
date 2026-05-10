# Lex Assist: Project Blueprint & Technical Stack

This document provides an exhaustive breakdown of the technologies, algorithms, and mathematical logic used to build the **Lex Assist** Agentic RAG platform.

---

## 1. High-Level Architecture
The project follows a **Decoupled Client-Server Architecture**:
*   **Frontend**: React 18 (Vite) — A single-page application focused on high-performance interactive states.
*   **Backend**: FastAPI (Python 3.11+) — An asynchronous, high-concurrency API layer.
*   **Vector Engine**: FAISS (Facebook AI Similarity Search) — Used for sub-millisecond dense vector retrieval.
*   **Keyword Engine**: BM25 (Okapi) — Used for precise lexical/term-frequency matching.

---

## 2. Model Selection & Rationale

### **A. LLM: Gemini 1.5 Flash**
*   **Why**: We selected Gemini 1.5 Flash as the primary reasoning engine due to its unique balance of speed and "Long Context" capabilities.
*   **Context Window**: 1,000,000+ tokens.
*   **Role**: It handles Intent Classification (Step 0), Query Expansion (Step 1), Metadata Extraction (Ingestion), and final Legal Reasoning (Step 8). Its ability to ingest entire legal acts into its working memory allows for high-fidelity reasoning.

### **B. Embedding Model: BAAI/bge-base-en-v1.5**
*   **Why**: BGE (Baidu General Embedding) is currently one of the top-performing models on the MTEB (Massive Text Embedding Benchmark). It is specifically optimized for retrieval tasks.
*   **Dimensions**: 768-dimensional vectors.
*   **Context Window**: 512 tokens.
*   **Similarity Metric**: **Cosine Similarity**. Unlike dot product, cosine similarity is scale-invariant, making it more robust for varying legal paragraph lengths.

### **C. Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2**
*   **Why**: While vector search is fast, it loses nuance. A Cross-Encoder processes the (Query, Chunk) pair simultaneously, allowing for interaction between the words in the query and the chunk, providing much higher precision.

---

## 3. The 10-Step Agentic RAG Pipeline

| Step | Technique | Mathematical/Algorithmic Basis |
| :--- | :--- | :--- |
| **1. Intent Routing** | Zero-Shot Classif. | LLM-based decision boundary to separate `Legal RAG` from `General Chat`. |
| **2. Query Expansion** | Multi-Query Gen. | Expands $Q \rightarrow \{Q_1, Q_2, \dots, Q_n\}$ to increase "recall" by including synonyms. |
| **3. Metadata Filter** | Boolean Indexing | Narrows search space to $S \subset KB$ based on Act/Category tags. |
| **4. Hybrid Search** | **RRF Fusion** | Merges Dense (FAISS) + Sparse (BM25) rankings using the RRF formula. |
| **5. Compression** | Context Distillation | Removes stop-words and legal boilerplate to maximize information density. |
| **6. Re-ranking** | **Cross-Encoder** | Re-scores top 20 chunks using a transformer-based relevance model. |
| **7. MMR Diversity** | **Maximal Marginal Relevance** | Penalizes redundancy to ensure the top $K$ chunks cover different legal angles. |
| **8. Ref Traversal** | Recursive Retrieval | Follows internal citations (e.g., "See Section 12") to fetch missing context. |
| **9. LLM Reasoning** | Chain-of-Thought | Gemini uses "Thinking Steps" to decompose legal analysis. |
| **10. Self-Correct** | Hallucination Check | Cross-verifies every citation generated against the source index. |

---

## 4. Mathematical Foundations

### **A. Cosine Similarity (Semantic Match)**
The similarity between a query vector $\mathbf{q}$ and a document vector $\mathbf{d}$ is calculated as:
$$ \text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}} $$
*   **Usage**: Used by FAISS to find chunks that are conceptually similar to the user's question.

### **B. BM25 Okapi (Keyword Match)**
For legal queries involving specific section numbers (e.g., "Section 34"), keyword matching is superior to vectors.
$$ \text{score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})} $$
*   **IDF**: Inverse Document Frequency.
*   **$k_1, b$**: Saturation and length normalization constants (typically 1.5 and 0.75).

### **C. Reciprocal Rank Fusion (Hybrid Merge)**
Hybrid search combines the strengths of BM25 and Vector Search. We merge them using RRF to avoid "score scale mismatch":
$$ \text{RRFScore}(d) = \sum_{r \in R} \frac{w_r}{k + \text{rank}(r, d)} $$
*   **$k$**: Smoothing constant (default 60).
*   **$w_r$**: Weights (0.6 for Embedding, 0.4 for BM25).

### **D. MMR: Maximal Marginal Relevance (Diversity)**
To prevent the LLM from receiving 5 identical chunks, we use MMR to select the next chunk $D_i$ that is relevant but different from already selected chunks $S$:
$$ \text{MMR} = \arg \max_{D_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in S} \text{Sim}_2(D_i, D_j) \right] $$
*   **$\text{Sim}_1$**: Relevance (Query-Doc similarity).
*   **$\text{Sim}_2$**: Redundancy (Doc-Doc similarity).
*   **$\lambda$**: Balance parameter (0.7). High $\lambda$ = high relevance; low $\lambda$ = high diversity.

---

## 5. Why Hybrid + MMR?

### **The Hybrid Advantage**
Legal documents are full of "Hard Keywords" (Acts, Section numbers, Case names) and "Soft Concepts" (Negligence, Liability, Duty of Care). 
- **BM25** ensures that if you ask for "Section 12", you actually get Section 12.
- **FAISS** ensures that if you ask about "medical mistakes", you get results for "medical negligence".
By using **RRF**, we get the best of both worlds.

### **The MMR Rationale**
In RAG, token space is precious. If the top 10 search results are all minor variations of the same "Definition of Consumer", the LLM might miss the "Exceptions" or "Remedies" found in lower-ranked but more diverse chunks. MMR "pushes down" redundant info and "pulls up" novel facts, providing the LLM with a 360-degree view of the legal issue.

---

## 6. Agentic RAG vs. Traditional RAG

### **The Traditional Approach (The Baseline)**
Traditional RAG systems typically follow a linear, "one-shot" path:
`User Query → Embedding → Vector Search → LLM Generation`.
*   **Weakness**: If the initial search fails (e.g., poor query phrasing), the system has no way to recover. It often returns redundant chunks or hallucinated citations.

### **The Lex Assist Advantage (Agentic RAG)**
Our pipeline is "Agentic" because it treats retrieval as a multi-step reasoning task rather than a simple database lookup:
1.  **Self-Correction**: If the LLM produces a citation that doesn't exist in our verified index, the system catches it and re-runs the prompt with a "Correction Instruction."
2.  **Query Expansion**: It doesn't trust the user's phrasing. It uses Gemini to expand a casual question into professional legal terminology *before* searching.
3.  **Reference Traversal**: It understands that legal sections don't exist in a vacuum. If Section 12 mentions Section 5, the system automatically fetches Section 5 to provide a complete answer.
4.  **Intent Guardrails**: It avoids the "RAG-on-everything" trap. By classifying intent first, it provides instant responses for greetings and general help, reserving compute for complex legal research.

---

## 7. Overcoming Challenges in Consumer Law RAG

Legal RAG is significantly more difficult than general RAG due to the precision required. Below is how Lex Assist solves common industry failures:

| Challenge | Traditional RAG Failure | Lex Assist Solution | Why it Matters |
| :--- | :--- | :--- | :--- |
| **Isolated Context** | Chunks are retrieved as fragments. A section might refer to "the said Act" without context. | **Reference Traversal** | Automatically fetches referenced sections (e.g., "See Sec 5") even if not in top-K search results. |
| **Citation Hallucination** | LLMs often "guess" section numbers or mix up CPA 1986 with CPA 2019. | **Citation Verification** | Every section generated is cross-checked against the database. If it doesn't exist, the system triggers a **Self-Correction** loop. |
| **Phrasing Mismatch** | A user says "I was scammed," but the law says "Unfair Trade Practice." | **Query Expansion** | Gemini translates layman terms into legal keywords *before* retrieval, ensuring we find the right law. |
| **Information Redundancy** | Top results are often 5 different pages of the same legal preamble. | **MMR (Maximal Marginal Relevance)** | Penalizes similarity between results, forcing the engine to pick diverse chunks (e.g., 1 definition, 1 remedy, 1 exception). |
| **Boilerplate Noise** | Headers, footers, and page numbers pollute the embedding space. | **Context Compression** | Uses regex and LLM distillation to strip non-legal noise from chunks before they reach the reasoning engine. |
| **Numerical Sensitivity** | Vector models struggle to distinguish "Section 12" from "Section 13". | **Hybrid Search** | BM25 keyword matching treats "Section 12" as a unique token, ensuring 100% accuracy for specific citations. |

---

## 8. Dual-Embedding Strategy: Vector vs. Vectorless

### **Why Both?**
Retrieving legal data requires two different types of "intelligence":

| Feature | Vector (Dense) | Vectorless (Sparse/BM25) |
| :--- | :--- | :--- |
| **Logic** | Semantic/Conceptual Similarity | Lexical/Keyword Matching |
| **Best For** | "Can I sue for a broken phone?" | "Section 2(1)(d) of CPA" |
| **Strength** | Handles synonyms and context. | Handles specific codes and numbers. |
| **Weakness** | Can "smear" specific numbers (e.g., confused Section 12 and 13). | Cannot find "fraud" if you search for "cheating." |

### **How they are Created**

#### **1. Dense Vector Creation (FAISS)**
1.  **Chunking**: Documents are split into 512-character overlapping chunks.
2.  **Encoding**: Each chunk is passed through the `BGE-base-en-v1.5` transformer model.
3.  **Vectorization**: The model outputs a 768-dimensional array of floating-point numbers representing the "meaning" of that text.
4.  **Indexing**: These vectors are stored in a FAISS index using an **Inner Product (FlatIP)** structure for fast similarity lookups.

#### **2. Sparse Vectorless Creation (BM25)**
1.  **Tokenization**: The same chunks are cleaned and split into individual words (tokens).
2.  **Frequency Analysis**: We calculate:
    *   **TF (Term Frequency)**: How often a word appears in a chunk.
    *   **IDF (Inverse Document Frequency)**: How unique a word is across the entire legal library.
3.  **Persistence**: This lookup table is serialized into a `.pkl` file (Pickle format), allowing for instant keyword scoring without any neural network overhead.

---

## 9. How Citation & Page Indexes are Created

Unlike standard RAG which only stores text, Lex Assist builds a **Structured Legal Registry** during ingestion. This is what enables precise citations and the "Reference Traversal" feature.

### **The Creation Workflow**

1.  **Structural Extraction (Regex Engine)**:
    As each document is chunked, the system runs a specialized **Legal Structure Extractor**. It uses optimized Regular Expressions (Regex) to scan for:
    *   **Acts**: e.g., `Consumer Protection Act, 2019`
    *   **Sections**: e.g., `Section 34`, `Section 2(1)(d)`
    *   **Clauses/Sub-sections**: e.g., `Clause (a)`, `Sub-section (2)`
    
2.  **Metadata Binding**:
    These extracted tags are permanently "bound" to the chunk. In the `chunks.json` registry, every chunk of text has a metadata header:
    ```json
    {
      "chunk_id": 42,
      "text": "...",
      "doc": "CPA_2019.pdf",
      "page": 12,
      "act": "Consumer Protection Act, 2019",
      "section": "34",
      "all_sections": ["34", "35"]
    }
    ```

3.  **The Page-Level Map**:
    The system creates a reverse index called the `page_index`. It maps every document page to the list of chunks it contains:
    `"CPA_2019.pdf::p12" → [42, 43, 44]`
    *   **Why?** This allows the system to instantly fetch an entire page of a document if the LLM cites a section that is only partially captured in a single chunk.

4.  **AI-Powered Document Profiling**:
    For every file, Gemini 1.5 Flash generates a "Document Profile" which includes a high-level summary, effective date, and primary jurisdiction. This is saved in `metadata.json` for each folder, helping the **Metadata Filter** (Step 3 of the pipeline) decide which folders to search.

---

## 10. Security & Persistence
*   **Bcrypt**: Passwords are never stored in plain text. We use `bcrypt` with a work factor of 12 for salted hashing.
*   **JWT**: Stateless authentication using JSON Web Tokens ensures secure API access.
*   **SQLite**: Used for relational data (users/chats) to ensure ACID compliance and easy portability.

