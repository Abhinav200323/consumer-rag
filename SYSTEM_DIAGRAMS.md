# Lex Assist: System Architecture Diagrams

This document contains the visual representation of the Lex Assist platform's functionality and data flow.

---

## 1. Use Case Diagram
The Use Case Diagram illustrates the high-level functions available to different users of the Lex Assist system and how they interact with the core Agentic RAG engine.

```mermaid
useCaseDiagram
    actor User as "Consumer (User)"
    actor Lawyer as "Legal Expert (Lawyer)"
    actor Admin as "System Administrator"

    User --> (Ask Legal Question)
    User --> (Upload Evidence / OCR)
    User --> (View Legal Advice & Citations)
    User --> (Generate Legal Notice)
    User --> (Chat with Lawyer)
    
    Lawyer --> (Manage Customer Inquiries)
    Lawyer --> (Reply to User)

    Admin --> (Ingest Legal Documents)
    Admin --> (Manage Knowledge Base)
    Admin --> (Monitor System Performance)

    (Ask Legal Question) ..> (Agentic RAG Pipeline) : <<include>>
    (Agentic RAG Pipeline) ..> (Citation Verification) : <<include>>
    (Generate Legal Notice) ..> (Drafting Engine) : <<include>>
```

---

## 2. DFD Level 0: Context Diagram
The Context Diagram defines the boundary between the Lex Assist system and its environment, showing the high-level data exchange with external entities.

```mermaid
graph LR
    User((User))
    Lawyer((Lawyer))
    Admin((Admin))
    
    subgraph " "
        System{Lex Assist System}
    end

    User -- "Legal Queries, Evidence, Auth Credentials" --> System
    System -- "Legal Advice, Citations, Drafted Notices" --> User
    System -- "Session Status, Auth Tokens" --> User

    Lawyer -- "Inquiry Responses, Status Updates" --> System
    System -- "Customer Legal Inquiries" --> Lawyer

    Admin -- "Raw Legal Docs, System Configs" --> System
    System -- "System Logs & Performance Metrics" --> Admin
```

---

## 3. DFD Level 1: Process Decomposition
The Level 1 DFD breaks down the system into its primary functional modules and shows how data moves between processes and persistent storage.

```mermaid
graph TD
    %% Entities
    U((User))
    L((Lawyer))
    A((Admin))

    %% Processes
    subgraph "Functional Processes"
        P1[User Auth & Profile Management]
        P2[Intent Routing & Input Processing]
        P3[Agentic Retrieval Engine]
        P4[Reasoning & Citation Verification]
        P5[Drafting & Output Formatting]
        P6[Communication & Messaging Service]
        P7[Knowledge Ingestion & Indexing]
    end

    %% Data Stores
    subgraph "Data Storage"
        DB1[(SQLite: Users, Chats, Messages)]
        DB2[(FAISS: Dense Vector Index)]
        DB3[(BM25: Sparse Keyword Index)]
        DB4[(Local Storage: PDF Source & JSON Chunks)]
    end

    %% Data Flows
    U -- "Login / Query" --> P1
    P1 -- "User Data" --> DB1
    P1 -- "Verified Session & Query" --> P2
    
    P2 -- "OCR Text & Expanded Query" --> P3
    P3 -- "Metadata Filter Search" --> DB4
    P3 -- "Semantic Vector Search" --> DB2
    P3 -- "Lexical Keyword Search" --> DB3
    
    P3 -- "Retrieved Legal Chunks" --> P4
    P4 -- "Validated Legal Analysis" --> P5
    P5 -- "Final Advice & Citations" --> U

    U -- "Message" --> P6
    P6 -- "Message Data" --> DB1
    P6 -- "Forwarded Inquiry" --> L
    L -- "Response" --> P6
    
    A -- "Legal Document Upload" --> P7
    P7 -- "Generated Embeddings" --> DB2
    P7 -- "Extracted Keywords" --> DB3
    P7 -- "Structured Chunks & Metadata" --> DB4
```
