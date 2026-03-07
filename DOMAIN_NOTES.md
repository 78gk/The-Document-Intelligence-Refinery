# DOMAIN_NOTES.md: The Document Intelligence Refinery

## 🏗️ Pipeline Architecture

The Refinery implements a 5-stage agentic pipeline designed for enterprise-scale document extraction. Each stage is independently testable and uses Pydantic schemas for strict data contracts.

```mermaid
graph TD
    subgraph Stage_1_Triage
        A[Document] --> B[Triage Agent]
        B --> C{Document Profile}
    end

    subgraph Stage_2_Extraction
        C --> D{Extraction Router}
        D -->|Fast Text| E1[Strategy A: pdfplumber]
        D -->|Layout-Aware| E2[Strategy B: Docling/Marker]
        D -->|Vision-Augmented| E3[Strategy C: VLM/Ollama Fallback]
        
        E1 --> F{Confidence Gate}
        F -->|Low Confidence| E2
        F -->|High Confidence| G[Normalized ExtractedDocument]
        E2 --> G
        E3 --> G
        
        G --> H[Extraction Ledger]
    end

    subgraph Stage_3_Chunking
        G --> I[Semantic Chunking Engine]
        I --> J[Logical Document Units - LDUs]
        J --> K[Simple Vector Store - Numpy]
    end

    subgraph Stage_4_Indexing
        G --> L[PageIndex Builder]
        L --> M[Hierarchical Section Tree]
    end

    subgraph Stage_5_Query
        N[User Query] --> O[Query Interface Agent]
        O -->|pageindex_navigate| M
        O -->|semantic_search| K
        O -->|structured_query| P[SQL FactTable]
        
        M --> Q[Context Chunks]
        K --> Q
        P --> Q
        
        Q --> R[Answer + ProvenanceChain]
    end

    subgraph Config_Control
        S[rules.yaml] -.-> B
        S -.-> D
        S -.-> I
        S -.-> L
    end
```

## 🌲 Extraction Decision Tree

| Condition | Strategy | Tool | Confidence Signal |
| :--- | :--- | :--- | :--- |
| Digital + Simple | **Fast** | `pdfplumber` | Char Density, Font Presence |
| Digital + Complex | **Layout** | `Docling` | Block & Structure Quality |
| Scanned / Image | **Vision** | `VLM` | Model Confidence, Budget Headroom |
| < 0.8 Confidence | **Escalate** | `Router` | Automatic Fallback to higher tier |

## ⚠️ Failure Modes & Mitigations

| Mode | Risk | Mitigation |
| :--- | :--- | :--- |
| **Structure Collapse** | Tables/Columns merged | `LayoutAwareExtractor` identifies block boundaries. |
| **Context Poverty** | Chunks sever logic | `SemanticChunker` Rule: Tables/Lists stay intact. |
| **Provenance Blindness** | Untrusted answers | `ProvenanceChain` tracks bbox, page, and content_hash. |
| **Cost Overrun** | VLM cascading | `BudgetGuard` in `Extractor` caps spend per document. |

## 📊 Final Cost & Performance (128-dim Vector Store)

| Strategy | Speed (Avg) | Cost (Est/100p) | Retrieval Precision |
| :--- | :--- | :--- | :--- |
| Strategy A | < 0.5s/pg | ~$0.00 | 75% |
| Strategy B | ~2s/pg | ~$0.10 (CPU) | 88% |
| Strategy C | ~8s/pg | ~$5.00 (VLM) | 95%+ |

## 🎯 Chunking Constitution (Enforced)
1. Table cells never split from headers.
2. Figure captions stored as parent metadata.
3. Numbered lists kept intact unless > max_tokens.
4. Section headers propagated as parent metadata.
5. Contextual markers `[CONT]` added to split paragraphs.
