# TECHNICAL DESIGN: Challenge 1 - Structured Clustering

## Data Flow Architecture

The system follows a tiered architecture designed for extensibility and graph-readiness.

### 1. Level 0: Data Foundation (data_engine)
- Component: processor.py
  - Logic: RecursiveCharacterTextSplitter from LangChain or custom logic.
  - Chunk Size: 1000 characters.
  - Overlap: 100 characters.
  - Metadata: {source: path, chunk_id: uuid, index: int}.
- Component: vector_store.py
  - Implementation: ChromaDB PersistentClient.
  - Storage: /data_engine/storage/chroma_db/.
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2.
  - Batching: Implement batch_size=10 for embeddings to optimize local processing time.
  - Graph-Readiness: Metadata fields for source, chunk_id, and sequential relations (prev_id, next_id).

### 2. Level 1: Dual Execution Pipeline (challenges/c01_clustering_rau)
- Component: pipeline.py
  - Input: top_k_chunks from Level 0.
  - Path A (Naive RAG):
    - Process: Concatenate all chunks -> Single Summarization Prompt.
    - Trace: langfuse_trace("naive_rag").
  - Path B (Structured RAU):
    - Process: Similarity Matrix -> K-Means (k=3) -> Cluster Summaries -> Final Synthesis.
    - Trace: langfuse_trace("structured_rau").
  - Comparison Logic:
    - Execute both paths for every query (parallel via threading).
    - Capture numerical metrics from each span.

### 3. Advanced Metrology and Traceability
- Metrics Implementation (comparison.py):
  - Silhouette Score: Use sklearn.metrics.silhouette_score on cluster labels.
  - Entity Preservation: Ratio of (Extracted Entities in Summary) / (Total Chunks Entities).
  - Faithfulness (LLM-as-a-Judge):
    - Evaluator: Gemma-3-4b.
    - Prompt: "Verify each claim in the summary against the source text. Score 1-10 based on factual grounding. No icons."
  - Token Consumption: Character count / 4 (standard RAG estimation) for both In/Out.
- Langfuse Monitoring (Observability):
  - Trace ID: Run timestamp + Path ID.
  - Scoring: Use langfuse.score(name, value, metadata, trace_id) for each path.
  - Traceability Data: Store cluster themes and centroids as part of the RAU trace metadata.

## UI / Dashboard Design (ui/dashboard.py)
- Layout:
  - Sidebar: Parameter configuration (K clusters, retrieval limits).
  - Main (Execution):
    - Metric Comparison Bar: Gauges comparing RAG vs RAU on Density, Speed, and Faithfulness.
    - Results Area: Two columns with st.markdown rendering.
    - Traceability Center (st.tabs):
      - Tab 1: Retrieval Set (Heatmap).
      - Tab 2: RAU Reasoner (Cluster Summaries & Silhouette Plot).
      - Tab 3: Detailed Metrology (DataTable comparison of concepts found).
    - Footer: Monitoring links to Langfuse Cloud.

## Component Interaction Diagram
1. User enters query in Streamlit.
2. Vector Store retrieves chunks.
3. Pipeline handles clustering and LLM calls.
4. LangFuse records spans.
5. Streamlit renders results.

## Zero Icons Policy
- All strings, logs, and comments must be alphanumeric only.
- No Unicode symbols, emojis, or graphic text markers.
