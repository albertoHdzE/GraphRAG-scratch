# SPECIFICATION: Challenge 1 - Structured Clustering

## Project Overview
This project implements a Structured Clustering pipeline (RAU - Retrieval Augmented Understanding) to process legal documents from the CUAD dataset. The goal is to compare the efficacy of structured retrieval (clustering) against naive retrieval methods, using local model execution and professional observability.

## Functional Requirements

### Level 0: Shared Data Engine
- Source Data: Local CUAD dataset subset (10-20 txt contracts).
- Extraction: Recursive character-based chunking with overlap.
- Vector Store: ChromaDB for persistence and metadata management.
- Batch Processing: Implement batch embeddings (5-10 chunks) for efficiency.
- Schema: Must store Document ID, Chunk ID, Text content, and Source path.

### Level 1: Dual Processing Pipeline
- Workflow A (Naive RAG):
  - Retrieve top-10 chunks.
  - Summarize with basic prompt.
  - Metrics: Latency, Token Usage.
- Workflow B (Structured RAU):
  - Step 1 (Retrieval): Shared with Path A.
  - Step 2 (Analysis): Similarity Matrix & Clustering execution.
  - Step 3 (Traceable Summary): Cluster-level themes + final synthesis.
- Comparative Analysis (Metrology):
  - Information Density: Entity-to-token ratio.
  - Cluster Purity: Silhouette score and intra-cluster cohesion.
  - Faithfulness: Automated LLM verification (1-10) of claim accuracy vs source context.
  - Total Token Usage: Comparison of prompt/completion overhead.

### UI / Frontend
- Side-by-Side Comparison: Dual columns for output text.
- Traceability Tab: Detailed breakdown of the RAU reasoning chain.
- KPI Dashboard: Grouped cards for Speed, Quality (Density/Faithfulness), and Efficiency (Tokens).
- Monitoring Link: Direct access to Langfuse Cloud traces/scores for each run.

## Non-Functional Requirements
- Local Execution: All LLM calls must hit localhost:1234.
- Observability: Complete trace of LLM calls in LangFuse.
- Performance: Generate an automated performance report after execution.
- Metrics:
  - Information Density: Ratio of unique retrieved info preserved in summary.
  - Cluster Purity: LLM-assessed thematic consistency of the cluster.
- Constraints: Zero icons or emojis anywhere in the codebase or reports.

## System Architecture
- Data Storage: FAISS/ChromaDB for vector embeddings.
- LLM Provider: LM Studio (Gemma-3-4b profile).
- Observability: LangFuse (Python SDK).
- Frontend: Streamlit.

## Definition of Done
1. Functional Level 0 data loader and chunker.
2. Functional Level 1 RAU pipeline with clustering logic.
3. LangFuse successfully tracking traces and latencies.
4. Streamlit dashboard displaying visual comparisons and performance metrics.
5. performance_report.md updated with latest execution results.
