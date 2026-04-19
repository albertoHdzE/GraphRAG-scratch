# GraphRAG From Scratch (Scientific Baseline)

## Abstract

This repository provides a clean, minimal, and mathematically explicit implementation of classic RAG, clustering-based retrieval-augmented understanding (RAU), and **GraphRAG**. 

The goal is not to build another full-featured framework, but to break GraphRAG down to its core components: vector embeddings, graph construction, traversal logic, and relationship-aware synthesis. By keeping everything transparent and inspectable, this codebase serves as a reproducible reference for understanding how and why GraphRAG works (or doesn't) compared to simpler approaches.

## Why this project exists

Most discussions around GraphRAG blur together three distinct pieces:
1. The graph structure itself (nodes and edges over document chunks),
2. The retrieval strategy (how we select and expand relevant nodes),
3. The final LLM synthesis prompt (how relationships are presented to the model).

Here, each piece is implemented and exposed explicitly. This separation makes it much easier to study:
- When and how graph-based retrieval improves multi-hop reasoning,
- Where it introduces noise or fails (hub nodes, weak connections, threshold sensitivity),
- And what the actual metrics are telling us.

## Repository Structure

- **`data_engine/`** — Document ingestion, chunking, embedding, and persistence (ChromaDB)
- **`challenges/`**
  - `c01_clustering_rau/` — Classic RAG vs. clustering-based RAU
  - `c02_rag_graphrag/` — Full GraphRAG implementation (graph + traversal + synthesis)
- **`ui/dashboard.py`** — Streamlit dashboard to compare all three approaches side-by-side
- **`launch_app.sh`** — Convenience launcher

## System Overview

All approaches share the same embedding space and vector store:

1. **Classic RAG**  
   Retrieve top-*k* chunks by cosine similarity, concatenate, and summarize.

2. **RAU (Clustering baseline)**  
   Retrieve top-*k* chunks, build a similarity matrix, cluster them (KMeans), summarize each cluster, then perform a final synthesis step.

3. **GraphRAG**  
   Build an explicit graph over chunks, perform seed retrieval + guided traversal to create a query-specific subgraph, then synthesize using both content *and* explicit relationships.

## GraphRAG Formalization

We model the collection as a graph **G = (V, E)** where:

- **V** — chunk nodes, each containing text, embedding vector, and metadata
- **E** — two types of edges

### Nodes
```python
node = {
    "id": chunk_id,
    "text": chunk_text,
    "embedding": vector,
    "metadata": {...}
}