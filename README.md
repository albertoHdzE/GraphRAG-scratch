# GraphRAG From Scratch (Scientific Baseline)

## Thesis and research direction

This project is motivated by a concrete scientific concern: RAG systems are often evaluated primarily by the plausibility of their final answers, even though the final step is an LLM whose objective is to produce coherent language. A sufficiently capable model can frequently produce a coherent narrative that blends retrieved fragments into an answer-shaped statement, even when the retrieved evidence is incomplete, weakly connected, or partially irrelevant. Coherence is therefore not a reliable proxy for correctness.

From a complexity science and information-theoretic perspective, this raises operational questions that remain open in many RAG deployments:

- Evidence veracity: are the retrieved chunks true and correctly attributed to the underlying corpus, or are they artifacts of embedding similarity and chunking choices
- Structural validity: for GraphRAG, is the induced subgraph a faithful representation of the relevant causal or logical dependencies, or merely a convenient neighborhood in embedding space
- Completeness and coverage: is the graph and its edge policy sufficient to make the relevant pathways reachable, or does it systematically miss long-range dependencies and rare but high-impact clauses
- Stability under updates: when new chunks are added, do local insertion heuristics preserve global structure, or do they introduce shortcut edges and hub effects that change navigability and interpretability
- Alternatives and efficiency: are there faster or more robust ways to organize evidence than incremental graph growth, including periodic rebuilds or re-optimization of structure

The guiding thesis is not that RAG is invalid, but that the scientific object being evaluated is a coupled dynamical system: a retrieval policy over a representation (vector store and graph) plus a generator that can amplify weak signals into fluent conclusions. The correct evaluation target is therefore the evidence-selection process and its induced structure, not merely the final text.

We use natural systems as an analogy for what a rigorous objective could look like. In biology, informational success is not just local correctness but system-level integration: information is encoded, error-corrected, and redundantly distributed such that the organism remains functional under perturbations. DNA is both a compact code and a mechanism-compatible substrate; replication and repair enforce constraints that preserve identity over time. By contrast, many RAG pipelines rely on unconstrained text coherence at the final step and only weak structural constraints upstream. This motivates explicit diagnostics and ablations that quantify when a retrieval structure is integrated enough to support reliable reasoning, and when rebuilding or re-indexing is a safer default than incremental updates.

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
