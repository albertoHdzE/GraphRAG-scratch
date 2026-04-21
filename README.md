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

## Benchmark data and storage

This repository uses two storage layers for the benchmark:

- Corpus (contracts): a small plaintext sample under `data_engine/contracts/` used as a lightweight benchmark for end-to-end runs.
- Vector database: ChromaDB persistence under `data_engine/storage/chroma_db/` created by ingesting the corpus and storing embeddings plus metadata.

The contracts are representative of the style used in the Contract Understanding Atticus Dataset (CUAD) corpus of SEC-sourced commercial agreements. CUAD v1 is a widely used academic benchmark for contract analysis.

- Dataset landing page: https://www.atticusprojectai.org/cuad
- Dataset code and instructions: https://github.com/TheAtticusProject/cuad

If you want to scale experiments beyond the small included sample, download CUAD and replace or extend `data_engine/contracts/`, then re-run ingestion to rebuild the vector store.

## TODO roadmap: databases and deployment (local and GCP)

### Storage backends

Vector indexes (this repo currently uses ChromaDB):

- [ ] PostgreSQL + pgvector (local, Cloud SQL, AlloyDB)
- [ ] Qdrant (local, managed options)
- [ ] Milvus (local, managed options)
- [ ] Weaviate (local, managed options)
- [ ] Elasticsearch or OpenSearch vector search (local, managed)
- [ ] Pinecone (managed)
- [ ] Vertex AI Vector Search (GCP)
- [ ] BigQuery vector search (GCP, analytics-first workloads)
- [ ] FAISS (local, single-node baseline for controlled experiments)

Graph stores (this repo currently uses an in-memory adjacency list in Python for maximal transparency):

- [ ] Neo4j (local, Aura, GCP deployment)
- [ ] Memgraph (local, GCP deployment)
- [ ] ArangoDB (graph plus document, local and managed)
- [ ] JanusGraph (for large graphs with external storage backends)
- [ ] TigerGraph (enterprise graph database)

### Hallucination and reliability strategies

Evidence and attribution:

- [ ] Enforce citations: require answer claims to quote short spans and reference node IDs and sources
- [ ] Abstention policy: return "insufficient evidence" when retrieval confidence is low
- [ ] Multi-retriever checks: compare vector-only vs graph-expanded evidence; flag contradictions

Prompt and context controls:

- [ ] Strict prompt budgeting for GraphRAG (node count, node length, edge count) to prevent context overflow
- [ ] Anti-echo detection and fallback summarization when models repeat the prompt context
- [ ] Query decomposition for multi-hop questions (sub-queries, then merge with explicit provenance)

Verification and evaluation:

- [ ] LLM-as-a-judge plus lightweight rule checks (numbers, dates, party names) against retrieved text
- [ ] Regression suite of "needle-in-haystack" and multi-hop queries with stored expected evidence
- [ ] Trace auditing with Langfuse: log retrieval sets, prompts, and scored diagnostics for every run

### Cross-platform parity (local and GCP)

- [ ] Configuration parity: same pipeline parameters via environment variables and config files
- [ ] Reproducible builds: Docker or equivalent, pinned dependencies, deterministic benchmark scripts
- [ ] Data lifecycle: separate raw corpora, derived chunks, embeddings, and graph artifacts by version
- [ ] Observability parity: structured logs, traces, and metrics (latency, coverage, coherence) in both environments

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
