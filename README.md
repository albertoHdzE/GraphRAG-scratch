# GraphRAG From Scratch (Scientific Baseline)

## Abstract
This repository is a deliberately minimal, *mathematically explicit* implementation of retrieval-augmented generation (RAG), retrieval-augmented understanding (RAU via clustering), and **GraphRAG** (graph construction + traversal + relationship-aware synthesis). The objective is not to ship a “feature-complete” system, but to **demystify GraphRAG by reducing it to first principles**: vector spaces, similarity thresholds, adjacency lists, traversal scoring, and measurable retrieval diagnostics.

If you want GraphRAG explained as an algorithm rather than a product, this codebase is intended to be a reproducible reference point.

## Why this project exists (scientific bias)
Many GraphRAG discussions conflate:
1) the *graph itself* (a structure over chunks),  
2) the *retrieval policy* (how we traverse/select nodes), and  
3) the *LLM synthesis prompt* (how relationships are communicated).

Here, each component is implemented explicitly and kept inspectable. This makes it easier to reason about:
- when GraphRAG improves multi-hop retrieval,
- when it fails (graph noise, threshold sensitivity, hub nodes),
- and what the metrics actually measure.

## Repository map
- `data_engine/`
  - ingestion, chunking, and persistence into ChromaDB
  - embedding model: Sentence-Transformers (local)
- `challenges/`
  - `c01_clustering_rau/`: baseline RAG vs RAU (clustering)
  - `c02_rag_graphrag/`: **GraphRAG from scratch** (graph + traversal + synthesis)
- `ui/dashboard.py`: Streamlit evaluation dashboard (RAG vs RAU vs GraphRAG)
- `launch_app.sh`: launcher (uses `.venv`, reads `.env`)

## System overview
All three flows share the same embedding space and persistent vector store:

1. **RAG (flat retrieval)**  
   Retrieve top-*k* chunks by vector similarity; concatenate; summarize.

2. **RAU (clustering baseline)**  
   Retrieve top-*k* chunks; embed; compute similarity matrix; cluster (KMeans); summarize clusters; then global synthesis.

3. **GraphRAG (explicit graph + traversal)**  
   Build an explicit graph over chunks, then do seed retrieval + traversal expansion to form a query-specific subgraph, and finally synthesize using **relationships**, not just text.

## GraphRAG: formalization
We construct a graph:

**G = (V, E)**  
- **V**: chunk nodes  
- **E**: relationships (edges) between chunks

### Node definition
Each chunk becomes a node:
```python
node = {
  "id": chunk_id,
  "text": chunk_text,
  "embedding": vector,
  "metadata": {...}
}
```

### Edge definition
We add two edge families:

1) **Sequential edges** (document order)  
If chunk *i* precedes chunk *j*:
```text
i <-> j  with weight = 1.0
```

2) **Semantic similarity edges** (vector space proximity)  
Cosine similarity:
```text
sim(i, j) = (v_i · v_j) / (||v_i|| ||v_j||)
```
Add an edge if:
```text
sim(i, j) > 0.75
```
Store:
```python
edge = {
  "source": id_i,
  "target": id_j,
  "weight": similarity,
  "type": "semantic"
}
```

### Graph storage (no graph frameworks)
We store the graph as:
- node registry: `node_id -> node`
- adjacency list: `node_id -> [(neighbor_id, weight, type), ...]`

This is intentional: it keeps complexity visible and avoids “black box” graph libraries.

## Graph retrieval (seed + traversal)
GraphRAG retrieval is a two-stage policy:

### Stage A: seed retrieval
Use the vector DB to retrieve top-*k* seed nodes:
```text
Seeds = TopK(query, V)
```

### Stage B: graph expansion
Traverse from seeds for a fixed depth (BFS-like but weight-prioritized). Nodes are scored by:
```text
score(node) = α * sim(query, node) + β * edge_weight_sum
```
where `edge_weight_sum` is the cumulative edge weight along the best known path from any seed.

This creates a *query-conditioned subgraph*, not a static context window.

### Subgraph extraction
From the expanded set:
- keep only node-induced edges
- prune weak edges below a threshold (default `0.60`)

## Graph-based synthesis (relationship-aware prompting)
GraphRAG synthesis is not “just more chunks”. The prompt explicitly includes:
- a list of nodes (with provenance metadata)
- a list of edges such as:
```text
Node 3 -> Node 7 (type=semantic, weight=0.84)
```

This forces the LLM to condition on *structure* (multi-hop paths), not only lexical overlap.

## Metrics (beyond token counts)
In addition to latency/tokens/faithfulness-style evaluations, this repo adds graph-specific diagnostics:

### graph_coverage
```text
graph_coverage = |Retrieved ∩ Relevant| / |Relevant|
```
Operationally, “Relevant” is approximated as the top-*K* global vector hits for the query (a proxy set). This is not perfect ground truth, but it is measurable and stable.

### path_coherence
Average weight of edges actually used during traversal:
```text
path_coherence = mean(edge_weight_used_in_expansion)
```
Interpretation: higher coherence indicates traversal steps followed stronger relationships (less graph noise).

## How to run
### Prerequisites
- A local OpenAI-compatible server (LM Studio or equivalent) is expected at:
```text
http://localhost:1234/v1
```
- `.env` contains Langfuse credentials (optional but recommended).

### Launch
From repo root:
```bash
bash launch_app.sh
```
Then open:
```text
http://localhost:8501/
```

## Technology stack (intentionally pragmatic)
- **Vector DB**: ChromaDB (persistent)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **UI**: Streamlit
- **RAU baseline**: KMeans clustering + similarity matrix inspection
- **Observability**: Langfuse tracing (OpenAI wrapper)
- **LLM interface**: OpenAI-compatible HTTP API (LM Studio recommended)

## Suggested experiments (to expose strengths and weaknesses)
1) **Multi-hop query sensitivity**  
Increase traversal depth and observe:
- graph_coverage changes
- path_coherence changes
- whether answers start referencing meaningful cross-clause dependencies

2) **Semantic threshold sweep**  
Vary `sim > 0.75` and measure:
- edge density growth
- emergence of hub nodes (over-connection)
- degradation of path_coherence (graph noise)

3) **Edge pruning threshold sweep**  
Vary prune threshold (e.g. 0.50..0.80) and inspect:
- subgraph size
- whether relationship lists become too sparse to support reasoning

4) **Compare “more chunks” vs “structured context”**  
Hold token budget constant, compare:
- flat RAG with more top-k chunks
- GraphRAG with fewer nodes but explicit relationships

## Known limitations (by design)
- This is not a knowledge graph of entities; it is a **chunk graph**.
- “Relevance” used in graph_coverage is a proxy (top vector hits), not labeled ground truth.
- Graph quality depends on chunking strategy (size/overlap) and embedding model geometry.
- A strong graph can still yield poor synthesis if the LLM ignores edges; hence explicit edge prompting.

## Where the GraphRAG code lives
All GraphRAG logic is in:
- `challenges/c02_rag_graphrag/graph_builder.py`
- `challenges/c02_rag_graphrag/graph_store.py`
- `challenges/c02_rag_graphrag/graph_retriever.py`
- `challenges/c02_rag_graphrag/pipeline.py`

The UI comparison is in:
- `ui/dashboard.py`

