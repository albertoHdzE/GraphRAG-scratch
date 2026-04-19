import os
from dotenv import load_dotenv

# Load credentials
load_dotenv()

import time
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import uuid
from langfuse.openai import OpenAI # Use Langfuse wrapper
from sklearn.cluster import KMeans
from langfuse import Langfuse
from data_engine.vector_store import VectorStoreManager
from challenges.c02_rag_graphrag.graph_builder import GraphBuilder
from challenges.c02_rag_graphrag.graph_retriever import GraphRetriever

LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")


def _build_fallback_trace_url(trace_id: str) -> str:
    if not trace_id:
        return LANGFUSE_BASE_URL
    return f"{LANGFUSE_BASE_URL}/traces/{trace_id}"


def _safe_get_trace_url(trace_id: str) -> str:
    if not trace_id:
        return LANGFUSE_BASE_URL
    if not langfuse:
        return _build_fallback_trace_url(trace_id)

    try:
        trace_url = langfuse.get_trace_url(trace_id=trace_id)
        if isinstance(trace_url, str) and trace_url.startswith("http"):
            return trace_url
    except Exception as e:
        print(f"Trace URL generation error ({trace_id}): {e}")

    return _build_fallback_trace_url(trace_id)


# Initialize Langfuse client for scores and URL retrieval
try:
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse = Langfuse()
    else:
        print("Langfuse credentials missing. Trace URLs will use fallback format.")
        langfuse = None
except Exception as e:
    print(f"Langfuse init error: {e}")
    langfuse = None

# Initialize OpenAI-compatible local client (LM Studio) via Langfuse wrapper
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class DualPipeline:
    def __init__(self, vector_store: VectorStoreManager):
        self.vstore = vector_store
        self.model_name = "google/gemma-3-4b"
        self._graph_store = None
        self._graph_retriever = None

    def _safe_chat_completion(self, *, prompt: str, trace_id: str, name: str) -> str:
        """
        Calls the local OpenAI-compatible server (LM Studio) via Langfuse wrapper.
        If the local server is unavailable, returns a deterministic fallback so the
        UI and comparison pipeline still run end-to-end.
        """
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                trace_id=trace_id,
                name=name,
            )
            return response.choices[0].message.content
        except Exception as e:
            return (
                "LLM backend unavailable (could not reach http://localhost:1234).\n\n"
                f"Fallback output generated without LLM.\n"
                f"Error: {type(e).__name__}: {e}\n\n"
                "Context excerpt:\n"
                + prompt[:2000]
            )

    def _ensure_graph_ready(self):
        """
        Build the explicit chunk graph once, lazily.
        """
        if self._graph_store is not None and self._graph_retriever is not None:
            return

        builder = GraphBuilder(
            self.vstore,
            semantic_similarity_threshold=0.75,
            semantic_top_k=12,
        )
        self._graph_store = builder.build()
        self._graph_retriever = GraphRetriever(self._graph_store, self.vstore)

    def naive_rag_flow(self, query: str, chunks: List[str]):
        """
        Classic RAG: Retrieval -> Concatenation -> Summarization
        """
        start_time = time.time()
        trace_id = uuid.uuid4().hex
        
        # Concatenate all chunks
        context = "\n\n".join(chunks)
        
        prompt = f"""
        Document Context:
        {context}

        Question: {query}
        
        Task: Synthesize a comprehensive summary based on the provided context. 
        Ensure zero icons or emojis are used in your response.
        """
        
        # Automatic tracing via Langfuse OpenAI wrapper
        summary = self._safe_chat_completion(
            prompt=prompt,
            trace_id=trace_id,
            name="Naive RAG Synthesis",
        )
        latency = (time.time() - start_time) * 1000
        
        trace_url = _safe_get_trace_url(trace_id)
        print(f"[OBSERVABILITY] Path A (Naive) Trace ID: {trace_id}")
        print(f"[OBSERVABILITY] Path A (Naive) URL: {trace_url}")
        
        metadata = {
            "latency": latency,
            "path": "naive_rag",
            "chunk_count": len(chunks)
        }
        
        return {
            "summary": summary, 
            "latency": latency, 
            "trace_id": trace_id,
            "trace_url": trace_url,
            "metadata": metadata
        }

    def structured_rau_flow(self, query: str, chunks: List[str]):
        """
        Structured RAU: Retrieval -> Similarity Matrix -> Clustering -> Synth
        """
        start_time = time.time()
        trace_id = uuid.uuid4().hex
        
        # 1. Similarity Calculation
        embeddings = self.vstore.model.encode(chunks)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # 2. Clustering
        n_clusters = min(3, len(chunks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # 3. Cluster Summarization
        cluster_summaries = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_text = "\n\n".join([chunks[idx] for idx in cluster_indices])
            
            c_prompt = f"Summarize the core theme of this specific legal cluster:\n\n{cluster_text}\n\nNo icons."

            c_summary = self._safe_chat_completion(
                prompt=c_prompt,
                trace_id=trace_id,
                name=f"Cluster {i} Theme Analysis",
            )
            cluster_summaries.append(c_summary)
            
        # 4. Final Global Synthesis
        final_prompt = f"""
        Synthesized Cluster Themes:
        {chr(10).join(cluster_summaries)}

        Question: {query}
        
        Task: Create a final structured summary based on these cluster themes.
        Ensure zero icons or emojis are used in your response.
        """
        
        final_summary = self._safe_chat_completion(
            prompt=final_prompt,
            trace_id=trace_id,
            name="RAU Final Global Synthesis",
        )
        latency = (time.time() - start_time) * 1000
        
        trace_url = _safe_get_trace_url(trace_id)
        print(f"[OBSERVABILITY] Path B (RAU) Trace ID: {trace_id}")
        print(f"[OBSERVABILITY] Path B (RAU) URL: {trace_url}")

        metadata = {
            "latency": latency,
            "path": "structured_rau",
            "n_clusters": n_clusters,
            "labels": labels.tolist()
        }
        
        return {
            "summary": final_summary, 
            "latency": latency, 
            "trace_id": trace_id,
            "trace_url": trace_url,
            "labels": labels.tolist(),
            "embeddings": embeddings.tolist(),
            "similarity_matrix": similarity_matrix.tolist(),
            "cluster_summaries": cluster_summaries,
            "metadata": metadata
        }

    def graphrag_flow(self, query: str, *, top_k: int = 10, depth: int = 2):
        """
        GraphRAG:
          1) Retrieve top-k seed chunks using vector DB
          2) Expand via graph traversal (weighted)
          3) Extract pruned subgraph
          4) Synthesize answer with relationship-aware prompt
        """
        self._ensure_graph_ready()

        start_time = time.time()
        trace_id = uuid.uuid4().hex

        retrieval = self._graph_retriever.retrieve(
            query,
            top_k=top_k,
            depth=depth,
            alpha=0.75,
            beta=0.25,
            max_nodes=40,
            edge_prune_threshold=0.60,
        )

        nodes = retrieval["nodes"]
        edges = retrieval["edges"]

        # Create stable, human-readable node labels for the prompt.
        label_map: Dict[str, str] = {n["id"]: f"Node {i+1}" for i, n in enumerate(nodes)}

        node_lines: List[str] = []
        for n in nodes:
            md = n.get("metadata", {}) or {}
            src = md.get("source", "unknown_source")
            idx = md.get("chunk_index", "NA")
            node_lines.append(
                f"[{label_map[n['id']]}] source={src} chunk_index={idx}\n{n['text']}"
            )

        edge_lines: List[str] = []
        for e in sorted(edges, key=lambda x: x["weight"], reverse=True):
            src_lbl = label_map.get(e["source"], e["source"])
            dst_lbl = label_map.get(e["target"], e["target"])
            edge_lines.append(
                f"{src_lbl} -> {dst_lbl} (type={e['type']}, weight={e['weight']:.2f})"
            )

        graph_context = (
            "Graph Context (nodes and explicit relationships):\n\n"
            "NODES:\n"
            + "\n\n".join(node_lines[:25])
            + "\n\nEDGES:\n"
            + "\n".join(edge_lines[:80])
        )

        prompt = f"""
        You are answering a legal question using an explicit graph of contract chunks.

        {graph_context}

        Question: {query}

        Task:
        1) Identify the key clauses relevant to the question.
        2) Use the EDGES to explain how concepts connect across nodes (multi-hop reasoning).
        3) Provide a concise, structured answer that explicitly references at least 2-3 relationships (e.g., Node X -> Node Y).

        Output constraints:
        - Plain text only.
        - No icons or emojis.
        """

        summary = self._safe_chat_completion(
            prompt=prompt,
            trace_id=trace_id,
            name="GraphRAG Graph-Based Synthesis",
        )
        latency = (time.time() - start_time) * 1000

        trace_url = _safe_get_trace_url(trace_id)
        print(f"[OBSERVABILITY] Path C (GraphRAG) Trace ID: {trace_id}")
        print(f"[OBSERVABILITY] Path C (GraphRAG) URL: {trace_url}")

        metadata = {
            "latency": latency,
            "path": "graphrag",
            **(retrieval.get("metadata") or {}),
        }

        return {
            "summary": summary,
            "latency": latency,
            "trace_id": trace_id,
            "trace_url": trace_url,
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "seed_ids": retrieval.get("seed_ids", []),
            },
            "metadata": metadata,
        }

    def execute_comparison(self, query: str):
        """
        Runs RAG, RAU, and GraphRAG in parallel.
        """
        # Fetch Top-10 Chunks (Shared Retrieval)
        res = self.vstore.query(query, top_k=10)
        chunks = res["documents"][0]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_rag = executor.submit(self.naive_rag_flow, query, chunks)
            future_rau = executor.submit(self.structured_rau_flow, query, chunks)
            future_graphrag = executor.submit(self.graphrag_flow, query)
            
            rag_result = future_rag.result()
            rau_result = future_rau.result()
            graphrag_result = future_graphrag.result()

        if langfuse:
            try:
                langfuse.flush()
            except Exception as e:
                print(f"Langfuse flush error: {e}")
            
        return {
            "query": query,
            "chunks": chunks,
            "rag": rag_result,
            "rau": rau_result,
            "graphrag": graphrag_result,
        }

    def push_scores_to_langfuse(self, trace_id: str, scores: Dict[str, float]):
        """
        Sends numerical measurements to Langfuse Cloud.
        """
        if not langfuse: return
        
        for name, value in scores.items():
            try:
                langfuse.score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    data_type="NUMERIC"
                )
            except Exception as e:
                print(f"Error pushing score {name}: {e}")
