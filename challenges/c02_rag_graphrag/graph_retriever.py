from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import time

from challenges.c02_rag_graphrag.graph_store import GraphStore
from data_engine.vector_store import VectorStoreManager


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize(a)
    b_n = _normalize(b)
    return float(np.dot(a_n, b_n))


@dataclass
class RetrievalResult:
    node_ids: List[str]
    edges: List[Dict]
    seed_ids: List[str]
    used_edge_weights: List[float]
    graph_coverage: float
    path_coherence: float


class GraphRetriever:
    """
    GraphRAG retrieval:
      1) Seed retrieval via vector DB (top-k)
      2) Graph expansion via weighted traversal
      3) Subgraph extraction (node-induced subgraph + edge pruning)
    """

    def __init__(self, graph_store: GraphStore, vector_store: VectorStoreManager):
        self.store = graph_store
        self.vstore = vector_store

        # Pre-normalize embeddings for fast cosine computations.
        self._emb_norm: Dict[str, np.ndarray] = {}
        for node_id, node in self.store.nodes.items():
            emb = np.asarray(node["embedding"], dtype=np.float32)
            self._emb_norm[node_id] = _normalize(emb)

    def _cosine_sim_from_chroma_distance(self, distance: float) -> float:
        return float(1.0 - distance)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        depth: int = 2,
        alpha: float = 0.75,
        beta: float = 0.25,
        max_nodes: int = 40,
        edge_prune_threshold: float = 0.60,
        global_relevance_k: int = 20,
    ) -> Dict:
        """
        Returns a dict containing:
          - nodes: list[dict] (id,text,metadata)
          - edges: list[edge dict]
          - metadata: retrieval diagnostics + new graph metrics
        """
        t0 = time.time()
        query_emb = np.asarray(self.vstore.model.encode([query])[0], dtype=np.float32)
        query_emb_n = _normalize(query_emb)

        # 1) Seed retrieval (vector DB).
        t_seed0 = time.time()
        seed_res = self.vstore.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["distances"],
        )
        seed_latency_ms = (time.time() - t_seed0) * 1000.0
        seed_ids = (seed_res.get("ids") or [[]])[0]
        seed_distances = (seed_res.get("distances") or [[]])[0]
        seed_sims = {
            node_id: self._cosine_sim_from_chroma_distance(dist)
            for node_id, dist in zip(seed_ids, seed_distances)
            if node_id in self.store.nodes
        }
        seed_ids = [sid for sid in seed_ids if sid in self.store.nodes]

        # 2) Expand via weighted best-first traversal (bounded by depth and max_nodes).
        # We combine:
        #   score(node) = alpha * sim(query, node) + beta * cumulative_edge_weight
        t_graph0 = time.time()
        best_score: Dict[str, float] = {}
        best_edge_sum: Dict[str, float] = {}
        parent: Dict[str, Tuple[str, float]] = {}  # child -> (parent_id, edge_weight)
        best_depth: Dict[str, int] = {}

        # Use a max-heap via negative scores.
        heap: List[Tuple[float, str]] = []

        def push(node_id: str, score: float) -> None:
            heapq.heappush(heap, (-score, node_id))

        for sid in seed_ids:
            sim_q = seed_sims.get(sid, cosine_sim(query_emb_n, self._emb_norm[sid]))
            best_edge_sum[sid] = 0.0
            best_depth[sid] = 0
            s = alpha * sim_q
            best_score[sid] = s
            push(sid, s)

        visited: Set[str] = set(seed_ids)
        used_edge_weights: List[float] = []

        while heap and len(visited) < max_nodes:
            neg_s, cur = heapq.heappop(heap)
            cur_score = -neg_s
            cur_depth = best_depth.get(cur, 0)
            if cur_depth >= depth:
                continue

            neighs = self.store.neighbors(cur)
            neighs_sorted = sorted(neighs, key=lambda x: x[1], reverse=True)

            for neigh_id, w, _t in neighs_sorted:
                if neigh_id not in self.store.nodes:
                    continue

                new_depth = cur_depth + 1
                new_edge_sum = best_edge_sum.get(cur, 0.0) + float(w)
                sim_q = cosine_sim(query_emb_n, self._emb_norm[neigh_id])
                score = alpha * sim_q + beta * new_edge_sum

                prev = best_score.get(neigh_id)
                if prev is None or score > prev:
                    best_score[neigh_id] = score
                    best_edge_sum[neigh_id] = new_edge_sum
                    best_depth[neigh_id] = new_depth
                    parent[neigh_id] = (cur, float(w))
                    push(neigh_id, score)
                    visited.add(neigh_id)
        graph_latency_ms = (time.time() - t_graph0) * 1000.0

        # Select top nodes by score (keep seeds even if they scored poorly).
        ranked = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
        selected_ids = [nid for nid, _s in ranked[:max_nodes]]
        selected_set = set(selected_ids) | set(seed_ids)
        selected_ids = list(selected_set)

        # Compute used path weights from parent pointers (only for nodes we keep).
        for nid in selected_ids:
            if nid in seed_ids:
                continue
            if nid in parent:
                used_edge_weights.append(parent[nid][1])

        # 3) Subgraph extraction + prune weak edges.
        edges: List[Dict] = []
        seen_pairs: Set[Tuple[str, str, str]] = set()
        for src in selected_ids:
            for dst, w, t in self.store.neighbors(src):
                if dst not in selected_set:
                    continue
                if float(w) < edge_prune_threshold:
                    continue
                a, b = (src, dst) if src < dst else (dst, src)
                key = (a, b, t)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "weight": float(w),
                        "type": t,
                    }
                )

        # 4) Graph metrics.
        # 4.1 Coverage: treat "top global relevance_k results" as a relevance proxy.
        cov = 0.0
        try:
            rel_res = self.vstore.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=global_relevance_k,
                include=[],
            )
            relevant_ids = set((rel_res.get("ids") or [[]])[0])
            if relevant_ids:
                cov = len(selected_set.intersection(relevant_ids)) / len(relevant_ids)
        except Exception:
            cov = 0.0

        # 4.2 Path coherence: average edge weight in the used traversal steps.
        if used_edge_weights:
            path_coh = float(sum(used_edge_weights) / len(used_edge_weights))
        else:
            path_coh = 0.0

        nodes = [
            {
                "id": nid,
                "text": self.store.nodes[nid]["text"],
                "metadata": self.store.nodes[nid].get("metadata", {}) or {},
            }
            for nid in selected_ids
            if nid in self.store.nodes
        ]

        return {
            "seed_ids": seed_ids,
            "node_ids": selected_ids,
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "seed_latency_ms": seed_latency_ms,
                "graph_latency_ms": graph_latency_ms,
                "retrieval_latency_ms": (time.time() - t0) * 1000.0,
                "traversal_depth": depth,
                "graph_nodes_count": len(nodes),
                "graph_edges_count": len(edges),
                "graph_coverage": cov,
                "path_coherence": path_coh,
                "edge_prune_threshold": edge_prune_threshold,
                "alpha": alpha,
                "beta": beta,
            },
        }
