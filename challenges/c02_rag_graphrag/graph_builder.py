from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from challenges.c02_rag_graphrag.graph_store import GraphStore
from data_engine.vector_store import VectorStoreManager


class GraphBuilder:
    """
    Builds an explicit chunk graph G = (V, E) where:
      - V: chunks (id, text, embedding, metadata)
      - E: sequential + semantic similarity edges

    Constraints:
      - No high-level graph algorithms / frameworks
      - No GraphRAG libraries
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        *,
        semantic_similarity_threshold: float = 0.75,
        semantic_top_k: int = 12,
    ):
        self.vstore = vector_store
        self.semantic_similarity_threshold = float(semantic_similarity_threshold)
        self.semantic_top_k = int(semantic_top_k)

    def _cosine_sim_from_chroma_distance(self, distance: float) -> float:
        """
        Chroma cosine "distance" is typically (1 - cosine_similarity).
        """
        return float(1.0 - distance)

    def build(self, *, limit: Optional[int] = None) -> GraphStore:
        """
        Builds a full graph from the underlying vector store collection.

        limit: for debugging / fast iteration; if provided, truncates nodes.
        """
        store = GraphStore()

        # 1) Load all nodes (chunks) from the persistent store.
        raw = self.vstore.collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        ids: List[str] = raw.get("ids", [])
        docs: List[str] = raw.get("documents", [])
        embs: List[List[float]] = raw.get("embeddings", [])
        metas: List[Dict] = raw.get("metadatas", [])

        if limit is not None:
            ids = ids[:limit]
            docs = docs[:limit]
            embs = embs[:limit]
            metas = metas[:limit]

        # 2) Register nodes.
        for node_id, text, emb, meta in zip(ids, docs, embs, metas):
            store.add_node(
                {
                    "id": node_id,
                    "text": text,
                    "embedding": emb,
                    "metadata": meta or {},
                }
            )

        # 3) Sequential edges (prev <-> next), already encoded in metadata by the processor.
        for node_id in ids:
            node = store.get_node(node_id)
            if not node:
                continue
            md = node.get("metadata", {}) or {}
            prev_id = md.get("prev_id") or ""
            next_id = md.get("next_id") or ""

            if prev_id and prev_id in store.nodes:
                store.add_edge(node_id, prev_id, 1.0, "sequential", bidirectional=True)
            if next_id and next_id in store.nodes:
                store.add_edge(node_id, next_id, 1.0, "sequential", bidirectional=True)

        # 4) Semantic edges (cosine similarity > threshold).
        # Avoid O(n^2) by using the vector DB as an approximate neighbor generator,
        # then compute similarity from returned cosine distances.
        seen_pairs: Set[Tuple[str, str]] = set()

        for node_id in ids:
            node = store.get_node(node_id)
            if not node:
                continue

            emb = node["embedding"]

            # Query neighbors for each chunk embedding.
            res = self.vstore.collection.query(
                query_embeddings=[emb],
                n_results=self.semantic_top_k + 1,
                include=["distances"],
            )

            neigh_ids = (res.get("ids") or [[]])[0]
            neigh_distances = (res.get("distances") or [[]])[0]

            for neigh_id, dist in zip(neigh_ids, neigh_distances):
                if neigh_id == node_id:
                    continue
                if neigh_id not in store.nodes:
                    continue

                sim = self._cosine_sim_from_chroma_distance(dist)
                if sim < self.semantic_similarity_threshold:
                    continue

                a, b = (node_id, neigh_id) if node_id < neigh_id else (neigh_id, node_id)
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))

                store.add_edge(node_id, neigh_id, sim, "semantic", bidirectional=True)

        return store
