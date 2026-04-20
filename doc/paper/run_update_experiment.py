import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from challenges.c02_rag_graphrag.graph_store import GraphStore
from data_engine.vector_store import VectorStoreManager


@dataclass(frozen=True)
class GraphStats:
    node_count: int
    undirected_edge_count: int
    avg_degree: float
    clustering_coeff: float
    lcc_fraction: float


def _cosine_sim_from_chroma_distance(distance: float) -> float:
    return float(1.0 - float(distance))


def _undirected_edge_set(store: GraphStore, *, min_weight: float = 0.0) -> Set[Tuple[str, str, str]]:
    edges: Set[Tuple[str, str, str]] = set()
    for src, neighs in store.graph.items():
        for dst, w, t in neighs:
            if float(w) < min_weight:
                continue
            a, b = (src, dst) if src < dst else (dst, src)
            edges.add((a, b, t))
    return edges


def _adjacency_undirected(store: GraphStore, *, min_weight: float = 0.0) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {nid: set() for nid in store.nodes.keys()}
    for src, neighs in store.graph.items():
        for dst, w, _t in neighs:
            if float(w) < min_weight:
                continue
            if src not in adj or dst not in adj:
                continue
            adj[src].add(dst)
            adj[dst].add(src)
    return adj


def _clustering_coefficient(adj: Dict[str, Set[str]]) -> float:
    vals: List[float] = []
    for v, neighs in adj.items():
        k = len(neighs)
        if k < 2:
            continue
        neigh_list = list(neighs)
        links = 0
        for i in range(k):
            a = neigh_list[i]
            na = adj.get(a, set())
            for j in range(i + 1, k):
                b = neigh_list[j]
                if b in na:
                    links += 1
        denom = k * (k - 1) / 2.0
        vals.append(links / denom if denom > 0 else 0.0)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _largest_component_fraction(adj: Dict[str, Set[str]]) -> float:
    seen: Set[str] = set()
    sizes: List[int] = []
    for start in adj.keys():
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for n in adj.get(cur, set()):
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        sizes.append(size)
    if not sizes:
        return 0.0
    return float(max(sizes) / max(1, len(adj)))


def compute_stats(store: GraphStore) -> GraphStats:
    node_count = store.node_count()
    edges = _undirected_edge_set(store)
    undirected_edge_count = len(edges)
    avg_degree = (2.0 * undirected_edge_count / node_count) if node_count else 0.0
    adj = _adjacency_undirected(store)
    cc = _clustering_coefficient(adj)
    lcc = _largest_component_fraction(adj)
    return GraphStats(
        node_count=node_count,
        undirected_edge_count=undirected_edge_count,
        avg_degree=float(avg_degree),
        clustering_coeff=float(cc),
        lcc_fraction=float(lcc),
    )


def build_subset_graph(
    vstore: VectorStoreManager,
    subset_ids: Set[str],
    *,
    semantic_similarity_threshold: float = 0.75,
    semantic_top_k: int = 12,
) -> GraphStore:
    raw = vstore.collection.get(include=["documents", "embeddings", "metadatas"])
    ids: List[str] = raw.get("ids", [])
    docs: List[str] = raw.get("documents", [])
    embs: List[List[float]] = raw.get("embeddings", [])
    metas: List[Dict] = raw.get("metadatas", [])

    store = GraphStore()
    for node_id, text, emb, meta in zip(ids, docs, embs, metas):
        if node_id not in subset_ids:
            continue
        store.add_node({"id": node_id, "text": text, "embedding": emb, "metadata": meta or {}})

    for node_id in list(store.nodes.keys()):
        md = store.nodes[node_id].get("metadata", {}) or {}
        prev_id = md.get("prev_id") or ""
        next_id = md.get("next_id") or ""
        if prev_id and prev_id in store.nodes:
            store.add_edge(node_id, prev_id, 1.0, "sequential", bidirectional=True)
        if next_id and next_id in store.nodes:
            store.add_edge(node_id, next_id, 1.0, "sequential", bidirectional=True)

    seen_pairs: Set[Tuple[str, str]] = set()
    for node_id in list(store.nodes.keys()):
        emb = store.nodes[node_id]["embedding"]
        res = vstore.collection.query(
            query_embeddings=[emb],
            n_results=semantic_top_k + 1,
            include=["distances"],
        )
        neigh_ids = (res.get("ids") or [[]])[0]
        neigh_distances = (res.get("distances") or [[]])[0]
        for neigh_id, dist in zip(neigh_ids, neigh_distances):
            if neigh_id == node_id:
                continue
            if neigh_id not in store.nodes:
                continue
            sim = _cosine_sim_from_chroma_distance(dist)
            if sim < semantic_similarity_threshold:
                continue
            a, b = (node_id, neigh_id) if node_id < neigh_id else (neigh_id, node_id)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            store.add_edge(node_id, neigh_id, sim, "semantic", bidirectional=True)

    return store


def insert_nodes_incrementally(
    vstore: VectorStoreManager,
    store: GraphStore,
    insert_ids: List[str],
    *,
    semantic_similarity_threshold: float = 0.75,
    semantic_top_k: int = 12,
) -> None:
    raw = vstore.collection.get(include=["documents", "embeddings", "metadatas"])
    by_id: Dict[str, Tuple[str, List[float], Dict]] = {}
    for node_id, text, emb, meta in zip(raw.get("ids", []), raw.get("documents", []), raw.get("embeddings", []), raw.get("metadatas", [])):
        by_id[node_id] = (text, emb, meta or {})

    for node_id in insert_ids:
        if node_id in store.nodes:
            continue
        if node_id not in by_id:
            continue
        text, emb, meta = by_id[node_id]
        store.add_node({"id": node_id, "text": text, "embedding": emb, "metadata": meta})

        prev_id = meta.get("prev_id") or ""
        next_id = meta.get("next_id") or ""
        if prev_id and prev_id in store.nodes:
            store.add_edge(node_id, prev_id, 1.0, "sequential", bidirectional=True)
        if next_id and next_id in store.nodes:
            store.add_edge(node_id, next_id, 1.0, "sequential", bidirectional=True)

        res = vstore.collection.query(
            query_embeddings=[emb],
            n_results=semantic_top_k + 1,
            include=["distances"],
        )
        neigh_ids = (res.get("ids") or [[]])[0]
        neigh_distances = (res.get("distances") or [[]])[0]
        for neigh_id, dist in zip(neigh_ids, neigh_distances):
            if neigh_id == node_id:
                continue
            if neigh_id not in store.nodes:
                continue
            sim = _cosine_sim_from_chroma_distance(dist)
            if sim < semantic_similarity_threshold:
                continue
            store.add_edge(node_id, neigh_id, sim, "semantic", bidirectional=True)


def write_update_table(out_path: Path, rows: List[Dict]) -> None:
    df = pd.DataFrame(rows)
    latex = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Regime & Avg degree & Clustering & LCC fraction & Edge Jaccard \\\\\n"
        "\\midrule\n"
    )
    for _, r in df.iterrows():
        latex += (
            f"{r['regime']} & {r['avg_degree']:.3f} & {r['clustering_coeff']:.3f} & {r['lcc_fraction']:.3f} & {r['edge_jaccard']:.3f} \\\\\n"
        )
    latex += "\\bottomrule\n\\end{tabular}\n"
    out_path.write_text(latex, encoding="utf-8")


def edge_jaccard(a: Set[Tuple[str, str, str]], b: Set[Tuple[str, str, str]]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a.intersection(b))
    uni = len(a.union(b))
    return float(inter / uni) if uni else 0.0


def main() -> None:
    load_dotenv()
    random.seed(42)

    base = Path(__file__).resolve().parent
    tables_dir = base / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vstore = VectorStoreManager(persist_directory="data_engine/storage/chroma_db")
    all_ids = vstore.collection.get(include=[])["ids"]

    all_ids = list(all_ids)
    random.shuffle(all_ids)

    split = int(0.8 * len(all_ids))
    base_ids = set(all_ids[:split])
    insert_ids = list(all_ids[split:])

    full_store = build_subset_graph(vstore, set(all_ids))
    full_stats = compute_stats(full_store)
    full_edges = _undirected_edge_set(full_store)

    base_store = build_subset_graph(vstore, base_ids)
    base_stats = compute_stats(base_store)

    insert_store = build_subset_graph(vstore, base_ids)
    insert_nodes_incrementally(vstore, insert_store, insert_ids)
    insert_stats = compute_stats(insert_store)
    insert_edges = _undirected_edge_set(insert_store)

    rows = [
        {
            "regime": "initial 80pct",
            "avg_degree": base_stats.avg_degree,
            "clustering_coeff": base_stats.clustering_coeff,
            "lcc_fraction": base_stats.lcc_fraction,
            "edge_jaccard": edge_jaccard(_undirected_edge_set(base_store), full_edges),
        },
        {
            "regime": "incremental insert",
            "avg_degree": insert_stats.avg_degree,
            "clustering_coeff": insert_stats.clustering_coeff,
            "lcc_fraction": insert_stats.lcc_fraction,
            "edge_jaccard": edge_jaccard(insert_edges, full_edges),
        },
        {
            "regime": "rebuild full",
            "avg_degree": full_stats.avg_degree,
            "clustering_coeff": full_stats.clustering_coeff,
            "lcc_fraction": full_stats.lcc_fraction,
            "edge_jaccard": 1.0,
        },
    ]

    write_update_table(tables_dir / "update_experiment.tex", rows)
    print(f"Wrote {tables_dir / 'update_experiment.tex'}")


if __name__ == "__main__":
    main()

