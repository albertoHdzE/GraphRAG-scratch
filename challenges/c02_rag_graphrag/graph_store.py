from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    weight: float
    type: str  # "sequential" | "semantic" | ...


class GraphStore:
    """
    Minimal explicit graph storage:

    - Node registry: node_id -> node dict
    - Adjacency list: node_id -> List[(neighbor_id, weight, type)]
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.graph: Dict[str, List[Tuple[str, float, str]]] = {}

    def add_node(self, node: Dict[str, Any]) -> None:
        node_id = node["id"]
        self.nodes[node_id] = node
        self.graph.setdefault(node_id, [])

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        type: str,
        *,
        bidirectional: bool = True,
    ) -> None:
        if source not in self.graph:
            self.graph[source] = []
        self.graph[source].append((target, float(weight), type))

        if bidirectional:
            if target not in self.graph:
                self.graph[target] = []
            self.graph[target].append((source, float(weight), type))

    def neighbors(
        self,
        node_id: str,
        *,
        min_weight: Optional[float] = None,
        edge_type: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        res: List[Tuple[str, float, str]] = []
        for neigh_id, w, t in self.graph.get(node_id, []):
            if min_weight is not None and w < min_weight:
                continue
            if edge_type is not None and t != edge_type:
                continue
            res.append((neigh_id, w, t))
        return res

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        # Each bidirectional edge is stored twice; keep "stored edges" as-is.
        return sum(len(v) for v in self.graph.values())

