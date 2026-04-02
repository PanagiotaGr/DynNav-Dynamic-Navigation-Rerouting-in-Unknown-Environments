"""
Contribution 17: Topological + Semantic Maps
=============================================
Builds and navigates a graph of semantic zones (rooms, corridors, doorways)
rather than a dense metric occupancy grid.  Nodes in the graph represent
named semantic regions; edges represent traversable transitions.

Open-vocabulary grounding: CLIP embeddings (or a stub) score how well
a visual observation matches a semantic label, enabling instruction following
such as "go to the kitchen" without a pre-built map.

Research Question (RQ-Topo): Does a topological-semantic representation
reduce planning complexity and improve long-horizon navigation success rate
in large unknown environments?

References:
    Savinov et al. (2018) "Semi-parametric Topological Memory for Navigation"
    Gu et al. (2022) "Open-vocabulary Mobile Manipulation via Embodied Visual Matching"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph nodes and edges
# ---------------------------------------------------------------------------

@dataclass
class SemanticNode:
    node_id: int
    label: str                     # e.g. "kitchen", "corridor_A", "doorway_1"
    centroid_xy: tuple[float, float]  # approximate metric centroid
    embedding: Optional[np.ndarray] = None   # CLIP or learned embedding
    properties: dict = field(default_factory=dict)

    def similarity(self, query_emb: np.ndarray) -> float:
        """Cosine similarity to a query embedding."""
        if self.embedding is None:
            return 0.0
        a = self.embedding / (np.linalg.norm(self.embedding) + 1e-8)
        b = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        return float(np.dot(a, b))


@dataclass
class SemanticEdge:
    src: int
    dst: int
    weight: float = 1.0            # traversal cost
    passable: bool = True
    transition_type: str = "open"  # "open" | "door" | "narrow"


# ---------------------------------------------------------------------------
# Topological map
# ---------------------------------------------------------------------------

class TopologicalSemanticMap:
    """
    A sparse graph of semantic zones with metric centroids and embeddings.
    Supports:
        - Node creation from observations
        - Open-vocabulary search (find node closest to a text/visual query)
        - Dijkstra path planning over the graph
        - Dynamic edge invalidation (obstacle discovered → mark impassable)
    """

    def __init__(self):
        self.nodes: dict[int, SemanticNode] = {}
        self.edges: list[SemanticEdge] = []
        self._adj: dict[int, list[SemanticEdge]] = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, label: str,
                 centroid_xy: tuple[float, float],
                 embedding: Optional[np.ndarray] = None,
                 **props) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = SemanticNode(nid, label, centroid_xy,
                                        embedding, props)
        self._adj[nid] = []
        logger.debug("Added node %d: %s at %s", nid, label, centroid_xy)
        return nid

    def add_edge(self, src: int, dst: int,
                 weight: float = 1.0,
                 bidirectional: bool = True,
                 transition_type: str = "open") -> None:
        e = SemanticEdge(src, dst, weight, True, transition_type)
        self.edges.append(e)
        self._adj[src].append(e)
        if bidirectional:
            e2 = SemanticEdge(dst, src, weight, True, transition_type)
            self.edges.append(e2)
            self._adj[dst].append(e2)

    def invalidate_edge(self, src: int, dst: int) -> None:
        """Mark an edge impassable (obstacle discovered)."""
        for e in self._adj.get(src, []):
            if e.dst == dst:
                e.passable = False
                logger.info("Edge %d→%d invalidated", src, dst)

    # ------------------------------------------------------------------
    # Open-vocabulary grounding
    # ------------------------------------------------------------------

    def ground_query(self, query_embedding: np.ndarray,
                     top_k: int = 3) -> list[tuple[int, str, float]]:
        """
        Find top-k nodes whose embedding best matches a query.
        Returns list of (node_id, label, similarity).
        """
        scored = [
            (nid, node.label, node.similarity(query_embedding))
            for nid, node in self.nodes.items()
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def embed_label_stub(self, label: str) -> np.ndarray:
        """
        Stub CLIP embedding: uses a deterministic hash of the label string.
        Replace with: clip_model.encode_text(label).cpu().numpy()
        """
        rng = np.random.default_rng(hash(label) % (2 ** 32))
        emb = rng.standard_normal(512)
        return emb / (np.linalg.norm(emb) + 1e-8)

    # ------------------------------------------------------------------
    # Path planning (Dijkstra)
    # ------------------------------------------------------------------

    def plan(self, start_id: int,
             goal_id: int) -> tuple[list[int], float]:
        """
        Dijkstra over the semantic graph.
        Returns (path_as_node_ids, total_cost).
        Ignores impassable edges.
        """
        import heapq

        if start_id not in self.nodes or goal_id not in self.nodes:
            raise ValueError(f"Node {start_id} or {goal_id} not in map")

        dist = {nid: np.inf for nid in self.nodes}
        prev: dict[int, Optional[int]] = {nid: None for nid in self.nodes}
        dist[start_id] = 0.0
        heap = [(0.0, start_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for edge in self._adj.get(u, []):
                if not edge.passable:
                    continue
                alt = dist[u] + edge.weight
                if alt < dist[edge.dst]:
                    dist[edge.dst] = alt
                    prev[edge.dst] = u
                    heapq.heappush(heap, (alt, edge.dst))

        if np.isinf(dist[goal_id]):
            logger.warning("No path from %d to %d", start_id, goal_id)
            return [], np.inf

        path, cur = [], goal_id
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path, dist[goal_id]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.node_id,
                    "label": n.label,
                    "centroid": n.centroid_xy,
                    "properties": n.properties,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "src": e.src, "dst": e.dst,
                    "weight": e.weight, "passable": e.passable,
                    "type": e.transition_type,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopologicalSemanticMap":
        m = cls()
        for n in data["nodes"]:
            nid = m.add_node(n["label"], tuple(n["centroid"]), **n.get("properties", {}))
            assert nid == n["id"]
        for e in data["edges"]:
            edge = SemanticEdge(e["src"], e["dst"], e["weight"],
                                e["passable"], e["type"])
            m.edges.append(edge)
            m._adj[e["src"]].append(edge)
        return m

    def summary(self) -> str:
        return (
            f"TopologicalSemanticMap | "
            f"{len(self.nodes)} nodes | "
            f"{len(self.edges)} edges | "
            f"{sum(1 for e in self.edges if not e.passable)} blocked"
        )
