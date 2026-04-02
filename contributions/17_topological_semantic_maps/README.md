# Contribution 17 — Topological Semantic Maps

[![Module](https://img.shields.io/badge/Module-17-purple)](.) [![Type](https://img.shields.io/badge/Type-Mapping%20%2F%20Grounding-blue)](.) [![Tests](https://img.shields.io/badge/Tests-4%20passing-brightgreen)](.)

## Overview

**Graph of semantic zones** (kitchen, corridor, doorway) instead of a dense metric grid. Nodes represent named regions; edges represent traversable transitions. Supports **open-vocabulary grounding** via CLIP embeddings — "go to the kitchen" without a pre-built map.

## Research Question

> **RQ-Topo**: Does a topological-semantic representation reduce planning complexity and improve long-horizon success rate in large unknown environments?

## How It Works

```
Observations → zone detection → add nodes/edges → Dijkstra planning → metric waypoints
Natural language query → CLIP embed → cosine similarity → best matching zone
```

## Files

```
17_topological_semantic_maps/
├── topo_semantic_map.py    # TopologicalSemanticMap, SemanticNode, SemanticEdge
└── experiments/
```

## Quick Start

```python
from contributions.17_topological_semantic_maps.topo_semantic_map import TopologicalSemanticMap

m = TopologicalSemanticMap()
k = m.add_node("kitchen",    centroid_xy=(0.0, 0.0))
c = m.add_node("corridor",   centroid_xy=(3.0, 0.0))
l = m.add_node("living_room",centroid_xy=(6.0, 0.0))
m.add_edge(k, c, weight=3.0)
m.add_edge(c, l, weight=3.0)

# Plan a path
path, cost = m.plan(start_id=k, goal_id=l)

# Open-vocabulary search
emb = m.embed_label_stub("kitchen")
results = m.ground_query(emb, top_k=3)
```

## Integration

- **High-level layer** above metric occupancy grid
- **Connects to**: Contribution 19 (LLM Mission Planner) for instruction following
- **Serialisable** to/from JSON for persistence across sessions
