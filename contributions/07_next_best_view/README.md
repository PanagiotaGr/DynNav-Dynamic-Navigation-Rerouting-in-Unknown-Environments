# Contribution 07 — Next-Best-View Exploration

[![Module](https://img.shields.io/badge/Module-07-purple)](.) [![Type](https://img.shields.io/badge/Type-Active%20Exploration-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

**Information-gain maximisation** for active exploration of unknown environments. The robot selects the next viewpoint that maximises expected information gain (reduces map entropy) subject to travel cost — enabling efficient mapping without human guidance.

## Research Question

> **RQ7 (implied)**: How can robots explore unknown spaces efficiently while minimising travel cost?

## How It Works

```
Current map → entropy map → candidate viewpoints → information gain estimate → cost-weighted selection → navigate
```

- **Entropy map**: per-cell uncertainty H(p_occ)
- **Information gain**: expected entropy reduction from candidate viewpoint
- **Cost**: travel distance + risk to reach viewpoint
- **Selection**: argmax(IG / cost) — greedy information-theoretic planner

## Files

```
07_next_best_view/ (see also: ig_explorer/)
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/07_next_best_view/experiments/eval_nbv.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Frontier cells | Boundary between known and unknown space |
| Information gain | Expected entropy reduction from a viewpoint |
| NBV | Next-Best-View — viewpoint maximising IG/cost |

## Integration

- **Uses**: `ig_explorer/` information-gain exploration module
- **Extended by**: Contribution 23 (Gaussian Splatting) for 3D frontier detection
- **Extended by**: Contribution 24 (NeRF uncertainty) for richer IG estimates
