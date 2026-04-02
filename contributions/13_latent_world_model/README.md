# Contribution 13 — Latent World Model

[![Module](https://img.shields.io/badge/Module-13-purple)](.) [![Type](https://img.shields.io/badge/Type-Model--Based%20RL-blue)](.) [![Tests](https://img.shields.io/badge/Tests-4%20passing-brightgreen)](.)

## Overview

**Dreamer-v3 style RSSM** (Recurrent State Space Model) that learns a compact latent representation of the environment and performs *mental rollouts* — imagined future trajectories — before committing to any real action.

## Research Question

> **RQ-WM**: Do mental rollouts in latent space reduce irreversible failures compared to reactive replanning alone?

## How It Works

```
Observation → RSSM posterior → latent (h, z) → imagine N action sequences → pick best → execute
```

1. Real observations update the recurrent latent state `(h, z)`
2. Before execution, `WorldModelPlanner` simulates K candidate action sequences in latent space
3. Sequences are scored by imagined reward + irreversibility penalty
4. Best sequence is executed; robot updates belief after each real step

## Files

```
13_latent_world_model/
├── latent_world_model.py    # RSSM, WorldModelPlanner
└── experiments/
```

## Quick Start

```python
from contributions.13_latent_world_model.latent_world_model import WorldModelPlanner, RSSMConfig

cfg = RSSMConfig(obs_dim=64, action_dim=2, horizon=12)
planner = WorldModelPlanner(config=cfg)

# Update belief with real observation
planner.update_belief(obs, action)

# Plan via mental rollouts
sequences = planner.generate_random_sequences(n=16, horizon=12)
best_seq, expected_return = planner.select_best_action_sequence(sequences)
```

## Key Classes

| Class | Description |
|-------|-------------|
| `RSSM` | Recurrent State Space Model — prior, posterior, reward predictor |
| `WorldModelPlanner` | Mental rollouts + best-action selection |
| `RSSMConfig` | Latent dim, hidden dim, horizon, irreversibility penalty |

## Integration

- **Wraps**: safe-mode planner (`contributions/05_safe_mode_navigation/`)
- **Combines with**: Contribution 04 (irreversibility) for pre-screening
- **Production**: replace numpy stubs with PyTorch GRU + MLP
