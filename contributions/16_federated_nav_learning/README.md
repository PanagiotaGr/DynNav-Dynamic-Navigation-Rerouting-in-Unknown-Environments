# Contribution 16 — Federated Navigation Learning

[![Module](https://img.shields.io/badge/Module-16-purple)](.) [![Type](https://img.shields.io/badge/Type-Federated%20Learning-blue)](.) [![Tests](https://img.shields.io/badge/Tests-3%20passing-brightgreen)](.)

## Overview

**Privacy-preserving multi-robot learning** using FedAvg. Each robot trains a local navigation model on its own sensor data, shares only encrypted model deltas — never raw maps or trajectories — and receives a globally improved model.

## Research Question

> **RQ-Fed**: Does federated learning improve generalisation to unseen environments without sharing private map data?

## How It Works

```
Global model → broadcast to N robots → local training → DP-noised model delta → FedAvg → new global model
```

Supports:
- **FedAvg** (McMahan et al. 2017): weighted average of client updates
- **Differential Privacy**: Gaussian mechanism with (ε, δ)-DP guarantees
- **Contribution-weighted** aggregation: robots with more data have more influence

## Files

```
16_federated_nav_learning/
├── federated_nav.py    # FederatedServer, FederatedRobotClient, NavModel
└── experiments/
```

## Quick Start

```python
from contributions.16_federated_nav_learning.federated_nav import (
    FedNavConfig, NavModel, FederatedRobotClient, FederatedServer
)

cfg = FedNavConfig(n_robots=6, global_rounds=20, dp_epsilon=1.0)
global_model = NavModel.random_init(in_d=16, out_d=2)
clients = [FederatedRobotClient(i, global_model, cfg) for i in range(6)]
server = FederatedServer(global_model, cfg)

history = server.run_training(clients)
print(f"Final val_MSE: {history[-1]['val_mse']:.4f}")
```

## Integration

- **Distributed version** of learned heuristics (`contributions/01_learned_astar/`)
- **Combines with**: Contribution 26 (Swarm Consensus) for coordinated fleet learning
