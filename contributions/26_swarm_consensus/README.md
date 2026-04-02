# Contribution 26 — Swarm Consensus Navigation

[![Module](https://img.shields.io/badge/Module-26-purple)](.) [![Type](https://img.shields.io/badge/Type-Multi--Robot%20%2F%20Distributed-blue)](.) [![Tests](https://img.shields.io/badge/Tests-5%20passing-brightgreen)](.)

## Overview

**Byzantine Fault-Tolerant (BFT) consensus** for multi-robot navigation plan selection. N robots independently compute local plans and broadcast cost estimates; the swarm uses weighted-median consensus — robust to up to ⌊(N-1)/3⌋ compromised or faulty robots.

## Research Question

> **RQ-Swarm**: Does BFT swarm consensus improve navigation robustness in the presence of compromised robots vs. naive majority voting?

## How It Works

```
N robots compute local A* plans → broadcast proposals → outlier detection (MAD) → weighted-median consensus → execute agreed plan
```

BFT guarantee: tolerates up to ⌊(N-1)/3⌋ Byzantine robots.

## Files

```
26_swarm_consensus/
├── swarm_consensus.py    # SwarmCoordinator, BFTConsensus, SwarmRobot, NavProposal
└── experiments/
```

## Quick Start

```python
from contributions.26_swarm_consensus.swarm_consensus import SwarmCoordinator
import numpy as np

# 6 robots, 1 Byzantine (faulty)
coord = SwarmCoordinator(n_robots=6, n_byzantine=1)

grid = np.zeros((20, 20))
grid[8:12, 8:12] = 1.0   # obstacle block

result = coord.plan(grid, start=(0, 0), goal=(18, 18))

print(f"Agreed cost:       {result.agreed_cost:.2f}")
print(f"Byzantine detected: {result.n_byzantine_detected}")
print(f"Method:            {result.method}")
print(f"Path length:       {len(result.agreed_path)} steps")
```

## Fault Types Simulated

| Type | Behaviour |
|------|-----------|
| `random` | Random corrupted path + random high cost |
| `constant_bad` | Always reports cost=9999 (lies) |
| `silent` | Does not respond (DoS) |

## Integration

- **Extends**: `contributions/09_multi_robot/` coordination framework
- **Combines with**: Contribution 16 (Federated Learning) for fleet-wide model updates
- **Security**: pairs with Contribution 25 (Adversarial) and Contribution 08 (IDS)
