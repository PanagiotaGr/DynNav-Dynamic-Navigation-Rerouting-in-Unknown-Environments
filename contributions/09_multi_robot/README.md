# Contribution 09 — Multi-Robot Coordination

[![Module](https://img.shields.io/badge/Module-09-purple)](.) [![Type](https://img.shields.io/badge/Type-Multi--Robot%20Systems-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

**Decentralised multi-robot coordination** under uncertainty. Multiple robots share partial map observations, coordinate path planning to avoid conflicts, and allocate risk budgets across the team — without a central coordinator.

## Research Question

> **RQ7**: How can multiple robots coordinate under uncertainty?

## How It Works

```
N robots → local maps → gossip protocol → shared belief → conflict-free path allocation → execution
```

- **Map merging**: robots share occupancy observations via message passing
- **Conflict resolution**: decentralised priority-based path reservation
- **Risk allocation**: total team risk budget distributed per robot
- **Disagreement detection**: flags robots with inconsistent beliefs

## Files

```
09_multi_robot/
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/09_multi_robot/experiments/eval_multi_robot.py
```

## Coordination Strategies

| Strategy | Description | Scalability |
|----------|-------------|-------------|
| Centralised | Single coordinator plans all paths | Low |
| Decentralised priority | Robots reserve paths by priority | Medium |
| Market-based | Auction for path segments | High |

## Integration

- **Extended by**: Contribution 16 (Federated Learning) for fleet-wide model sharing
- **Extended by**: Contribution 26 (BFT Swarm Consensus) for fault-tolerant planning
- **ROS2**: multi-robot topics and namespacing
