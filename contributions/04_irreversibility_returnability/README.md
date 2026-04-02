# Contribution 04 — Irreversibility & Returnability

[![Module](https://img.shields.io/badge/Module-04-purple)](.) [![Type](https://img.shields.io/badge/Type-Safety%20%2F%20Planning-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

Prevents the robot from entering **unrecoverable states** by checking returnability before committing to any action. A state is returnable if the robot can reach a safe base position from it under current uncertainty.

## Research Question

> **RQ4**: How can navigation systems avoid irreversible decisions?

## How It Works

```
Candidate action → simulate forward → check: can robot return to safe state? → allow/block
```

- **Returnability check**: backward reachability analysis from candidate state
- **Feasibility threshold**: minimum clearance required for U-turn / recovery
- **Integration**: pre-screens actions before A* commitment

## Files

```
04_irreversibility_returnability/
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/04_irreversibility_returnability/experiments/eval_returnability.py
```

## Key Concepts

- **Returnable state**: ∃ path back to safe zone under current map + uncertainty
- **Irreversibility penalty**: added to cost of states with low returnability
- **Dead-end detection**: graph pruning of states with no return path

## Integration

- **Pre-screens**: all candidate actions before execution
- **Extended by**: Contribution 13 (world model mental rollouts for pre-screening)
- **Extended by**: Contribution 18 (CBF as hard irreversibility barrier)
