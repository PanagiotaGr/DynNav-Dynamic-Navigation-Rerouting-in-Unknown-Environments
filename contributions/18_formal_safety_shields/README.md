# Contribution 18 — Formal Safety Shields

[![Module](https://img.shields.io/badge/Module-18-purple)](.) [![Type](https://img.shields.io/badge/Type-Formal%20Methods-blue)](.) [![Tests](https://img.shields.io/badge/Tests-4%20passing-brightgreen)](.)

## Overview

**Runtime safety guarantees** via two complementary mechanisms:
1. **STL Monitor**: evaluates Signal Temporal Logic specs over the robot's state trajectory
2. **CBF Filter**: modifies velocity commands in real-time to guarantee obstacle avoidance (Lyapunov-style certificates)

## Research Question

> **RQ-Formal**: Do STL+CBF safety shields reduce constraint violations without significantly degrading navigation efficiency?

## How It Works

```
Planner command → CBF QP filter → safe command
State trajectory → STL monitor → violation alert
```

**CBF condition**: `∇h(x)·u + α·h(x) ≥ 0`  where `h(x) = ‖robot − obstacle‖ − r`

## Files

```
18_formal_safety_shields/
├── formal_safety_shields.py          # STLMonitor, CBFSafetyFilter, SafetyShield
├── experiments/
│   └── eval_safety_shields.py       # Shielded vs unshielded comparison
└── results/
```

## Quick Start

```bash
python contributions/18_formal_safety_shields/experiments/eval_safety_shields.py \
    --n_episodes 50 --out_csv contributions/18_formal_safety_shields/results/shield_eval.csv
```

```python
from contributions.18_formal_safety_shields.formal_safety_shields import (
    STLAtom, STLAlways, STLMonitor, CBFSafetyFilter, SafetyShield
)

shield = SafetyShield(monitor, cbf_filter)
u_safe, info = shield.step(u_desired, robot_pos, obstacles, robot_state)
```

## Integration

- **Drop-in wrapper** around any planner output
- **Pairs with**: Contribution 04 (irreversibility) for hard safety guarantees
- **Logs**: STL robustness ρ values and CBF correction norms per step
