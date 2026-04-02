# Contribution 03 — Belief-Space & Risk-Aware Planning

[![Module](https://img.shields.io/badge/Module-03-purple)](.) [![Type](https://img.shields.io/badge/Type-Risk--Aware%20Planning-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

Extends classical A* with **belief-space representations** and **risk-weighted cost functions**. The planner explicitly reasons about uncertainty and trades off path efficiency against collision risk using Conditional Value-at-Risk (CVaR) optimisation.

## Research Question

> **RQ3**: How should robots reason about risk and safety in dynamic environments?

## How It Works

```
Belief state (μ, Σ) + occupancy → risk map → CVaR cost → risk-weighted A* → safe path
```

- Risk cost: `c_risk(s) = CVaR_α(collision_prob_at_s)`
- Total cost: `f(s) = g(s) + λ·r(s) + h(s)`
- λ controls safety/efficiency trade-off

## Files

```
03_belief_risk_planning/
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/03_belief_risk_planning/experiments/eval_belief_risk.py
```

## Risk Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| Expected risk | E[collision] | Average-case planning |
| CVaR-95 | E[risk \| top 5%] | Conservative planning |
| Worst-case | max(risk) | Safety-critical |

## Integration

- **Receives**: uncertainty from Contribution 02
- **Extended by**: Contribution 12 (diffusion risk maps)
- **Extended by**: Contribution 18 (CBF hard constraints)
