# Contribution 05 — Safe-Mode Navigation

[![Module](https://img.shields.io/badge/Module-05-purple)](.) [![Type](https://img.shields.io/badge/Type-Adaptive%20Safety-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

**Adaptive safe-mode** mechanism that switches the robot to a conservative navigation strategy when risk exceeds a threshold — reducing speed, increasing safety margins, and triggering replanning. Automatically deactivates when risk drops back to normal.

## Research Question

> **RQ3/RQ6**: How should navigation adapt when safety is threatened or resources are constrained?

## How It Works

```
Risk monitor → threshold exceeded → activate safe mode → conservative planner → risk normalises → deactivate
```

Safe-mode behaviours:
- Reduce maximum velocity
- Increase obstacle inflation radius
- Switch from greedy to conservative A*
- Trigger immediate replanning
- Alert human operator if risk persists

## Files

```
05_safe_mode_navigation/
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/05_safe_mode_navigation/experiments/eval_safe_mode.py
```

## State Machine

```
NORMAL → (risk > threshold) → SAFE_MODE → (risk < threshold, T steps) → NORMAL
SAFE_MODE → (risk > critical) → EMERGENCY_STOP
```

## Integration

- **Triggered by**: Contribution 03 (risk planner) and Contribution 08 (IDS alerts)
- **Extended by**: Contribution 13 (world model pre-screening)
- **Extended by**: Contribution 18 (CBF as always-on safety layer)
