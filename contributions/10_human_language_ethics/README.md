# Contribution 10 — Human-Aware & Ethics-Guided Navigation

[![Module](https://img.shields.io/badge/Module-10-purple)](.) [![Type](https://img.shields.io/badge/Type-HRI%20%2F%20Ethics-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

Extends DynNav with **human preference modelling**, **trust-aware decision making**, and **ethical zone constraints**. The robot respects no-go zones (hospitals, private areas), adapts speed near humans, and incorporates operator trust levels into planning.

## Research Question

> **RQ8**: Can navigation incorporate human preferences and trust?

## How It Works

```
Human proximity + ethical zones + operator trust → preference-weighted cost → trust-aware planner
```

- **Ethical zones**: defined in `ethical_zones.json` — hard no-go or soft penalty regions
- **Human proximity**: detected humans → reduce speed, increase clearance
- **Trust model**: operator trust score scales autonomy level (high trust = more autonomous)
- **Language interface**: natural language preferences → planning constraints

## Files

```
10_human_language_ethics/
├── experiments/
└── results/

ethical_zones.json    # Zone definitions (root of repo)
```

## Quick Start

```bash
python contributions/10_human_language_ethics/experiments/eval_human_ethics.py
```

## Ethical Zone Types

| Zone Type | Behaviour |
|-----------|-----------|
| `no_go` | Hard constraint — never enter |
| `slow_zone` | Reduce max velocity to 0.2 m/s |
| `announce` | Announce robot presence via speaker |
| `avoid_if_possible` | Soft penalty in cost function |

## Integration

- **Extended by**: Contribution 11 (VLM) for visual human detection
- **Extended by**: Contribution 19 (LLM Mission Planner) for language instructions
- **Extended by**: Contribution 20 (Failure Explainer) for human-readable reports
- **Config**: `ethical_zones.json` in repo root
