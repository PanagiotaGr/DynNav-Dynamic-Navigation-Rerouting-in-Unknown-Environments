# Contribution 12 — Diffusion Occupancy Maps

[![Module](https://img.shields.io/badge/Module-12-purple)](.) [![Type](https://img.shields.io/badge/Type-Probabilistic%20Planning-blue)](.) [![Tests](https://img.shields.io/badge/Tests-4%20passing-brightgreen)](.)

## Overview

Score-based **diffusion model** for probabilistic occupancy prediction. Instead of a single deterministic occupancy grid, produces a *distribution* over future occupancy maps — enabling CVaR-95 risk estimation for the downstream planner.

## Research Question

> **RQ-Diff**: Does a diffusion-based occupancy model reduce collision rate and improve risk estimation compared to deterministic predictions?

## How It Works

```
Occupancy history → DDPM reverse diffusion (N samples) → CVaR risk map → risk-weighted A*
```

1. Recent occupancy frames are used as conditioning signal
2. Reverse diffusion generates N plausible future occupancy maps
3. CVaR-95 (worst-case 5%) gives a conservative risk estimate per cell
4. Risk map replaces deterministic inflation in the A* cost function

## Files

```
12_diffusion_occupancy/
├── diffusion_occupancy.py              # Core: DiffusionOccupancyPredictor
├── experiments/
│   └── eval_diffusion_occupancy.py    # Benchmark vs deterministic baseline
└── results/
```

## Quick Start

```bash
python contributions/12_diffusion_occupancy/experiments/eval_diffusion_occupancy.py \
    --n_scenarios 30 --n_samples 10 \
    --out_csv contributions/12_diffusion_occupancy/results/diffusion_eval.csv
```

## Key Classes

| Class | Description |
|-------|-------------|
| `DiffusionOccupancyPredictor` | Main predictor — runs reverse diffusion, computes risk maps |
| `ScoreNetwork` | Stub score estimator (replace with U-Net from 🤗 Diffusers) |
| `DiffusionOccupancyConfig` | Grid size, T steps, n_samples, beta schedule |

## Integration

- **Replaces**: deterministic occupancy inflation in `contributions/03_belief_risk_planning/`
- **Output**: `{"mean", "std", "cvar_95", "samples"}` risk maps
- **Combines with**: Contribution 24 (NeRF uncertainty) as additional prior

## Production Upgrade

```python
# Replace ScoreNetwork stub with HuggingFace UNet:
from diffusers import UNet2DModel
score_net = UNet2DModel(sample_size=64, in_channels=1, out_channels=1)
```
