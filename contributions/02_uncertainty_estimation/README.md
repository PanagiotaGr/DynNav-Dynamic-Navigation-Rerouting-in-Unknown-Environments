# Contribution 02 — Uncertainty Estimation

[![Module](https://img.shields.io/badge/Module-02-purple)](.) [![Type](https://img.shields.io/badge/Type-Probabilistic%20Sensing-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

Explicit **uncertainty modelling** for robot state estimation using Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF). Quantifies sensing uncertainty, localisation error, and map confidence for use in downstream risk-aware planning.

## Research Question

> **RQ2**: How can uncertainty be explicitly incorporated into navigation decisions?

## How It Works

```
Sensor readings → EKF/UKF → belief state (μ, Σ) → uncertainty metrics → planner
```

- **EKF**: linearised Kalman filter for Gaussian noise models
- **UKF**: sigma-point propagation for non-linear systems
- **Output**: per-cell uncertainty estimates fed to risk planner (Contribution 03)

## Files

```
02_uncertainty_estimation/
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/02_uncertainty_estimation/experiments/eval_uncertainty.py
```

## Key Concepts

| Method | Use Case | Complexity |
|--------|----------|------------|
| EKF | Linear/mildly non-linear systems | O(n²) |
| UKF | Strongly non-linear systems | O(n³) |
| Particle Filter | Multimodal distributions | O(N particles) |

## Integration

- **Feeds into**: Contribution 03 (belief-space planning)
- **Extended by**: Contribution 12 (diffusion maps) for occupancy uncertainty
- **Extended by**: Contribution 24 (NeRF uncertainty) for exploration
