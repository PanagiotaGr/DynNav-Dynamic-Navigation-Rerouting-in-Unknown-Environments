# Contribution 24 — NeRF Uncertainty Maps

[![Module](https://img.shields.io/badge/Module-24-purple)](.) [![Type](https://img.shields.io/badge/Type-3D%20Mapping%20%2F%20Uncertainty-blue)](.) [![Tests](https://img.shields.io/badge/Tests-5%20passing-brightgreen)](.)

## Overview

Uses **Neural Radiance Field (NeRF) rendering confidence** as a proxy for spatial uncertainty. Low rendering confidence (high photometric error + MC-Dropout variance) indicates unobserved or uncertain areas — guiding the exploration planner.

## Research Question

> **RQ-NeRF**: Does NeRF-derived uncertainty provide better exploration guidance than entropy-based occupancy uncertainty?

## How It Works

```
Camera poses → ray casting → TinyNeRF (MC-Dropout, N passes) → variance map → exploration weights
```

- **MC-Dropout**: N forward passes with dropout → sigma variance = uncertainty
- **Volume rendering**: depth + colour + uncertainty per ray
- **2D projection**: uncertainty aggregated onto navigation grid

## Files

```
24_nerf_uncertainty/
├── nerf_uncertainty.py    # TinyNeRF, PositionalEncoder, NeRFUncertaintyMapper
└── experiments/
```

## Quick Start

```python
from contributions.24_nerf_uncertainty.nerf_uncertainty import NeRFUncertaintyMapper, NeRFConfig
import numpy as np

mapper = NeRFUncertaintyMapper(cfg=NeRFConfig(grid_size=(64, 64)))

poses = [np.eye(4) for _ in range(10)]   # replace with real camera poses
unc_map = mapper.build_uncertainty_map(poses)

# Convert to exploration priority weights
weights = mapper.uncertainty_to_exploration_weights(unc_map, occupancy=occ_grid)
next_frontier = np.unravel_index(weights.argmax(), weights.shape)
```

## Integration

- **Combines with**: Contribution 23 (Gaussian Splatting) for frontier weighting
- **Combines with**: Contribution 12 (Diffusion) as uncertainty prior
- **Feeds**: `ig_explorer/` next-best-view planner
