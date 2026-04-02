# Contribution 23 — Gaussian Splatting Mapper

[![Module](https://img.shields.io/badge/Module-23-purple)](.) [![Type](https://img.shields.io/badge/Type-3D%20Mapping-blue)](.) [![Tests](https://img.shields.io/badge/Tests-5%20passing-brightgreen)](.)

## Overview

**Incremental 3D Gaussian Splatting map** built from RGB(D) frames. Each Gaussian represents a local scene element with position, covariance (shape), opacity, and colour. Supports uncertainty estimation, 2D occupancy extraction, and frontier detection for exploration.

## Research Question

> **RQ-3DGS**: Does a Gaussian-Splatting 3D map provide better uncertainty estimates for navigation than a 2D occupancy grid?

## How It Works

```
RGB-D frames → add/merge Gaussians → project to 2D → occupancy grid + uncertainty map + frontiers
```

- **Merge**: nearby Gaussians are updated via EKF-style fusion (no duplicates)
- **Prune**: low-confidence Gaussians removed as map evolves
- **Project**: Gaussians projected onto 2D plane → occupancy grid for DynNav planner

## Files

```
23_gaussian_splatting_mapper/
├── gaussian_splatting_map.py    # GaussianSplattingMap, Gaussian3D
└── experiments/
```

## Quick Start

```python
from contributions.23_gaussian_splatting_mapper.gaussian_splatting_map import GaussianSplattingMap
import numpy as np

gsmap = GaussianSplattingMap()

for i in range(20):
    points = np.random.rand(50, 3)   # replace with real depth cloud
    points[:, 2] = 0.5               # z ~ floor level
    gsmap.add_frame(points)

occ_grid  = gsmap.to_occupancy_grid()   # (H, W) for A*
unc_map   = gsmap.uncertainty_map()     # (H, W) for exploration
frontiers = gsmap.frontier_cells()      # list of (row, col)
print(gsmap.stats())
```

## Integration

- **Feeds**: A* planner with richer 3D occupancy
- **Combines with**: Contribution 24 (NeRF uncertainty) for exploration weighting
- **Production**: replace numpy stubs with nerfstudio / gaussian-splatting repo
