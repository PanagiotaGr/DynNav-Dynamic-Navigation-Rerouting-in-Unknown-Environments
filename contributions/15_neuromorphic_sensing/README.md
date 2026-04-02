# Contribution 15 — Neuromorphic Sensing

[![Module](https://img.shields.io/badge/Module-15-purple)](.) [![Type](https://img.shields.io/badge/Type-Neuromorphic%20%2F%20Bio--inspired-blue)](.) [![Tests](https://img.shields.io/badge/Tests-3%20passing-brightgreen)](.)

## Overview

**Event camera (DVS) simulator + Spiking Neural Network (SNN)** for ultra-low-latency obstacle detection. Event cameras fire asynchronously per-pixel at microsecond resolution — matching moving obstacles far better than frame-based cameras.

## Research Question

> **RQ-Neuro**: Does event-camera-based obstacle detection reduce reaction latency in high-speed scenarios vs frame-based pipelines?

## How It Works

```
Greyscale frames → DVS simulator → async events (μs timestamps) → time surface → SNN → obstacle grid
```

- DVS fires ON/OFF events when log-intensity changes exceed threshold
- Events are converted to a 2-channel time surface (exponential decay)
- Leaky Integrate-and-Fire (LIF) neurons process spikes per spatial cell
- Output: (N×M) obstacle probability map at microsecond latency

## Files

```
15_neuromorphic_sensing/
├── neuromorphic_sensing.py    # DVSSimulator, SNNObstacleDetector, LIFNeuron
└── experiments/
```

## Quick Start

```python
from contributions.15_neuromorphic_sensing.neuromorphic_sensing import (
    DVSSimulator, SNNObstacleDetector, event_to_time_surface
)
import numpy as np

sim = DVSSimulator()
detector = SNNObstacleDetector(grid_n=4, grid_m=4)

frame1 = np.random.rand(240, 320)
frame2 = np.clip(frame1 + 0.3, 0, 1)   # simulate motion

sim.process_frame(frame1)
events = sim.process_frame(frame2)       # → list of DVSEvent

surface = event_to_time_surface(events, h=240, w=320)
obstacle_map = detector.detect(surface)  # → (4, 4) probability grid
```

## Integration

- **Parallel sensing** alongside LiDAR SLAM (`lidar_ros2/`)
- **Fused** in belief state for fast-moving obstacle response
- **ROS2**: publish event stream on `/dvs/events` topic
