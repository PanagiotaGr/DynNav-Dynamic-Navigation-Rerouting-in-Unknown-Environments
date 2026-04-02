# Contribution 06 — Energy & Connectivity-Aware Planning

[![Module](https://img.shields.io/badge/Module-06-purple)](.) [![Type](https://img.shields.io/badge/Type-Resource--Aware%20Planning-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

Extends DynNav's planner with **energy budget** and **communication connectivity** constraints. The robot plans paths that respect battery limits and maintain WiFi/ROS2 link quality — automatically routing to charging stations or relay positions when needed.

## Research Question

> **RQ6**: How should navigation adapt under resource constraints?

## How It Works

```
Battery level + connectivity map → resource-aware cost function → constrained A* → path with pit-stops
```

- **Energy model**: path cost includes motor power × distance
- **Connectivity model**: signal strength map → link quality per cell
- **Constraint**: path must be completable within battery budget
- **Fallback**: route via charging station if budget insufficient

## Files

```
06_energy_connectivity/
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/06_energy_connectivity/experiments/eval_energy_connectivity.py
```

## Resource Metrics

| Metric | Description |
|--------|-------------|
| Energy cost | Motor power × distance × terrain factor |
| Connectivity | RSSI / link quality per cell |
| Budget margin | Remaining energy after planned path |

## Integration

- **Receives**: battery state from robot hardware / ROS2 topic
- **Combines with**: Contribution 03 (risk planning) for multi-objective optimisation
- **Extended by**: Contribution 22 (curriculum RL) with energy as reward signal
