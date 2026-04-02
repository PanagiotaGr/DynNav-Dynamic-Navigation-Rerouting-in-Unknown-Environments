# Contribution 22 — Curriculum RL Training

[![Module](https://img.shields.io/badge/Module-22-purple)](.) [![Type](https://img.shields.io/badge/Type-Reinforcement%20Learning-blue)](.) [![Tests](https://img.shields.io/badge/Tests-5%20passing-brightgreen)](.)

## Overview

**Adaptive difficulty scheduling** for RL training. The agent starts in simple environments and automatically progresses to harder ones as it masters each stage — dramatically reducing sample complexity.

## Research Question

> **RQ-Curr**: Does curriculum training reduce sample complexity and improve generalisation to unseen environments?

## Difficulty Dimensions

| Stage | Obstacles | Map Size | Obstacle Speed | Sensor Noise |
|-------|-----------|----------|----------------|--------------|
| easy | 2 | 5×5 m | static | 0.0 |
| medium | 5 | 8×8 m | static | 0.05 |
| hard | 10 | 10×10 m | 0.1 m/s | 0.10 |
| expert | 18 | 15×15 m | 0.3 m/s | 0.20 |
| extreme | 25 | 20×20 m | 0.5 m/s | 0.30 |

## Files

```
22_curriculum_rl/
├── curriculum_rl.py    # CurriculumScheduler, DifficultyLevel, CurriculumNavEnv
└── experiments/
```

## Quick Start

```bash
python -c "
from contributions.22_curriculum_rl.curriculum_rl import run_curriculum_training
run_curriculum_training(n_episodes=500, out_csv='contributions/22_curriculum_rl/results/curriculum.csv')
"
```

## Strategies

- **ADAPTIVE** (default): advance when `success_rate > threshold` over rolling window
- **FIXED**: advance every N episodes regardless of performance
- **REVERSE**: start near goal, progressively expand start region (Florensa 2017)

## Integration

- **Wraps**: Contribution 21 (PPO agent) environment
- **Uses**: `data_curriculum/` directory already in DynNav repo
