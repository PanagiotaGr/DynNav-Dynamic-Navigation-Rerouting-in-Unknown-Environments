# Contribution 25 — Adversarial Attack Simulator

[![Module](https://img.shields.io/badge/Module-25-purple)](.) [![Type](https://img.shields.io/badge/Type-Cybersecurity%20%2F%20Robustness-blue)](.) [![Tests](https://img.shields.io/badge/Tests-6%20passing-brightgreen)](.)

## Overview

Generates **physics-plausible adversarial attacks** on robot sensor data to evaluate DynNav's robustness. Tests FGSM/PGD gradient attacks on neural inputs, LiDAR spoofing (phantom obstacles, point removal), and odometry drift injection.

## Research Question

> **RQ-Adv**: How robust is DynNav's pipeline to adversarial sensor manipulation, and does the IDS (Contribution 08) detect these attacks?

## Attack Types

| Attack | Target | Method |
|--------|--------|--------|
| FGSM | Neural network inputs | ε·sign(∇L) perturbation |
| PGD | Neural network inputs | Iterative projected gradient |
| LiDAR Spoof Add | Point cloud | Inject N phantom obstacle clusters |
| LiDAR Spoof Remove | Point cloud | Remove points near target region |
| LiDAR Blind | Point cloud | Zero out angular sector |
| Odometry Drift | Position estimate | Gaussian drift injection |

## Files

```
25_adversarial_attack_simulator/
├── adversarial_attacks.py    # GradientAttacker, LiDARAttacker, OdometrySpoofer, RobustnessEvaluator
└── experiments/
```

## Quick Start

```python
from contributions.25_adversarial_attack_simulator.adversarial_attacks import (
    RobustnessEvaluator, AttackConfig
)
import numpy as np

evaluator = RobustnessEvaluator(AttackConfig(epsilon=0.1, pgd_steps=20))

obs_samples  = [np.random.rand(16) for _ in range(10)]
point_clouds = [np.random.rand(500, 3) * 10]
loss_fn      = lambda x: float(-np.sum(x))

results = evaluator.evaluate(obs_samples, loss_fn, point_clouds)
print(results)
# → {'fgsm_mean_loss_increase': 1.23, 'lidar_phantom_points_added': 40, ...}
```

## Integration

- **Tests**: `contributions/08_security_ids/` intrusion detection
- **Combines with**: Contribution 14 (Causal) to attribute attack impact
