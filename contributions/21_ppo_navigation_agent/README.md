# Contribution 21 — PPO Navigation Agent

[![Module](https://img.shields.io/badge/Module-21-purple)](.) [![Type](https://img.shields.io/badge/Type-Reinforcement%20Learning-blue)](.) [![Tests](https://img.shields.io/badge/Tests-5%20passing-brightgreen)](.)

## Overview

**Proximal Policy Optimization (PPO)** agent for autonomous navigation. Learns a policy π(a|s) mapping observations to velocity commands, with reward shaping from DynNav's risk and safety modules.

## Research Question

> **RQ-PPO**: Does RL-based navigation with risk-shaped rewards outperform classical A* in highly dynamic environments?

## Reward Structure

```
r = r_progress - λ_risk · r_risk - λ_collision · r_collision + r_goal
```

| Component | Value |
|-----------|-------|
| Step penalty | -0.01 |
| Progress reward | +0.1 / (dist_goal + 0.1) |
| Obstacle penalty | -0.5 · max(0, 1.5 - min_obs_dist) |
| Collision penalty | -5.0 |
| Goal reward | +10.0 |

## Files

```
21_ppo_navigation_agent/
├── ppo_nav_agent.py    # PPOAgent, ActorNetwork, CriticNetwork, NavEnv, RolloutBuffer
└── experiments/
```

## Quick Start

```python
from contributions.21_ppo_navigation_agent.ppo_nav_agent import PPOAgent, PPOConfig, NavEnv

cfg = PPOConfig(obs_dim=14, hidden_dim=128, rollout_len=512)
agent = PPOAgent(cfg)
env = NavEnv(cfg)

training_log = agent.train(env, n_updates=100)
print(f"Final reward: {training_log[-1]['mean_ep_reward']:.2f}")
```

## Integration

- **Combines with**: Contribution 12 (diffusion risk) for risk-shaped rewards
- **Combines with**: Contribution 18 (CBF shield) as safety wrapper during training
- **Extends with**: Contribution 22 (Curriculum RL) for staged training
