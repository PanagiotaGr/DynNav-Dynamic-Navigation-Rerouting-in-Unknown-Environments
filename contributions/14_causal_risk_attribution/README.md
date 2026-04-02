# Contribution 14 — Causal Risk Attribution

[![Module](https://img.shields.io/badge/Module-14-purple)](.) [![Type](https://img.shields.io/badge/Type-Causal%20AI-blue)](.) [![Tests](https://img.shields.io/badge/Tests-4%20passing-brightgreen)](.)

## Overview

**Structural Causal Model (SCM)** for navigation failure analysis. Goes beyond statistical correlation to answer *why* a failure occurred using counterfactual reasoning: "What would have happened if sensor noise had been zero?"

## Research Question

> **RQ-Causal**: Can counterfactual reasoning reduce repeated navigation failures by identifying root causes rather than symptoms?

## How It Works

```
Episode failure → SCM observational query → counterfactual intervention → root cause ranking
```

Causal graph:
```
sensor_noise → localization_error → obstacle_detection → path_risk → collision
            ↘ map_accuracy ↗
```

## Files

```
14_causal_risk_attribution/
├── causal_risk.py    # NavigationSCM, CausalNode, counterfactual queries
└── experiments/
```

## Quick Start

```python
from contributions.14_causal_risk_attribution.causal_risk import NavigationSCM

scm = NavigationSCM()

# After a collision episode — find root causes
noise = {"sensor_noise": 0.8, "localization_error": 0.3, ...}
ranking = scm.root_cause_ranking(noise, n_samples=200)
# → [("map_accuracy", -0.34), ("obstacle_detection", -0.33), ...]

# Counterfactual: what if sensor_noise had been 0?
cf = scm.counterfactual_query(noise, intervention={"sensor_noise": 0.0})
print(f"Collision without noise: {cf['collision']:.3f}")
```

## Key Methods

| Method | Description |
|--------|-------------|
| `observational_query(noise)` | Standard forward pass through causal graph |
| `counterfactual_query(noise, intervention)` | Do-calculus intervention |
| `average_causal_effect(X, Y, ...)` | Monte-Carlo ACE estimation |
| `root_cause_ranking(noise)` | Shapley-inspired ablation ranking |

## Integration

- **Post-hoc analysis** for: `contributions/08_security_ids/` anomaly events
- **Feeds into**: Contribution 20 (Multimodal Failure Explainer)
