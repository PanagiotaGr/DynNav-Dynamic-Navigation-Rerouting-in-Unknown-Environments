# Contribution 20 — Multimodal Failure Explainer

[![Module](https://img.shields.io/badge/Module-20-purple)](.) [![Type](https://img.shields.io/badge/Type-XAI%20%2F%20Diagnostics-blue)](.) [![Tests](https://img.shields.io/badge/Tests-4%20passing-brightgreen)](.)

## Overview

After a navigation failure, automatically generates a **structured human-readable failure report** combining VLM scene description, causal root-cause analysis (Contribution 14), STL robustness summary (Contribution 18), and suggested corrective actions.

## Research Question

> **RQ-Explain**: Does automated failure explanation reduce operator debugging time and improve replanning quality?

## How It Works

```
FailureEvent → VLM scene description + SCM root causes + STL trace → FailureReport (Markdown/JSON)
```

## Files

```
20_multimodal_failure_explainer/
├── multimodal_failure_explainer.py    # MultimodalFailureExplainer, FailureReport
└── experiments/
```

## Quick Start

```python
from contributions.20_multimodal_failure_explainer.multimodal_failure_explainer import (
    MultimodalFailureExplainer, FailureEvent, FailureType
)

explainer = MultimodalFailureExplainer(use_vlm=False, use_causal=True)

event = FailureEvent(
    failure_type=FailureType.COLLISION,
    timestamp=12.5,
    robot_pos=(3.2, 4.1),
    robot_vel=(0.3, 0.0),
    sensor_readings={"min_obstacle_dist": 0.15},
)

report = explainer.explain(event)
print(report.to_markdown())
```

## Sample Output

```markdown
# Failure Report — collision
**Time:** 12.50s | **Position:** (3.2, 4.1)

## Root Causes
- `map_accuracy`: causal contribution = -0.341
- `obstacle_detection`: causal contribution = -0.336

## Suggested Corrective Actions
1. Increase safety radius in CBF filter (Contribution 18)
2. Re-run diffusion occupancy predictor (Contribution 12)
```
