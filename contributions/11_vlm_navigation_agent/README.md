# Contribution 11 — VLM Navigation Agent

[![Module](https://img.shields.io/badge/Module-11-purple)](.) [![Type](https://img.shields.io/badge/Type-Perception%20%2F%20LLM-blue)](.) [![Status](https://img.shields.io/badge/Tests-5%20passing-brightgreen)](.)

## Overview

Uses a **Vision-Language Model (VLM)** — LLaVA, BakLLaVA, or GPT-4V — as a high-level semantic planner. Instead of hand-crafted semantic labelling, the VLM interprets camera frames directly and outputs structured navigation goals.

## Research Question

> **RQ-VLM**: Can a foundation VLM replace hand-crafted semantic labelling and improve goal-specification in unknown environments?

## How It Works

```
RGB Frame → VLM API → JSON goal → back-projection → metric waypoint → A* planner
```

1. Camera frame is encoded and sent to the VLM endpoint (local Ollama or OpenAI)
2. VLM returns: `{"region": "corridor", "goal": "go forward", "confidence": 0.85, "pixel_u": 320, "pixel_v": 240}`
3. Pixel hint is back-projected using depth map → metric (x, y) waypoint
4. Waypoint is handed to DynNav's existing A* / topological planner

## Files

```
11_vlm_navigation_agent/
├── vlm_planner.py              # Core: VLMNavigationPlanner class
├── experiments/
│   └── eval_vlm_planner.py    # Offline evaluation on frame dataset
└── results/                   # CSV outputs
```

## Quick Start

```bash
# Offline evaluation (stub mode — no VLM server needed)
python contributions/11_vlm_navigation_agent/experiments/eval_vlm_planner.py \
    --n_frames 20 --out_csv contributions/11_vlm_navigation_agent/results/vlm_eval.csv

# With local Ollama (LLaVA)
ollama pull llava
python contributions/11_vlm_navigation_agent/experiments/eval_vlm_planner.py \
    --model llava-1.6 --endpoint http://localhost:11434/api/chat
```

## Key Classes

| Class | Description |
|-------|-------------|
| `VLMNavigationPlanner` | Main planner — encodes frames, queries VLM, parses goals |
| `SemanticGoal` | Dataclass: label, confidence, pixel hint, metric waypoint |
| `VLMPlannerConfig` | Model name, endpoint, confidence threshold, prompt template |

## Integration Points

- **Replaces / augments**: `contributions/10_human_language_ethics/` semantic goal generator
- **Feeds into**: A* planner, topological map (Contribution 17)
- **Combines with**: Contribution 19 (LLM Mission Planner) for full language pipeline

## Production Upgrade

Replace the HTTP stub with a proper VLM call:
```python
# Local (free, private)
config = VLMPlannerConfig(model_name="llava-1.6", endpoint="http://localhost:11434/api/chat")

# Cloud (GPT-4V)
config = VLMPlannerConfig(model_name="gpt-4-vision-preview", endpoint="https://api.openai.com/v1/chat/completions")
```
