# Contribution 19 — LLM Mission Planner

[![Module](https://img.shields.io/badge/Module-19-purple)](.) [![Type](https://img.shields.io/badge/Type-LLM%20%2F%20NLP-blue)](.) [![Tests](https://img.shields.io/badge/Tests-6%20passing-brightgreen)](.)

## Overview

Translates **natural language mission instructions** into structured waypoint sequences compatible with DynNav's planning backend. Supports local LLMs (Ollama/llama3) with keyword-extraction fallback for offline use.

## Research Question

> **RQ-LLM**: Can natural-language mission specifications reduce task-completion time vs. manual waypoint programming?

## How It Works

```
"Go to the kitchen, then check the corridor" 
    → LLM/keyword parser 
    → [Waypoint(kitchen, priority=1), Waypoint(corridor, priority=2)]
    → metric coordinates via zone map
    → A* execution
```

## Files

```
19_llm_mission_planner/
├── llm_mission_planner.py    # LLMMissionPlanner, Mission, Waypoint
└── experiments/
```

## Quick Start

```python
from contributions.19_llm_mission_planner.llm_mission_planner import LLMMissionPlanner

planner = LLMMissionPlanner()

# Works offline (keyword fallback — no LLM needed)
mission = planner.parse("go to the kitchen then the corridor and exit")

for wp in mission.waypoints:
    print(f"  {wp.priority}. {wp.label} ({wp.action})")

# Attach metric coordinates
zone_map = {"kitchen": (2.0, 3.0), "corridor": (5.0, 1.0), "exit": (8.0, 0.0)}
mission = planner.resolve_to_metric(mission, zone_map)
```

## Integration

- **Feeds into**: Contribution 17 (Topological Maps) for zone lookup
- **Combines with**: Contribution 11 (VLM) for vision-grounded instructions
- **Offline mode**: keyword fallback requires zero dependencies
