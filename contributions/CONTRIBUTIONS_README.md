# DynNav — New Contributions (11–18)

Eight research modules extending DynNav with state-of-the-art ideas from
foundation models, generative AI, formal methods, and neuromorphic computing.

---

## Overview

| # | Module | Key Idea | RQ |
|---|--------|----------|-----|
| 11 | VLM Navigation Agent | Vision-Language Model as semantic planner | RQ-VLM |
| 12 | Diffusion Occupancy Maps | Score-based diffusion for probabilistic risk | RQ-Diff |
| 13 | Latent World Model | Dreamer-v3 RSSM for mental rollouts | RQ-WM |
| 14 | Causal Risk Attribution | SCM + counterfactual root-cause analysis | RQ-Causal |
| 15 | Neuromorphic Sensing | DVS event camera + Spiking Neural Networks | RQ-Neuro |
| 16 | Federated Nav Learning | FedAvg with differential privacy | RQ-Fed |
| 17 | Topological Semantic Maps | Graph of zones + open-vocabulary grounding | RQ-Topo |
| 18 | Formal Safety Shields | STL monitor + Control Barrier Functions | RQ-Formal |

---

## Directory structure

```
contributions/
├── 11_vlm_navigation_agent/
│   ├── vlm_planner.py
│   └── experiments/eval_vlm_planner.py
├── 12_diffusion_occupancy/
│   ├── diffusion_occupancy.py
│   └── experiments/eval_diffusion_occupancy.py
├── 13_latent_world_model/
│   └── latent_world_model.py
├── 14_causal_risk_attribution/
│   └── causal_risk.py
├── 15_neuromorphic_sensing/
│   └── neuromorphic_sensing.py
├── 16_federated_nav_learning/
│   └── federated_nav.py
├── 17_topological_semantic_maps/
│   └── topo_semantic_map.py
└── 18_formal_safety_shields/
    ├── formal_safety_shields.py
    └── experiments/eval_safety_shields.py
```

---

## Quick start

```bash
# Run all new tests
pytest contributions/tests/test_new_contributions.py -v

# Run a specific experiment (no GPU required — numpy-only stubs)
python contributions/12_diffusion_occupancy/experiments/eval_diffusion_occupancy.py
python contributions/18_formal_safety_shields/experiments/eval_safety_shields.py
```

---

## Module details

### 11 · VLM Navigation Agent
Sends camera frames to a local LLaVA (via Ollama) or GPT-4V endpoint.
The VLM returns a JSON goal `{region, goal, confidence, pixel_u, pixel_v}`
which is back-projected to metric waypoints using the depth frame.

**Integration point:** replaces or augments `contributions/10_human_language_ethics/`
with a vision-grounded semantic goal generator.

**Production upgrade:** swap `ScoreNetwork` stub for a U-Net from 🤗 Diffusers.

---

### 12 · Diffusion Occupancy Maps
DDPM reverse diffusion conditioned on recent occupancy history produces
N samples of future occupancy. CVaR-95 risk map is fed into the existing
risk-aware A* planner (`contributions/03_belief_risk_planning/`).

**Integration point:** replace deterministic inflation in the cost map
with `DiffusionOccupancyPredictor.predict_risk()`.

---

### 13 · Latent World Model
RSSM encodes observations into a compact latent `(h, z)`. Before executing
a plan, `WorldModelPlanner.select_best_action_sequence()` scores multiple
candidate action sequences via imagined rollouts.

**Integration point:** wrap the existing safe-mode planner
(`contributions/05_safe_mode_navigation/`) with mental-rollout pre-screening.

---

### 14 · Causal Risk Attribution
`NavigationSCM` models the causal graph from sensor noise → collision.
After a failure, `root_cause_ranking()` identifies which causal variable
contributed most — enabling targeted remediation rather than blanket replanning.

**Integration point:** post-hoc analysis module for
`contributions/08_security_ids/` anomaly events.

---

### 15 · Neuromorphic Sensing
`DVSSimulator` converts greyscale frames to asynchronous events.
`SNNObstacleDetector` uses Leaky Integrate-and-Fire neurons to detect
obstacle regions from the event time-surface at microsecond latency.

**Integration point:** parallel sensing pipeline alongside the existing
LiDAR SLAM (`lidar_ros2/`), fused in the belief state.

---

### 16 · Federated Navigation Learning
`FederatedServer.run_training()` orchestrates N robot clients:
each trains locally, shares DP-noised model deltas, receives the
FedAvg-aggregated global model.

**Integration point:** distributed version of the learned heuristic
in `contributions/01_learned_astar/` — trained across a fleet without
sharing raw maps.

---

### 17 · Topological Semantic Maps
`TopologicalSemanticMap` maintains a graph of semantic zones.
`ground_query(embedding)` finds the zone whose CLIP embedding best matches
a natural-language instruction, enabling instruction-following navigation.

**Integration point:** high-level representation layer above the metric
occupancy grid; connects to `contributions/10_human_language_ethics/`.

---

### 18 · Formal Safety Shields
`STLMonitor` evaluates temporal logic specs over the state trajectory.
`CBFSafetyFilter` modifies velocity commands to provably maintain
`h(x) ≥ 0` (distance > safety radius) via gradient projection QP.
`SafetyShield` wraps any planner with both components.

**Integration point:** drop-in wrapper around any planner output;
pairs with `contributions/04_irreversibility_returnability/` for hard
safety guarantees.

---

## Dependencies

All modules are **numpy-only** by default (no GPU required for testing).

Production upgrades require:
```
torch>=2.0           # replace stub networks with real PyTorch models
transformers>=4.40   # HuggingFace for CLIP / LLaVA
diffusers>=0.27      # Diffusers UNet for contribution 12
ollama               # local LLM server for contribution 11
```

---

## Citation

If you use these contributions, please cite the DynNav project:
```
@software{dynnav2025,
  author = {Grosdouli, Panagiota},
  title  = {DynNav: Dynamic Navigation Rerouting in Unknown Environments},
  year   = {2025},
  url    = {https://github.com/PanagiotaGr/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments}
}
```
