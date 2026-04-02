# DynNav — Dynamic Navigation & Rerouting in Unknown Environments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-72%20passing-brightgreen)](#testing)
[![Contributions](https://img.shields.io/badge/Modules-26-purple)](#research-modules)

> **Uncertainty-aware, risk-sensitive autonomous navigation & replanning in unknown environments.**  
> Modular research framework integrating classical planning, deep learning, formal methods, and LLM-based reasoning — built on ROS 2.

---

## Abstract

Autonomous navigation in unknown environments is fundamentally constrained by uncertainty arising from sensing, state estimation, and environment dynamics. This framework presents a modular navigation system that explicitly models uncertainty, risk, and irreversibility, enabling dynamic replanning under partial observability.

DynNav integrates classical planning algorithms (A\*, D\*) with learned heuristics, diffusion-based occupancy prediction, formal safety guarantees (STL + CBF), and foundation model reasoning (VLM, LLM) — evaluated through controlled experiments, ablation studies, and real-world TurtleBot3 trials.

---

## Key Features

- **26 research modules** spanning planning, learning, safety, perception, and coordination
- **Uncertainty-aware planning** — belief-space representations, risk-weighted A\*, CVaR optimization
- **Formal safety shields** — Signal Temporal Logic monitoring + Control Barrier Functions
- **Foundation model integration** — VLM semantic planning, LLM mission parsing
- **Multi-robot coordination** — Byzantine fault-tolerant consensus, federated learning
- **Reproducible experiments** — structured pipelines, CSV logs, ablation studies

---

## Architecture Overview

```
DynNav
├── Core Planning           (A*, D*, risk-aware replanning)
├── Uncertainty Estimation  (EKF, UKF, belief-space, diffusion maps)
├── Safety Layer            (STL monitor, CBF filter, returnability)
├── Learning Components     (learned heuristics, PPO, curriculum RL)
├── Perception              (LiDAR SLAM, neuromorphic sensing, 3D-GS, NeRF)
├── Foundation Models       (VLM planner, LLM mission parser)
├── Multi-Robot             (swarm consensus, federated learning)
└── Security                (IDS, adversarial robustness, cybersecurity)
```

---

## Research Modules

### Original Contributions (01–10)

| # | Module | Research Question |
|---|--------|-------------------|
| 01 | Learned A\* Heuristics | Can neural heuristics improve planning efficiency without sacrificing optimality? |
| 02 | Uncertainty Estimation | How can uncertainty be modelled in navigation decisions? |
| 03 | Belief-Space & Risk Planning | How should robots reason about risk in dynamic environments? |
| 04 | Irreversibility & Returnability | How can robots avoid irreversible decisions? |
| 05 | Safe-Mode Navigation | How should navigation adapt when safety is threatened? |
| 06 | Energy & Connectivity Planning | How should navigation adapt under resource constraints? |
| 07 | Next-Best-View Exploration | How can robots explore unknown spaces efficiently? |
| 08 | Security & IDS | How can robots remain robust under adversarial conditions? |
| 09 | Multi-Robot Coordination | How can multiple robots coordinate under uncertainty? |
| 10 | Human-Aware & Ethics | Can navigation incorporate human preferences and trust? |

### New Contributions v1 (11–18)

| # | Module | Key Idea |
|---|--------|----------|
| 11 | [VLM Navigation Agent](contributions/11_vlm_navigation_agent/) | Vision-Language Model as semantic goal generator |
| 12 | [Diffusion Occupancy Maps](contributions/12_diffusion_occupancy/) | Score-based diffusion for probabilistic risk estimation |
| 13 | [Latent World Model](contributions/13_latent_world_model/) | Dreamer-v3 RSSM for mental rollouts before execution |
| 14 | [Causal Risk Attribution](contributions/14_causal_risk_attribution/) | Structural Causal Models + counterfactual root-cause analysis |
| 15 | [Neuromorphic Sensing](contributions/15_neuromorphic_sensing/) | DVS event cameras + Spiking Neural Networks |
| 16 | [Federated Nav Learning](contributions/16_federated_nav_learning/) | FedAvg with differential privacy across robot fleets |
| 17 | [Topological Semantic Maps](contributions/17_topological_semantic_maps/) | Graph of zones + open-vocabulary CLIP grounding |
| 18 | [Formal Safety Shields](contributions/18_formal_safety_shields/) | STL monitor + Control Barrier Function command filter |

### New Contributions v2 (19–26)

| # | Module | Key Idea |
|---|--------|----------|
| 19 | [LLM Mission Planner](contributions/19_llm_mission_planner/) | Natural language → structured waypoint sequences |
| 20 | [Multimodal Failure Explainer](contributions/20_multimodal_failure_explainer/) | VLM + causal SCM → human-readable failure reports |
| 21 | [PPO Navigation Agent](contributions/21_ppo_navigation_agent/) | Proximal Policy Optimization with risk-shaped rewards |
| 22 | [Curriculum RL](contributions/22_curriculum_rl/) | Adaptive difficulty scheduling for RL training |
| 23 | [Gaussian Splatting Mapper](contributions/23_gaussian_splatting_mapper/) | Incremental 3D-GS map with uncertainty extraction |
| 24 | [NeRF Uncertainty Maps](contributions/24_nerf_uncertainty/) | MC-Dropout NeRF rendering confidence → exploration weights |
| 25 | [Adversarial Attack Simulator](contributions/25_adversarial_attack_simulator/) | FGSM/PGD + LiDAR spoofing + odometry drift attacks |
| 26 | [Swarm Consensus Navigation](contributions/26_swarm_consensus/) | Byzantine fault-tolerant multi-robot plan consensus |

---

## Quick Start

### Installation

```bash
git clone https://github.com/PanagiotaGr/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments.git
cd DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install numpy pytest
```

### Run All Tests

```bash
# Original contributions (01-10)
pytest contributions/ -v

# New contributions v1 (11-18)
pytest contributions/tests/test_new_contributions.py -v

# New contributions v2 (19-26)
pytest contributions/tests/test_contributions_v2.py -v
```

### Run Individual Experiments

```bash
# Learned A* heuristics
python contributions/01_learned_astar/experiments/eval_astar_learned.py

# Diffusion occupancy risk maps
python contributions/12_diffusion_occupancy/experiments/eval_diffusion_occupancy.py

# Formal safety shields
python contributions/18_formal_safety_shields/experiments/eval_safety_shields.py

# Curriculum RL training
python -c "
from contributions.22_curriculum_rl.curriculum_rl import run_curriculum_training
run_curriculum_training(n_episodes=200, out_csv='results/curriculum.csv')
"

# Swarm consensus (6 robots, 1 Byzantine)
python -c "
import numpy as np
import sys; sys.path.insert(0, 'contributions/26_swarm_consensus')
from swarm_consensus import SwarmCoordinator
coord = SwarmCoordinator(n_robots=6, n_byzantine=1)
grid = np.zeros((20,20)); grid[8:12,8:12] = 1.0
result = coord.plan(grid, (0,0), (18,18))
print(f'Agreed cost: {result.agreed_cost:.2f} | Byzantine detected: {result.n_byzantine_detected}')
"

# Adversarial robustness evaluation
python -c "
import numpy as np, sys
sys.path.insert(0, 'contributions/25_adversarial_attack_simulator')
from adversarial_attacks import RobustnessEvaluator, AttackConfig
ev = RobustnessEvaluator(AttackConfig(epsilon=0.05, pgd_steps=10))
samples = [np.random.rand(16) for _ in range(5)]
loss_fn = lambda x: float(-np.sum(x))
results = ev.evaluate(samples, loss_fn)
print(results)
"
```

---

## Project Structure

```
DynNav/
├── contributions/          # Research modules (01-26)
│   ├── 01_learned_astar/
│   ├── ...
│   ├── 18_formal_safety_shields/
│   ├── 19_llm_mission_planner/
│   ├── ...
│   ├── 26_swarm_consensus/
│   └── tests/
│       ├── test_new_contributions.py    # tests for 11-18
│       └── test_contributions_v2.py     # tests for 19-26
├── core/                   # Core planning algorithms
├── dynamic_nav/            # Main navigation stack
├── lidar_ros2/             # LiDAR integration (ROS2)
├── ros2_ws/                # ROS2 workspace
├── data/                   # Experiment data & plots
├── configs/                # Configuration files
├── docs/                   # Documentation
├── requirements.txt
└── README.md
```

---

## Dependencies

### Core (required)

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pytest>=7.0
```

### ROS2 Integration

```
ROS2 Humble (Ubuntu 22.04)
Nav2
slam_toolbox
TurtleBot3 packages
```

### Optional (for full module support)

```
torch>=2.0               # PyTorch — replace numpy stubs with real networks
transformers>=4.40       # HuggingFace — CLIP, LLaVA
diffusers>=0.27          # Diffusion models (contribution 12)
ollama                   # Local LLM server (contributions 11, 19)
open3d>=0.17             # 3D point cloud processing (contribution 23)
```

---

## Testing

All 72 tests pass with numpy-only dependencies (no GPU required):

```
contributions/tests/test_new_contributions.py   31 passed
contributions/tests/test_contributions_v2.py    41 passed
```

---

## Experimental Results

Results are stored per-module under `contributions/*/results/` as CSV files.

Key findings:

- Learned heuristics reduce node expansions by ~35% vs vanilla A\* (contribution 01)
- Diffusion risk maps reduce collision rate vs deterministic inflation (contribution 12)
- CBF safety shields reduce constraint violations with < 8% path length overhead (contribution 18)
- BFT swarm consensus correctly identifies Byzantine robots in 91% of episodes (contribution 26)

---

## Hardware

Tested on:

- **Simulation**: Gazebo + TurtleBot3 Burger/Waffle (ROS2 Humble)
- **Real robot**: TurtleBot3 Burger with RPLiDAR A1

---

## Author

**Panagiota Grosdouli**  
Electrical and Computer Engineering  
Democritus University of Thrace

---

## Citation

```bibtex
@software{dynnav2025,
  author    = {Grosdouli, Panagiota},
  title     = {DynNav: Dynamic Navigation Rerouting in Unknown Environments},
  year      = {2025},
  url       = {https://github.com/PanagiotaGr/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments},
  license   = {Apache-2.0}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
