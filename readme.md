<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=DynNav&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Dynamic%20Navigation%20%26%20Rerouting%20in%20Unknown%20Environments&descAlignY=55&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![TurtleBot3](https://img.shields.io/badge/TurtleBot3-Burger-FF6B35?style=for-the-badge&logo=robotframework&logoColor=white)](https://www.turtlebot.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-4CAF50?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-72%20passing-brightgreen?style=for-the-badge&logo=pytest)](.)
[![Modules](https://img.shields.io/badge/Modules-26-9C27B0?style=for-the-badge&logo=buffer)](contributions/)
[![Stars](https://img.shields.io/github/stars/PanagiotaGr/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments?style=for-the-badge&color=yellow)](.)

<br/>

**A robot that thinks before it moves.**  
*Uncertainty-aware · Risk-sensitive · Formally safe · Learning-augmented*

<br/>

[🚀 Quick Start](#-quick-start) · [🎥 Demo](#-demo) · [🔬 Research](#-research-modules) · [📊 Results](#-results) · [📖 Docs](#-documentation) · [🤝 Citation](#-citation)

<br/>

</div>

---

## 🎥 Demo

> **The robot navigates a completely unknown environment, detects obstacles in real time, re-routes when its path is blocked, and stops before entering unsafe areas — all autonomously.**

<div align="center">

### 🤖 Live Navigation Demo



</div>

### What you're seeing

| Moment | What DynNav does |
|--------|-----------------|
| Robot starts moving | A* planner computes optimal path; CBF safety shield activates |
| Obstacle appears | LiDAR detects → occupancy map updates → replanning triggered in < 100ms |
| Risk increases | Safe-mode activates: robot slows down, inflation radius grows |
| Path blocked | Topological map invalidates edge → Dijkstra finds alternative route |
| Goal reached | Mission logged, failure explainer generates episode report |

---

## 🤔 What Problem Does This Solve?

Real robots navigating real buildings face problems that textbook algorithms ignore:

```
❌  "The map is complete"          →  ✅  DynNav builds the map AS it navigates
❌  "Sensors are perfect"          →  ✅  DynNav models and uses uncertainty explicitly  
❌  "The environment is static"    →  ✅  DynNav replans in real time when things change
❌  "Safety = avoid obstacles"     →  ✅  DynNav uses formal guarantees (STL + CBF)
❌  "One robot at a time"          →  ✅  DynNav coordinates fleets with fault tolerance
❌  "Program coordinates to go"    →  ✅  DynNav understands "go to the kitchen"
```

---

## ⚡ Quick Start

### 1. Install

```bash
git clone https://github.com/PanagiotaGr/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments.git
cd DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy pytest
```

### 2. Run everything in 30 seconds

```bash
# Run all 26 modules — no GPU, no ROS2 required
python run_all_contributions.py --quick
```

You'll see output like:
```
[01/16] Module 11: 11_vlm_navigation_agent
        OK in 0.17s — {'encoded_frames': 3, 'parse_ok': True, 'confidence': 0.85}
[02/16] Module 12: 12_diffusion_occupancy  
        OK in 0.01s — {'mean_cvar': 0.85, 'path_cost': 20.0}
...
Results: 16 OK | 0 ERROR | 1.5s total
```

### 3. Run specific experiments

```bash
# Safety shields: how many collisions does the CBF prevent?
python contributions/18_formal_safety_shields/experiments/eval_safety_shields.py \
    --n_episodes 50

# Diffusion risk maps
python contributions/12_diffusion_occupancy/experiments/eval_diffusion_occupancy.py \
    --n_scenarios 30

# Try the swarm consensus (6 robots, 1 hacked)
python -c "
import numpy as np, sys
sys.path.insert(0, 'contributions/26_swarm_consensus')
from swarm_consensus import SwarmCoordinator
coord = SwarmCoordinator(n_robots=6, n_byzantine=1)
grid = np.zeros((20,20)); grid[8:12,8:12] = 1.0
result = coord.plan(grid, (0,0), (18,18))
print(f'✅ Agreed cost: {result.agreed_cost:.2f}')
print(f'🚨 Byzantine robots detected: {result.n_byzantine_detected}')
print(f'📍 Path length: {len(result.agreed_path)} steps')
"
```

### 4. With ROS2 + TurtleBot3

```bash
# Terminal 1: launch Gazebo simulation
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: start DynNav
ros2 launch launch/dynnav_full.launch.py

# Terminal 3: send a mission in plain English
python -c "
from contributions.19_llm_mission_planner.llm_mission_planner import LLMMissionPlanner
planner = LLMMissionPlanner()
mission = planner.parse('go to the kitchen, then check the corridor, then return to start')
print('Mission created:')
for wp in mission.waypoints:
    print(f'  {wp.priority}. {wp.label} ({wp.action})')
"
```

---

## 🏗️ How It Works

<div align="center">

```
  You say:  "Go to the kitchen"
                    │
                    ▼
     ┌─────────────────────────┐
     │   LLM Mission Planner   │  "kitchen" → waypoint coordinates
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │   Topological Map       │  Find "kitchen" in semantic graph
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │   Risk-Aware A*         │  Plan path avoiding dangerous areas
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │   Safety Shield         │  Filter every command: STL + CBF
     └────────────┬────────────┘
                  │
                  ▼
          🤖 Robot moves safely
```

</div>

---

## 🔬 Research Modules

DynNav has **26 research modules** across 8 areas. Click any to learn more:

### 🎯 Core Planning

| Module | What it does | Key idea |
|--------|-------------|----------|
| [01 Learned A*](contributions/01_learned_astar/) | Faster pathfinding | Neural heuristic → 35% fewer node expansions |
| [03 Risk Planning](contributions/03_belief_risk_planning/) | Avoids dangerous paths | CVaR optimisation — plans for worst-case scenarios |
| [07 Exploration](contributions/07_next_best_view/) | Maps unknown spaces | Goes where it will learn the most |

### 🔒 Safety (The robot won't hurt you)

| Module | What it does | Key idea |
|--------|-------------|----------|
| [04 Returnability](contributions/04_irreversibility_returnability/) | No dead-ends | Checks "can I get back?" before moving |
| [05 Safe-Mode](contributions/05_safe_mode_navigation/) | Slows when things get risky | Automatic speed reduction + replanning |
| [18 Safety Shields ⭐](contributions/18_formal_safety_shields/) | **Guaranteed** safety | STL logic + Control Barrier Functions — mathematically proven |

### 🧠 Intelligence

| Module | What it does | Key idea |
|--------|-------------|----------|
| [11 VLM Agent](contributions/11_vlm_navigation_agent/) | Sees and understands scenes | GPT-4V / LLaVA → "I see a corridor, go left" |
| [13 World Model](contributions/13_latent_world_model/) | Thinks before acting | Imagines 12 steps ahead before moving |
| [14 Causal AI](contributions/14_causal_risk_attribution/) | Learns from failures | Counterfactual: "the crash happened *because* of sensor noise" |
| [19 LLM Planner](contributions/19_llm_mission_planner/) | Understands plain English | "go to the kitchen" → navigation plan |

### 📊 Probabilistic Perception

| Module | What it does | Key idea |
|--------|-------------|----------|
| [02 Uncertainty](contributions/02_uncertainty_estimation/) | Knows what it doesn't know | EKF/UKF belief state |
| [12 Diffusion Maps](contributions/12_diffusion_occupancy/) | Predicts future obstacles | Generative AI for occupancy prediction |
| [15 Event Camera](contributions/15_neuromorphic_sensing/) | Ultra-fast obstacle detection | DVS + Spiking Neural Network at μs latency |
| [23 3D Mapping](contributions/23_gaussian_splatting_mapper/) | 3D scene understanding | Gaussian Splatting from robot camera |
| [24 NeRF Uncertainty](contributions/24_nerf_uncertainty/) | Knows unexplored areas | Neural Radiance Fields → exploration guide |

### 🤝 Multi-Robot

| Module | What it does | Key idea |
|--------|-------------|----------|
| [09 Coordination](contributions/09_multi_robot/) | Fleet navigation | No collisions between robots |
| [16 Federated Learning](contributions/16_federated_nav_learning/) | Robots learn together, privately | FedAvg + differential privacy |
| [26 Swarm Consensus ⭐](contributions/26_swarm_consensus/) | Fleet decisions even if robots are hacked | Byzantine fault tolerance: 1-in-3 robots can be compromised |

### 🛡️ Security

| Module | What it does | Key idea |
|--------|-------------|----------|
| [08 IDS](contributions/08_security_ids/) | Detects hacking attempts | χ²-test on sensor anomalies |
| [25 Attack Simulator](contributions/25_adversarial_attack_simulator/) | Tests robustness | FGSM/PGD + LiDAR spoofing attacks |

### 🎓 Learning

| Module | What it does | Key idea |
|--------|-------------|----------|
| [21 PPO Agent](contributions/21_ppo_navigation_agent/) | Learns to navigate | Reinforcement learning with risk-shaped rewards |
| [22 Curriculum RL](contributions/22_curriculum_rl/) | Trains efficiently | Starts easy, gets harder automatically |

### 👁️ Human-Aware

| Module | What it does | Key idea |
|--------|-------------|----------|
| [10 Ethics](contributions/10_human_language_ethics/) | Respects humans and spaces | Ethical zones + social distancing |
| [17 Semantic Maps](contributions/17_topological_semantic_maps/) | Understands place names | "kitchen", "corridor", "exit" as navigation goals |
| [20 Failure Explainer](contributions/20_multimodal_failure_explainer/) | Explains what went wrong | "The crash happened because sensor noise was too high" |

---

## 📊 Results

### Safety Shields Stop Collisions

```
Without safety shield:   ████████████████████  4.2 violations/episode
With STL + CBF shield:   █░░░░░░░░░░░░░░░░░░░  0.3 violations/episode
                                              ↑
                                       93% reduction
Path overhead: only +7.8%  ✅
```

### Swarm Consensus Handles Hacked Robots

```
6 robots, 1 is Byzantine (hacked/faulty):

Naive majority vote:    correctly identifies Byzantine:  60%
BFT weighted median:    correctly identifies Byzantine:  91%  ✅
Correct plan selected:  96%  ✅

Theory guarantees: can tolerate up to ⌊(n-1)/3⌋ Byzantine robots
```

### Federated Learning — Privacy-Preserving Fleet Training

```
Round  1: Val MSE = 0.374  (6 robots, no shared raw data)
Round 10: Val MSE = 0.361
Round 20: Val MSE = 0.360  ✅ convergence achieved
```

### Curriculum RL — Smarter Training

```
Flat training (no curriculum):  final success rate  23%
Adaptive curriculum:            final success rate  61%  ✅
                                       ↑
                              easy → medium → hard → expert
```

---

## 🗂️ Project Structure

```
DynNav/
│
├── 📦 contributions/           26 research modules
│   ├── 01_learned_astar/       Each has: code + experiments/ + results/ + README.md
│   ├── 02_uncertainty_estimation/
│   ├── ...
│   ├── 18_formal_safety_shields/   ⭐ Strongest formal methods module
│   ├── ...
│   └── 26_swarm_consensus/         ⭐ Byzantine fault-tolerant fleet decisions
│
├── 🤖 dynamic_nav/             Core navigation stack
├── 📡 lidar_ros2/              LiDAR + SLAM integration (ROS2)
├── 🛡️ cybersecurity_ros2/      Security monitoring nodes
├── 🗺️ ig_explorer/             Information-gain exploration
├── 🧠 neural_uncertainty/      Uncertainty estimation
├── 📸 photogrammetry_module/   3D photogrammetry
├── 🚀 ros2_ws/                 ROS2 workspace
│
├── ▶️ run_all_contributions.py  Run all 26 modules in ~2 seconds
├── 📊 TECHNICAL_REPORT.md      Full academic analysis
├── 📋 ethical_zones.json       No-go zone definitions
└── 📖 README.md                You are here
```

---

## 🛠️ Installation

### Minimal (Python only — no robot needed)

```bash
pip install numpy pytest
python run_all_contributions.py --quick  # runs in ~2 seconds
```

### Full stack (with ROS2)

```bash
# Ubuntu 22.04 required for ROS2 Humble
sudo apt install ros-humble-desktop ros-humble-nav2-bringup
sudo apt install ros-humble-turtlebot3*

pip install -r requirements.txt
```

### Optional (for full AI capabilities)

```bash
pip install torch>=2.0            # GPU training
pip install transformers>=4.40    # CLIP, LLaVA
pip install diffusers>=0.27       # Diffusion models
pip install open3d>=0.17          # 3D point clouds
ollama pull llava                 # Local LLM (free)
```

---

## 🧪 Testing

```bash
# All tests (72 total)
pytest contributions/tests/ -v

# Just the new modules
pytest contributions/tests/test_new_contributions.py -v   # modules 11-18
pytest contributions/tests/test_contributions_v2.py -v    # modules 19-26

# Run everything
python run_all_contributions.py
```

Expected output: `72 passed` ✅

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Full academic analysis of all 26 contributions with math |
| [CONTRIBUTIONS_README.md](contributions/CONTRIBUTIONS_README.md) | Quick overview of new modules |
| `contributions/*/README.md` | Individual README per module |
| `contributions/*/experiments/` | Runnable experiment scripts |
| `contributions/*/results/` | CSV output files |

---

## 🤖 Hardware

Built and tested on:

| Platform | Status |
|----------|--------|
| TurtleBot3 Burger (real robot) | ✅ Tested |
| Gazebo simulation (ROS2 Humble) | ✅ Tested |
| Ubuntu 22.04 | ✅ Supported |
| WSL2 (Windows) | ✅ Supported |
| Python-only (no ROS2) | ✅ All modules run |

---

## 🎓 Academic Context

DynNav is a research framework developed at the **Electrical & Computer Engineering Department, Democritus University of Thrace**.

It investigates how the following can coexist in a single navigation system:
- Formal safety guarantees (STL + CBF)
- Probabilistic uncertainty reasoning (diffusion models, EKF)
- Causal failure attribution (structural causal models)
- Multi-agent robustness (Byzantine fault tolerance)

The strongest contributions are:

| Rank | Contribution | Why it matters |
|------|-------------|----------------|
| ⭐⭐⭐ | [Formal Safety Shields](contributions/18_formal_safety_shields/) | Mathematically proven safety — unique in this stack |
| ⭐⭐⭐ | [Causal Risk Attribution](contributions/14_causal_risk_attribution/) | Novel: do-calculus for navigation failure diagnosis |
| ⭐⭐ | [CVaR Risk Planning](contributions/03_belief_risk_planning/) | Clean math, foundational for the whole system |
| ⭐⭐ | [Diffusion Occupancy](contributions/12_diffusion_occupancy/) | Emerging research direction with correct DDPM |
| ⭐⭐ | [BFT Swarm Consensus](contributions/26_swarm_consensus/) | Provably correct multi-robot fault tolerance |

Full academic analysis: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

---

## 🤝 Citation

If you use DynNav in your research:

```bibtex
@software{dynnav2025,
  author    = {Grosdouli, Panagiota},
  title     = {{DynNav}: Dynamic Navigation Rerouting in Unknown Environments},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/PanagiotaGr/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments},
  license   = {Apache-2.0},
  note      = {26-module framework: uncertainty-aware, risk-sensitive, formally safe autonomous navigation}
}
```

---

## 📬 Contact

**Panagiota Grosdouli**  
Electrical & Computer Engineering  
Democritus University of Thrace

---

## 📄 License

Copyright 2025 Panagiota Grosdouli — Apache License 2.0 — see [LICENSE](LICENSE)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**If DynNav helped you, consider giving it a ⭐**

*Built with curiosity. Tested with rigor. Shared openly.*

</div>
