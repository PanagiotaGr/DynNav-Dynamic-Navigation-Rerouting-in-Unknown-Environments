# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

## Summary
A research-oriented navigation framework for autonomous robots operating in unknown and evolving environments under uncertainty.  
The framework integrates **risk-aware replanning**, **uncertainty-aware exploration**, and **learning-augmented planning**, supported by extensive quantitative evaluation, ablation studies, and reproducible experiments.

---

## Overview
This repository presents a unified research framework for **autonomous robotic navigation under uncertainty**.

It targets key real-world challenges:
- Partial observability and incrementally built maps  
- Visual odometry drift and feature sparsity  
- Dynamic obstacles requiring frequent replanning  
- Trade-offs between optimality, safety, coverage, energy/connectivity, and computational cost  

The pipeline combines:
- Classical planning algorithms  
- Probabilistic state estimation  
- Learning-based heuristics  
- Explicit modeling of uncertainty, risk, and irreversibility  

Developed as an **individual research project** at the  
School of Electrical and Computer Engineering,  
Democritus University of Thrace (D.U.Th.).

---

## Abstract
Autonomous navigation in unknown environments is fundamentally constrained by uncertainty arising from sensing, estimation, and environment dynamics.  
This work introduces a navigation pipeline that explicitly reasons about uncertainty and risk, supports dynamic replanning, and integrates learned heuristics into classical planners while preserving formal guarantees.

The framework is validated through quantitative evaluation, parameter sweeps, and ablation studies, with emphasis on experimental reproducibility.

---

## Repository Structure
The repository is organized **by research contributions**:

```text
.
├── contributions/
│   ├── 01_learned_astar/
│   │   ├── code/          # training + learned heuristic utilities
│   │   ├── experiments/   # evaluation/benchmarks
│   │   ├── models/        # trained .pt / .npz
│   │   └── results/       # plots/csv outputs
│   ├── 02_uncertainty_calibration/
│   ├── 03_belief_risk_planning/
│   ├── 04_irreversibility_returnability/
│   ├── 05_safe_mode_navigation/
│   ├── 06_energy_connectivity/
│   ├── 07_nbv_exploration/
│   ├── 08_security_ids/
│   ├── 09_multi_robot/
│   ├── 10_human_language_ethics/
│      
├── docs/                  
├── results/              
├── figures/               
├── research_results/    
├── requirements.txt
└── CITATION.cff
```

---

## Installation
Recommended setup using a clean Python environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**Requirements**
- Python ≥ 3.9

---

## Part A — Offline Experiments (No ROS Required)

### A1) Learned A* vs Classical A*
Run evaluation:
```bash
python contributions/01_learned_astar/experiments/eval_astar_learned.py
```

Retrain heuristic:
```bash
python contributions/01_learned_astar/code/train_heuristic.py
```

**Outputs**
- Node expansions (learned vs classical)
- Path cost / optimality preservation
- Artifacts under:
  - `contributions/01_learned_astar/results/`
  - `contributions/01_learned_astar/models/`

---

### A2) Irreversibility-Aware Navigation
Run sweeps / demos:
```bash
python contributions/04_irreversibility_returnability/experiments/run_irreversibility_bottleneck_sweep.py
python contributions/04_irreversibility_returnability/experiments/run_irreversibility_tau_sweep.py
python contributions/04_irreversibility_returnability/experiments/run_irreversibility_demo.py
```

**Outputs**
- Feasibility phase transition vs threshold τ
- CSV logs + plots under:
  - `contributions/04_irreversibility_returnability/results/`

---

### A3) Risk-Weighted Planning (λ-sweep) & Risk-Budget Planning
Run risk-budget demo:
```bash
python contributions/03_belief_risk_planning/experiments/run_risk_budget_demo.py
```

Risk-budget utilities:
```bash
python contributions/03_belief_risk_planning/experiments/select_lambda_for_risk_budget.py
python contributions/03_belief_risk_planning/experiments/select_path_under_risk_budget.py
```

**Outputs**
- CSV logs + plots under:
  - `contributions/03_belief_risk_planning/results/`

---

### A4) Innovation-Based IDS for UKF
Replay-based IDS evaluation:
```bash
python contributions/08_security_ids/experiments/eval_ids_replay.py
```

---

### A5) TF Integrity IDS (CUSUM)
Sweep evaluation:
```bash
python contributions/08_security_ids/experiments/eval_ids_sweep.py
```

Calibration (CUSUM):
```bash
python contributions/08_security_ids/experiments/calibrate_tf_cusum.py
```

**Outputs**
- IDS logs + plots under:
  - `contributions/08_security_ids/results/`

---

### A6) Safe Mode Navigation
Adaptive safe mode demo:
```bash
python contributions/05_safe_mode_navigation/experiments/run_adaptive_tau_safe_mode_demo.py
```

Analysis:
```bash
python contributions/05_safe_mode_navigation/experiments/analyze_safe_mode_results.py
```

**Outputs**
- under `contributions/05_safe_mode_navigation/results/`

---

### A7) Energy + Connectivity-Aware Planning
Connectivity sweep:
```bash
python contributions/06_energy_connectivity/experiments/run_connectivity_sweep.py
```

Joint sweep:
```bash
python contributions/06_energy_connectivity/experiments/run_energy_connectivity_joint_sweep.py
```

Energy-risk-time demo:
```bash
python contributions/06_energy_connectivity/experiments/run_energy_risk_time_demo.py
```

**Outputs**
- under `contributions/06_energy_connectivity/results/`

---

### A8) NBV / Frontier Exploration
Benchmark / demos:
```bash
python contributions/07_nbv_exploration/experiments/run_nbv_frontier_demo.py
python contributions/07_nbv_exploration/experiments/run_nbv_irreversibility_demo.py
python contributions/07_nbv_exploration/experiments/run_nbv_random_vs_frontier_benchmark.py
```

**Outputs**
- under `contributions/07_nbv_exploration/results/`

---

### A9) Multi-Robot Experiments
Run multi-robot experiments:
```bash
python contributions/09_multi_robot/experiments/run_multi_robot_experiment.py
python contributions/09_multi_robot/experiments/run_multi_robot_risk_experiment.py
```

Analysis:
```bash
python contributions/09_multi_robot/experiments/analyze_multi_robot_risk_results.py
```

**Outputs**
- under `contributions/09_multi_robot/results/`

---

### A10) Human / Trust / Language / Ethics Demos
Demos live under:
- `contributions/10_human_language_ethics/demos/`

Example:
```bash
python contributions/10_human_language_ethics/demos/run_trust_and_preferences_demo.py
```

---

## Part B — ROS 2 & Gazebo Integration (Optional)
Requires:
- ROS 2
- Gazebo
- TurtleBot3 packages

See:
- `docs/README_LiDAR_SLAM_TurtleBot3_ROS2.md`

---

## Documentation
Extended research documentation is under `docs/`, including:
- `docs/README-large info.md`
- `docs/Abstract_and_Contributions.md`
- `docs/Irreversibility_Aware_Navigation_New_Contribution.md`
- `docs/README_safe_mode_experiments.md`
- `docs/README_trust_navigation.md`
- `docs/README_Innovation-Based_IDS_for_UKF_Sensor_Fusion.md`
- `docs/README_TF_Attack_Aware_IDS.md`

---

## Key Research Contributions
1. Uncertainty-aware dynamic navigation with online replanning  
2. Learned admissible A* heuristics with preserved optimality  
3. Belief–risk planning with adaptive self-trust / risk budgets  
4. Irreversibility- and returnability-aware navigation  
5. Security-aware estimation and planning hooks (UKF IDS, TF integrity)  
6. Human-, language-, and ethics-aware extensions  
7. Energy/connectivity-aware planning and safe mode policies  
8. Multi-robot disagreement and risk allocation under uncertainty  

---

## Reproducibility
- Multi-seed experiments and sweeps  
- Ablation studies  
- CSV logs and plots per contribution  

Primary artifacts are stored under:
- `contributions/*/results/`
- `contributions/*/models/` (when applicable)

---

## Citation
If you use this work, please cite:
```text
See CITATION.cff
```

---

## License
Apache License 2.0

---

## Disclaimer
For research and educational use only.  
Not validated for safety-critical deployment.

---

## Author
Panagiota Grosdouli  
Electrical & Computer Engineering  
Democritus University of Thrace (D.U.Th.)


-----
## Project Status
Actively developed research codebase.  
Modules may evolve as part of ongoing experimentation and publications.


