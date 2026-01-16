# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

> A research-oriented framework for **autonomous robotic navigation under uncertainty**, integrating **risk-aware replanning**, **uncertainty-aware exploration**, and **learning-augmented planning**, with strong emphasis on **quantitative evaluation**, **ablation studies**, and **reproducibility**.

---

## Summary

This repository presents a unified research framework for autonomous robots operating in **unknown, partially observable, and dynamically evolving environments**.
The system explicitly reasons about **uncertainty, risk, and irreversibility**, and supports **dynamic replanning** augmented by learned heuristics while preserving formal planning guarantees.

The project is developed as an **individual research codebase** at the
**School of Electrical and Computer Engineering**,
**Democritus University of Thrace (D.U.Th.)**.

---

## Overview

Autonomous navigation in real-world environments is constrained by multiple sources of uncertainty:

* Incrementally built or incomplete maps
* Visual odometry drift and feature sparsity
* Dynamic obstacles requiring frequent replanning
* Trade-offs between optimality, safety, coverage, energy, connectivity, and computation

This framework addresses these challenges by combining:

* **Classical planning algorithms**
* **Probabilistic state estimation**
* **Risk- and belief-aware planning**
* **Learning-based heuristics**
* **Explicit modeling of uncertainty, risk, and irreversibility**

The result is a **modular, extensible, and research-grade navigation pipeline** suitable for systematic experimentation and comparative studies.

---

## Abstract

Autonomous navigation in unknown environments is fundamentally limited by uncertainty arising from sensing, state estimation, and environment dynamics.
This work introduces a navigation pipeline that explicitly models uncertainty and risk, supports dynamic replanning, and integrates learned heuristics into classical planners while maintaining formal guarantees.

The framework is validated through extensive quantitative evaluation, parameter sweeps, and ablation studies, with strong emphasis on experimental reproducibility.

---

## Repository Structure

The repository is organized **by research contributions**, rather than by monolithic modules:

```text
.
├── contributions/
│   ├── 01_learned_astar/
│   │   ├── code/          # training + learned heuristic utilities
│   │   ├── experiments/   # evaluation and benchmarks
│   │   ├── models/        # trained models (.pt / .npz)
│   │   └── results/       # CSV logs and plots
│   ├── 02_uncertainty_calibration/
│   ├── 03_belief_risk_planning/
│   ├── 04_irreversibility_returnability/
│   ├── 05_safe_mode_navigation/
│   ├── 06_energy_connectivity/
│   ├── 07_nbv_exploration/
│   ├── 08_security_ids/
│   ├── 09_multi_robot/
│   └── 10_human_language_ethics/
│
├── docs/                  # extended documentation
├── results/               # aggregated outputs
├── figures/               # publication-ready figures
├── research_results/      # curated experiment summaries
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

* Python ≥ 3.9

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

* Node expansions (learned vs classical)
* Path cost and optimality preservation
* Results under:

  * `contributions/01_learned_astar/results/`
  * `contributions/01_learned_astar/models/`

---

### A2) Irreversibility-Aware Navigation

Run experiments:

```bash
python contributions/04_irreversibility_returnability/experiments/run_irreversibility_bottleneck_sweep.py
python contributions/04_irreversibility_returnability/experiments/run_irreversibility_tau_sweep.py
python contributions/04_irreversibility_returnability/experiments/run_irreversibility_demo.py
```

**Outputs**

* Feasibility phase transitions vs threshold τ
* CSV logs and plots under:

  * `contributions/04_irreversibility_returnability/results/`

---

### A3) Risk-Weighted & Risk-Budget Planning

Run demo:

```bash
python contributions/03_belief_risk_planning/experiments/run_risk_budget_demo.py
```

Utilities:

```bash
python contributions/03_belief_risk_planning/experiments/select_lambda_for_risk_budget.py
python contributions/03_belief_risk_planning/experiments/select_path_under_risk_budget.py
```

**Outputs**

* CSV logs and plots under:

  * `contributions/03_belief_risk_planning/results/`

---

### A4) Innovation-Based IDS for UKF

Replay-based evaluation:

```bash
python contributions/08_security_ids/experiments/eval_ids_replay.py
```

---

### A5) TF Integrity IDS (CUSUM)

Sweep evaluation:

```bash
python contributions/08_security_ids/experiments/eval_ids_sweep.py
```

Calibration:

```bash
python contributions/08_security_ids/experiments/calibrate_tf_cusum.py
```

**Outputs**

* IDS logs and plots under:

  * `contributions/08_security_ids/results/`

---

### A6) Safe Mode Navigation

Demo:

```bash
python contributions/05_safe_mode_navigation/experiments/run_adaptive_tau_safe_mode_demo.py
```

Analysis:

```bash
python contributions/05_safe_mode_navigation/experiments/analyze_safe_mode_results.py
```

---

### A7) Energy & Connectivity-Aware Planning

```bash
python contributions/06_energy_connectivity/experiments/run_connectivity_sweep.py
python contributions/06_energy_connectivity/experiments/run_energy_connectivity_joint_sweep.py
python contributions/06_energy_connectivity/experiments/run_energy_risk_time_demo.py
```

---

### A8) NBV / Frontier Exploration

```bash
python contributions/07_nbv_exploration/experiments/run_nbv_frontier_demo.py
python contributions/07_nbv_exploration/experiments/run_nbv_irreversibility_demo.py
python contributions/07_nbv_exploration/experiments/run_nbv_random_vs_frontier_benchmark.py
```

---

### A9) Multi-Robot Experiments

```bash
python contributions/09_multi_robot/experiments/run_multi_robot_experiment.py
python contributions/09_multi_robot/experiments/run_multi_robot_risk_experiment.py
python contributions/09_multi_robot/experiments/analyze_multi_robot_risk_results.py
```

---

### A10) Human / Trust / Language / Ethics

Demos under:

```text
contributions/10_human_language_ethics/demos/
```

Example:

```bash
python contributions/10_human_language_ethics/demos/run_trust_and_preferences_demo.py
```

---

## Part B — ROS 2 & Gazebo Integration (Optional)

Requires:

* ROS 2
* Gazebo
* TurtleBot3 packages

See:

```text
docs/README_LiDAR_SLAM_TurtleBot3_ROS2.md
```

---

## Documentation

Extended research documentation under `docs/`, including:

* large-scale overviews
* formal contribution descriptions
* safe-mode and trust navigation analyses
* IDS and security-focused reports

---

## Key Research Contributions

1. Uncertainty-aware dynamic navigation with online replanning
2. Learned admissible A* heuristics with preserved optimality
3. Belief–risk planning with adaptive self-trust and risk budgets
4. Irreversibility- and returnability-aware navigation
5. Security-aware estimation and planning (UKF IDS, TF integrity)
6. Human-, language-, and ethics-aware extensions
7. Energy/connectivity-aware planning and adaptive safe modes
8. Multi-robot disagreement resolution and risk allocation

---

## Reproducibility

* Multi-seed experiments and parameter sweeps
* Ablation studies
* CSV logs and publication-ready plots

Primary artifacts are stored under:

* `contributions/*/results/`
* `contributions/*/models/`

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

**Panagiota Grosdouli**
Electrical & Computer Engineering
Democritus University of Thrace (D.U.Th.)

---

## Project Status

Actively developed research codebase.
Modules may evolve as part of ongoing experimentation and publications.
