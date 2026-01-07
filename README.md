
# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

## TL;DR
A research-oriented navigation framework for autonomous robots operating in unknown and evolving environments under uncertainty.  
The framework integrates **risk-aware replanning**, **uncertainty-aware exploration**, and **learning-augmented planning**, supported by extensive quantitative evaluation and ablation studies.

---

## Overview
This repository presents a unified research framework for **autonomous robotic navigation under uncertainty**.

It addresses key challenges in real-world robotic systems:
- Partial observability and incrementally built maps  
- Visual odometry drift and feature sparsity  
- Dynamic obstacles requiring frequent replanning  
- Trade-offs between optimality, safety, coverage, and computational cost  

The proposed pipeline combines:
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

The framework is validated through extensive quantitative evaluation, parameter sweeps, and ablation studies, with full experimental reproducibility.

---

## Repository Structure
```
.
├── eval_*.py                     # Evaluation scripts
├── train_*.py                    # Learning / heuristic training
├── datasets/                     # Planner datasets (.npz)
├── results/                      # CSV logs and metrics
├── figures/                      # Generated plots
├── research_results/             # Aggregated experimental artifacts
├── README-large_info.md           # Full technical documentation
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
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Requirements**
- Python ≥ 3.9

---

## Part A — Offline Experiments (No ROS Required)

### A1) Learned A* vs Classical A*
```bash
python eval_astar_learned.py
```

**Outputs**
- Node expansions (learned vs classical)
- Path cost (optimality preserved)
- Optional plots under `figures/` or `results/`

Retraining:
```bash
python train_heuristic.py
```

---

### A2) Irreversibility-Aware Navigation
```bash
python run_irreversibility_bottleneck_sweep.py
```

**Outputs**
- Feasibility phase transition vs threshold τ
- CSV logs and plots

---

### A3) Risk-Weighted Planning (λ-sweep)
```bash
python run_risk_weighted_lambda_sweep.py
python plot_belief_risk_lambda_sweep.py
```

---

### A4) Innovation-Based IDS for UKF
```bash
python eval_ids_replay.py
python plot_ids_from_csv.py --csv ids_replay_log.csv
```

---

### A5) TF Integrity IDS (CUSUM)
```bash
python eval_ids_sweep.py
```

---

## Part B — ROS 2 & Gazebo Integration (Optional)
Requires:
- ROS 2
- Gazebo
- TurtleBot3 packages

See `README_LiDAR_SLAM_TurtleBot3_ROS2.md`.

---

## Documentation
The repository includes extensive research documentation:
- README-large_info.md  
- Abstract_and_Contributions.md  
- Irreversibility_Aware_Navigation_New_Contribution.md  
- README_safe_mode_experiments.md  
- README_trust_navigation.md  
- README_Innovation-Based_IDS_for_UKF_Sensor_Fusion.md  
- README_TF_Attack_Aware_IDS.md  

---

## Key Research Contributions
1. Uncertainty-aware dynamic navigation with online replanning  
2. Learned admissible A* heuristics with preserved optimality  
3. Belief–risk planning with adaptive self-trust  
4. Irreversibility- and returnability-aware navigation  
5. Security-aware estimation and planning hooks  
6. Human-, language-, and ethics-aware extensions  

---

## Reproducibility
- Multi-seed experiments  
- Ablation studies  
- CSV logs and plots  

Results are stored under `results/`, `figures/`, and `research_results/`.

---

## Citation
If you use this work, please cite:
```
See CITATION.cff
```

---

## License
Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International.

---

## Disclaimer
For research and educational use only.  
Not validated for safety-critical deployment.

---

## Author
**Panagiota Grosdouli**  
Electrical & Computer Engineering, D.U.Th.
