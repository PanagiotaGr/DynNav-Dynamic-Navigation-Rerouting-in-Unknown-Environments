# Dynamic Navigation and Safety-Aware Exploration in Unknown Environments

This repository provides a research-oriented pipeline for autonomous robotic navigation in partially known environments under sensing and localization uncertainty. The system integrates uncertainty-aware planning, learned heuristics, safety-oriented exploration, and attack-aware monitoring, with reproducible experiments, ablations, and statistical validation.

Developed as an individual research project at the School of Electrical and Computer Engineering, Democritus University of Thrace (D.U.Th.).

---

## Highlights

- **Uncertainty-aware navigation & replanning** in evolving maps, combining geometric cost with uncertainty/risk costs.
- **Irreversibility-aware planning** that avoids “curiosity traps” (hard-to-exit regions) using both **hard feasibility constraints (τ)** and **soft risk penalties (λ)**.
- **Returnability- & irreversibility-aware frontier NBV**: frontier-restricted next-best-view selection that balances information gain with safety and recoverability.
- **Learned A\*** heuristics with online refinement: large reduction in node expansions while preserving optimal path cost.
- **Self-trust and OOD-aware behavior**: adaptive risk weighting and safe-mode switching under difficult conditions.
- **Attack-aware state-estimation monitoring** (innovation-based IDS) for UKF-based fusion pipelines.

---

## Problem Setting

Realistic navigation in unknown environments is constrained by:
- incomplete maps during exploration,
- drift and failure modes in visual odometry and low-texture areas,
- uncertainty miscalibration (over/under-confident models),
- dynamic replanning requirements and multi-objective trade-offs (safety, coverage, time, computation).

Classical planners (A\*, RRT\*) often assume reliable state estimation and static cost maps, which becomes brittle under drift, evolving maps, and adversarial disturbances.

---

## Main Contributions

### 1) Irreversibility-Aware Planning (Hard τ vs Soft λ)
We introduce irreversibility-aware planning to avoid entering regions that are risky to escape.

- **Hard constraint (τ):** enforce feasibility only if the planned path satisfies an irreversibility threshold.
- **Soft penalty (λ):** keep planning always feasible but trade-off path length against risk exposure.

---

### 2) Returnability- & Irreversibility-Aware Frontier NBV
We propose a safety-oriented frontier NBV mechanism that avoids curiosity traps by combining:

- **Information gain surrogate** IG(g)
- **Irreversibility penalty** I(g)
- **Returnability** R(g)

Scoring:
score(g) = IG(g) − α·I(g) − β·(1 − R(g))

---

### 3) Learned A* Heuristics with Online Improvement
Neural heuristics for A* reduce node expansions while preserving optimal path cost, with online self-improvement.

---

### 4) Drift Prediction, Calibration, and Risk-Sensitive Planning
Learned drift/uncertainty models and calibration reshape risk distributions to improve robustness.

---

### 5) Self-Trust, OOD Awareness, and Safe-Mode Policies
Adaptive risk weighting and safe-mode activation under difficult conditions.

---

### 6) Attack-Aware Monitoring (Innovation-Based IDS for UKF)
Innovation-based IDS for detecting integrity issues in state estimation.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Disclaimer

Research and educational use only.

---

## Author

Panagiota Grosdouli  
Electrical & Computer Engineering, D.U.Th.  

License: CC BY-NC-SA 4.0
