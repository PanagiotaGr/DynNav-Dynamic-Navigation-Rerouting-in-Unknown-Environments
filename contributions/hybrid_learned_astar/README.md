# Hybrid Learned A* with Uncertainty-Guided Fallback

This module implements a hybrid planner that uses a learned heuristic when predictive uncertainty is low and falls back to an admissible Euclidean heuristic when uncertainty is high.

## Hybrid rule

- If `h_std <= tau`: use learned `h_mean`
- Else: use Euclidean heuristic

## Evaluation

Compared methods:
- classic_astar
- learned_uncertainty_astar(beta=0.0)
- hybrid_astar(tau)

Metrics:
- found rate
- node expansions
- path length
- suboptimality
- fallback rate
- mean uncertainty
- computation time

# Hybrid Learned A* with Uncertainty-Aware Fallback

## Overview

This module implements a **hybrid planning algorithm** that combines:

- Classical A* with an admissible heuristic (Euclidean distance)
- A learned neural heuristic with uncertainty estimation
- A fallback mechanism based on predictive uncertainty

The goal is to **preserve efficiency gains from learning** while maintaining **robustness and safety guarantees**.

---

## Method

The learned model predicts:

- \( h_{\text{mean}}(n) \): expected cost-to-go
- \( h_{\text{std}}(n) \): uncertainty estimate

Standard learned A* uses:

\[
f(n) = g(n) + h_{\text{mean}}(n)
\]

The hybrid planner introduces a threshold \( \tau \):

- If \( h_{\text{std}}(n) < \tau \) → use learned heuristic
- Else → fallback to admissible heuristic

This defines a **selective trust mechanism** based on uncertainty.

---

## Experimental Setup

- Grid size: 40×40
- Obstacle density: 20%
- Evaluation: 100 random problems
- Metrics:
  - Node expansions
  - Path length
  - Suboptimality
  - Fallback rate
  - Computation time

---

## Results

| Method | Expansions | Fallback Rate | Time (ms) |
|--------|-----------|--------------|----------|
| Classic A* | 1039 | 0.0 | 1.36 |
| Learned A* | 245 | 0.0 | 14.02 |
| Hybrid (τ=0.5) | 1040 | 0.96 | 43.83 |
| Hybrid (τ=1.0) | 1030 | 0.95 | 42.00 |
| Hybrid (τ=1.5) | 1008 | 0.92 | 41.51 |
| Hybrid (τ=2.0) | 1001 | 0.87 | 42.30 |
| Hybrid (τ=3.0) | 910  | 0.76 | 40.03 |

---

## Analysis

The results reveal three key behaviors:

### 1. Learned heuristic improves efficiency
The learned A* reduces node expansions by ~75% compared to classical A*, demonstrating that neural heuristics can significantly accelerate search.

### 2. Hybrid planner trades efficiency for robustness
For low values of \( \tau \), the planner frequently falls back to the admissible heuristic (fallback rate > 0.9), resulting in behavior similar to classical A*.

As \( \tau \) increases:
- fallback decreases
- efficiency improves
- behavior approaches learned A*

### 3. Uncertainty acts as a trust signal
The uncertainty estimate effectively controls when the learned heuristic is used. This enables a principled trade-off between:

- **speed (learning)**
- **safety (fallback)**

---

## Key Insight

The hybrid planner implements a form of:

> **risk-aware heuristic selection**

which is closely related to:

- safe learning
- bounded-suboptimal planning
- uncertainty-aware decision making

---

## Generated Figures

- `paper_expansions_comparison.png`
- `paper_suboptimality_comparison.png`
- `paper_tradeoff_scatter.png`
- `paper_tau_sweep_expansions.png`
- `paper_tau_sweep_suboptimality.png`

---

## How to Run

```bash
PYTHONPATH=. python -m contributions.hybrid_learned_astar.experiments.run_hybrid_experiment \
  --eval-grids 100 \
  --seed 42 \
  --taus 0.5 1.0 1.5 2.0 3.0
