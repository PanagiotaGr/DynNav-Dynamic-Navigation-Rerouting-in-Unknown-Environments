# Learned Heuristic with Uncertainty for A*

## Overview

This project introduces a learned heuristic with predictive uncertainty for A* search.
A neural network estimates both the expected cost-to-go and its uncertainty, which are integrated into the A* evaluation function.

---

## Method

We modify the A* scoring function as:

f(n) = g(n) + h_mean(n) + β · h_std(n)

where:

* h_mean: predicted cost-to-go
* h_std: predictive uncertainty
* β: risk parameter

---

## Results

| Method             | Expansions | Suboptimality |
| ------------------ | ---------: | ------------: |
| Classic A*         |      212.6 |         1.000 |
| Learned (β = -0.5) |      104.9 |        1.0005 |
| Learned (β = 0.0)  |   **79.9** |    **1.0005** |
| Learned (β = +0.5) |       81.2 |         1.009 |

✔ Up to **62% fewer node expansions**
✔ Near-optimal paths

---

## Structure

contributions/learned_uncertainty_astar/

* code/
* experiments/
* results/

---

## How to Run

cd contributions/learned_uncertainty_astar/experiments

python run_experiment.py 
--train-grids 200 
--eval-grids 100 
--epochs 80

---

## Key Idea

Learned heuristics + uncertainty = faster search with controllable trade-off.
