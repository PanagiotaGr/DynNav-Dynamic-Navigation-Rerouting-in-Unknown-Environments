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
