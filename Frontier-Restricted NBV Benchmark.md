# Frontier-Restricted NBV Benchmark
Random vs Frontier vs Frontier-Scored Goal Selection

## Overview
This repository benchmarks Next-Best-View (NBV) goal selection under uncertainty in partially explored environments. We compare three policies under identical candidate budgets to isolate the effect of (i) frontier-restricted candidate generation and (ii) safety/return-aware scoring.

Policies:
- Random Global: uniform sampling over free space
- Random Frontier: uniform sampling over the reachable frontier
- Frontier Scored: frontier candidates ranked by a composite objective (information gain + safety/return priors)

## Environment Representation
The environment is discretized into a 2D grid over free cells. Each cell is associated with:
- uncertainty estimate (from navigation/estimation)
- irreversibility score: I ∈ [0, 1]
- returnability score: R ∈ [0, 1]

Interpretation:
- Lower I is safer (less risk of entering difficult-to-recover regions).
- Higher R is better (more feasible to return to a base cell).

## Frontier Candidate Generation
A cell is considered frontier if it is free, sufficiently known, and adjacent (4-neighborhood) to unknown space.

Known-ish constraint:
- unc ≤ τ_known

Unknown-ish adjacency:
- at least one 4-neighbor satisfies unc ≥ τ_unc

Optional feasibility filters:
- minimum returnability: R ≥ R_min
- irreversibility cap: I ≤ I_cap

## Returnability Field
Given a base cell s0, we compute a return-cost grid over free space with a hard irreversibility constraint:
- cells with I > τ_R are disallowed

Returnability is then normalized to R ∈ [0, 1], where unreachable cells map to:
- R = 0

## Scoring Function (Frontier Scored)
For each candidate goal g we compute:
- information gain proxy: IG(g)
- local irreversibility: I_local(g)
- local returnability: R_local(g)

Composite score:
score(g) = IG(g) − α·I_local(g) − β·(1 − R_local(g))

where α and β tune the trade-off between exploration utility and operational safety.

## Benchmark Protocol
- 500 candidates per trial
- top-10 selected goals evaluated
- 30 trials per policy

We report mean ± std for:
- information gain (IG)
- irreversibility (I)
- returnability (R)

## Results Summary
Random Global achieves the highest IG but is not frontier-consistent and does not reflect practical exploration constraints. Random Frontier is frontier-consistent but often selects goals with poor returnability. Frontier Scored improves the frontier-consistent trade-off by increasing IG while improving returnability and reducing irreversibility.

Strong note (keep this in the paper):
Random Global provides an upper bound on IG but is not frontier-consistent; therefore, Random Frontier is the proper baseline to evaluate the benefit of scoring.

## Artifacts
Generated outputs:
- nbv_random_vs_frontier_benchmark.csv
- bench_top10_meanIG_3methods.png
- bench_top10_meanI_3methods.png
- bench_top10_meanR_3methods.png
