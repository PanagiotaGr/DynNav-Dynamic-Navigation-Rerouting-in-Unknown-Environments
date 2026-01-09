# Energy- and Connectivity-Aware Navigation with Safe Mode Activation
-----
## Overview

This module extends the DynNav framework with explicit modeling of
**communication connectivity** and **energy consumption**, and introduces
a **connectivity-triggered safe mode** for autonomous navigation in
uncertain environments.

The planner jointly reasons about:
- geometric path length,
- environmental risk,
- energy expenditure,
- and communication quality,

while dynamically adapting its behavior when persistent communication
degradation is detected.

-------
## Problem Setting

Autonomous robots operating in unknown environments often rely on
wireless communication for supervision, coordination, or data offloading.
However, communication quality is spatially varying and tightly coupled
with energy consumption and navigation decisions.

This module addresses the following questions:

- How should a robot trade off path length, risk, energy, and connectivity?
- When does poor communication require a behavioral mode switch?
- How does enforcing connectivity awareness affect planning complexity?
--

## Joint Cost Function

Navigation is formulated as a weighted multi-objective planning problem.
The per-step cost is defined as:

J = α·L + β·R + δ·E + γ·(1 − C)

where:
- L is the geometric step cost,
- R represents environmental risk,
- E is an energy-related cost,
- C ∈ [0, 1] denotes normalized communication connectivity,
- α, β, δ, γ are scalar trade-off parameters.

Higher values of γ bias the planner toward regions with stronger
communication quality.
----
## Connectivity Model

Communication quality is modeled over the environment as a spatial field.

For each cell x:
- C(x) represents normalized connectivity quality,
- P_loss(x) denotes packet loss probability.

Connectivity is derived from a simplified path-loss model with:
- distance-based attenuation,
- obstacle-induced penalties,
- optional stochastic shadowing.

This abstraction enables reproducible experiments without requiring
hardware-specific radio models.
-----
## Energy Model

Energy consumption is approximated using a spatially varying proxy
that captures:
- baseline motion cost,
- increased expenditure near obstacles or constrained regions.

Energy is normalized and incorporated directly into the planning cost,
allowing explicit exploration of energy–connectivity trade-offs.
------
## Connectivity-Triggered Safe Mode

A safe mode mechanism is introduced to handle persistent communication loss.

Trigger condition:
- If C(x) < C_min for k consecutive steps.

Response:
- The planner replans from the current state with an increased
  connectivity weight γ_safe,
- This biases navigation toward regions with improved communication quality.

This mechanism enables adaptive recovery from communication-degraded regions
without halting the mission.
-------
## Experimental Protocol

Experiments are conducted on randomly generated grid environments with
varying obstacle configurations.

For each environment:
- γ (connectivity weight) and δ (energy weight) are swept jointly,
- navigation is simulated step-by-step,
- safe mode activations and replanning events are logged.

All experiments are multi-seed and fully reproducible.
------
## Metrics

The following metrics are recorded:

- path_len: total path length,
- disconnect_steps: number of steps with C(x) < C_min,
- safe_mode_activations: number of safe mode triggers,
- replans: number of replanning events,
- conn_penalty: cumulative (1 − C) along the path,
- C_mean: mean connectivity along the executed path,
- C_min_path: minimum connectivity encountered.
-----
## Reproducibility

Run the full joint sweep:

python3 run_energy_connectivity_joint_sweep.py

Generate plots:

python3 plot_energy_connectivity_joint_sweep.py
python3 plot_joint_heatmaps.py

All results are stored as CSV logs and PNG figures.
------
## Key Observations

Empirical results reveal clear trade-offs:

- Increasing γ reduces communication degradation and safe mode activations,
  but increases planning effort and path length.
- Energy weighting δ shifts the planner toward shorter, lower-energy paths,
  sometimes at the expense of connectivity.
- Safe mode activation frequency decreases sharply beyond a critical γ,
  indicating a phase-transition-like behavior.
------
## Relation to the DynNav Framework

This module integrates naturally with existing DynNav components:

- risk-aware and irreversibility-aware planners,
- safe mode mechanisms based on uncertainty and failure detection,
- multi-objective and multi-robot extensions.

It demonstrates how communication and energy considerations can be treated
as first-class planning objectives in uncertainty-aware navigation.
----
## Citation

If you use this module or its ideas in academic work, please cite the
DynNav repository as software:

Panagiota Grosdouli.
DynNav: Uncertainty- and Risk-Aware Navigation in Unknown Environments.
GitHub repository, 2026.

See CITATION.cff for full citation metadata.
-------


