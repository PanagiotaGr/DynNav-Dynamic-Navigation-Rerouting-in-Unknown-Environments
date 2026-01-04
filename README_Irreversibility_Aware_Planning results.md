# Irreversibility-Aware Planning

This module introduces **irreversibility-aware path planning** for autonomous robotic navigation in unknown environments under uncertainty.  
The core idea is to explicitly model *points of no return* and enforce them as **hard feasibility constraints** during planning.

Unlike classical risk-aware navigation—where uncertainty is treated as a soft penalty in the objective—irreversibility is modeled as a **binary admissibility condition** controlled by a threshold parameter **τ**.

---

## 1. Motivation

In realistic navigation scenarios, certain regions are not merely risky but *irreversible*. Once entered, recovery may be impossible due to:

- accumulated localization drift,
- lack of visual features,
- topological dead-ends,
- loss of reliable state estimation.

Standard planners such as **A\***, **RRT\***, and risk-weighted planners do not distinguish between:

- *recoverable risk*, and
- *irreversible commitment*.

This module explicitly models **irreversibility** and studies its impact on:

- path feasibility,
- environment connectivity,
- search complexity.

---

## 2. Irreversibility Map

Each grid cell `s` is assigned an **irreversibility score**:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?I(s)%5Cin%5B0,1%5D" alt="I(s) in [0,1]" />
</p>

Higher values indicate a higher probability that entering the cell constitutes a *point of no return*.

### 2.1 Irreversibility Model

Irreversibility is computed as a weighted combination of normalized factors:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?I(s)=w_u%5Ccdot%5Chat%7BU%7D(s)%2Bw_f%5Ccdot(1-%5Chat%7BF%7D(s))%2Bw_d%5Ccdot%5Chat%7BD%7D(s)" alt="I(s)=wu Uhat(s)+wf(1-Fhat(s))+wd Dhat(s)" />
</p>

where:

- **Û(s)**: normalized localization uncertainty  
- **F̂(s)**: normalized feature density  
- **D̂(s)**: drift / instability indicator  
- **wᵤ, w_f, w_d**: weighting coefficients  

All terms are normalized to `[0,1]`. Non-traversable cells are assigned:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?I(s)=1" alt="I(s)=1" />
</p>

---

## 3. Irreversibility-Constrained Planning

A path `π` is **admissible if and only if**:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmax_%7Bs%5Cin%5Cpi%7D%20I(s)%20%5Cle%20%5Ctau" alt="max_{s in pi} I(s) <= tau" />
</p>

Cells with `I(s) > τ` are treated as **hard obstacles**.

This induces a **phase transition** in feasibility:

- feasible planning for `τ ≥ τ*`,
- infeasible planning for `τ < τ*`.

---

## 4. Experiments

### 4.1 Threshold Feasibility Sweep

**Script**
```bash
python run_irreversibility_tau_sweep.py
```

**Setup**
- Fixed start and goal
- Sweep:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Ctau%5Cin%5B0.30,1.00%5D" alt="tau in [0.30,1.00]" />
</p>

**Observation**  
A sharp feasibility threshold is observed at:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Ctau%5E*%5Capprox0.85" alt="tau* approx 0.85" />
</p>

Below this value, planning fails even though the geometric free space remains connected.

---

### 4.2 Bottleneck-Induced Disconnection

**Script**
```bash
python run_irreversibility_bottleneck_sweep.py
```

**Setup**  
A synthetic environment is constructed with:

- a high-irreversibility wall,
- a narrow low-irreversibility door.

For:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Ctau%3CI_%7B%5Ctext%7Bdoor%7D%7D" alt="tau < I_door" />
</p>

no path exists, demonstrating that irreversibility constraints can disconnect the environment.

---

### 4.3 Soft Risk-Weighted Baseline (Comparison)

**Script**
```bash
python run_risk_weighted_lambda_sweep.py
```

**Baseline objective**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?J(%5Cpi)=%5Csum%20c(s)%2B%5Clambda%5Csum%20I(s)" alt="J(pi)=sum c(s)+lambda sum I(s)" />
</p>

**Findings**
- Increasing `λ` reduces **mean** irreversibility exposure.
- However, the **maximum** irreversibility along the path remains constant:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmax%20I=0.95%20%5Cforall%20%5Clambda" alt="max I = 0.95 for all lambda" />
</p>

This shows that soft risk penalties do not prevent traversal of highly irreversible regions.

---

## 5. Results & Discussion

The irreversibility-aware planner exhibits a sharp feasibility transition governed by a critical threshold:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Ctau%5E*%5Capprox0.85" alt="tau* approx 0.85" />
</p>

Below this threshold, planning fails with the failure mode **“no path under irreversibility constraint”**, even when start and goal states are individually safe. This demonstrates that irreversibility constraints can fundamentally alter connectivity—not just cost.

In contrast, the risk-weighted baseline always finds a path by trading geometric cost against accumulated risk. While higher risk weights reduce the **mean** irreversibility along the trajectory, they fail to control **peak** irreversibility. As a result, the soft planner consistently traverses regions corresponding to points of no return.

---

## 6. Visual Evidence

Planned figures (generated by the plotting scripts):

- **Hard vs Soft Planning**
- **Path Overlay on Irreversibility Map**

---

## 7. Key Takeaways

- Hard irreversibility constraints induce **phase transitions** in feasibility.
- Failures can occur due to **constraint-induced disconnection**, not only unsafe start/goal states.
- Soft risk penalties reduce **average** exposure but cannot prevent **irreversible transitions**.
- Irreversibility-aware planning provides a qualitatively different safety mechanism.

---

## 8. Files

- `irreversibility_map.py`
- `irreversibility_planner.py`
- `run_irreversibility_tau_sweep.py`
- `run_irreversibility_bottleneck_sweep.py`
- `risk_weighted_planner.py`
- `run_risk_weighted_lambda_sweep.py`
- `plot_hard_vs_soft_comparison.py`
- `plot_path_overlay_hard_vs_soft.py`

---

## 9. Intended Use

This module is intended for:

- research and educational purposes,
- autonomous navigation under uncertainty,
- feasibility and safety analysis in motion planning.

---

© 2026 **Panagiota Grosdouli**  
School of Electrical & Computer Engineering  
Democritus University of Thrace
