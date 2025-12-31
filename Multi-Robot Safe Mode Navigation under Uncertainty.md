# Multi-Robot Safe Mode Navigation under Uncertainty

This repository implements a **trust-aware safe-mode mechanism** for multi-robot navigation under perception uncertainty.  
The system estimates **self-trust**, performs **policy switching** between Normal and Safe operation, and supports **risk-aware task allocation**, providing a clean **Safety–Performance Trade-off** with reproducible experiments.

## Problem Statement
In multi-robot systems operating in uncertain environments, sensor noise and localization drift may lead robots to traverse **high-risk regions**.  
This project introduces a mechanism that:

- estimates **Self-Trust** ( S(t) in [0,1] )
- dynamically switches policies between:
  - NORMAL_POLICY
  - SAFE_MODE_POLICY
- allocates tasks considering **risk metrics**

Goal: Reduce operational risk while maintaining mission performance, in a **controlled and tunable** manner.

## Safe Mode Policy
The policy switching follows:

- If **\( S(t) \ge \tau \)** → Normal Navigation
- If **\( S(t) < \tau \)** → Safe Mode Navigation

### SAFE Mode Objectives
- Lower total risk  
- Lower maximum target risk  
- Accept potentially longer path cost when needed  

This enables predictable, explainable behavior suitable for real autonomous systems.

## Key Research Insights

### Threshold Ablation
Tested thresholds:
```
τ ∈ {0.50, 0.60, 0.70, 0.80}
```

Observations:
- Initially (~5 steps) robots run NORMAL policy
- When self-trust drops below τ → SAFE mode engaged
- Clear risk vs distance trade-off:

| Policy | Distance | Risk | Max Risk | Total Cost |
|--------|---------:|------:|---------:|-----------:|
| NORMAL | 1.1 | 1.2 | 0.9 | 4.9 |
| SAFE   | 4.0 | 0.4 | 0.2 | 10.4 |

**Conclusion**
- SAFE mode dramatically reduces risk
- Cost & travel length increase
- τ controls *when* we prioritize safety

---

### λ-Sweep: Risk vs Distance Trade-off
Cost function:
\J = Distance + \lambda \cdot Risk\)

λ values tested:
```
λ ∈ {0, 0.5, 1, 2, 4, 8, 16}
```

Results:

| λ | Distance | Risk | Max Risk | Assignment | Cost |
|---|---------:|------:|---------:|-----------|------:|
| 0.0 | 1.1 | 1.2 | 0.9 | riskier targets | 1.1 |
| 0.5–2.0 | 1.1 | 1.2 | 0.9 | same | 1.7–3.5 |
| **4.0** | **4.0** | **0.4** | **0.2** | switches to SAFE | **5.6** |
| 8.0–16.0 | 4.0 | 0.4 | 0.2 | same | 7.2–10.4 |

A **Phase Transition** is observed near:
\(
\lambda \approx 4
\)
where the system shifts preference from fast risky paths to safer ones.

## Features
- Self-Trust Estimation module
- Dynamic Safe-Mode activation
- Risk-aware Task Allocation
- Threshold Ablation Experiments
- λ-Sweep Analysis
- Full Logging + CSV Export
- Reproducible Experimental Pipeline

## Project Structure
```
.
├── multi_robot_safe_mode_experiment.py
├── multi_robot_safe_mode_logged.py
├── safe_mode_threshold_ablation.py
├── analyze_lambda_sweep_risk_length.py
├── results/
│   ├── threshold_logs.csv
│   ├── lambda_sweep.csv
│   └── plots/
└── README.md
```

## Installation
Requires Python 3.8+

```bash
git clone <repo-url>
cd multi-robot-safe-mode
pip install -r requirements.txt
```

## Running Experiments
### Base Experiment
```bash
python3 multi_robot_safe_mode_experiment.py
```
### Logging Experiment
```bash
python3 multi_robot_safe_mode_logged.py
```
### Threshold Ablation
```bash
python3 safe_mode_threshold_ablation.py
```
### λ-Sweep Analysis
```bash
python3 analyze_lambda_sweep_risk_length.py
```

All results are saved in `results/` as CSV + plot images.

## Output Data
- distance
- risk
- max risk
- switching events
- cost metrics
- assignments

## Scientific Contribution
This system demonstrates:
- Explicit **Safety–Performance Trade-off**
- τ controls activation timing
- λ controls risk penalization
- SAFE mode reduces risk, increases length, ensures predictable behavior

Supports future improvements:
- dynamic τ
- adaptive λ
- per-robot trust modelling



## Summary
The system:
- detects when it should not trust itself
- activates safe navigation policy
- reduces operational risk
- exposes a tunable theoretical trade-off
- is fully reproducible
