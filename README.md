# Dynamic Navigation under Uncertainty
## Risk-Aware, Learning-Augmented Planning in Unknown Environments.

---

## Abstract

Autonomous navigation in unknown environments is fundamentally limited by uncertainty arising from sensing, state estimation, and environment dynamics.

This work presents a modular navigation framework that explicitly models uncertainty, risk, and irreversibility, enabling dynamic replanning under partial observability.

The framework integrates classical planning algorithms with learned heuristics while preserving formal guarantees, and extends decision-making through risk-aware planning, safety constraints, and resource-aware strategies.

The system is evaluated through controlled experiments, parameter sweeps, and ablation studies, with emphasis on reproducibility and quantitative analysis.

---

## 1. Introduction

Autonomous robots operating in real-world environments must make decisions under uncertainty.

Key challenges include:

- incomplete or evolving maps  
- noisy sensing and state estimation  
- dynamic obstacles  
- safety-critical constraints  

Traditional planning methods assume deterministic and fully known environments, which limits their applicability.

This work investigates how navigation systems can explicitly incorporate uncertainty, risk, and safety into planning and decision-making.

---

## 2. Research Questions

This work is structured around the following research questions:

### RQ1: Can learned heuristics improve planning efficiency without sacrificing optimality guarantees?

We investigate whether neural approximations of heuristic functions can reduce computational cost while preserving the guarantees of classical planners such as A*.

---

### RQ2: How can uncertainty be explicitly incorporated into navigation decisions?

We explore belief-space representations and risk-aware cost formulations for planning under partial observability.

---

### RQ3: How should robots reason about risk and safety in dynamic environments?

We study risk-weighted planning and risk budget constraints, analyzing trade-offs between optimality and safety.

---

### RQ4: How can navigation systems avoid irreversible decisions?

We introduce returnability constraints and analyze feasibility thresholds to prevent entry into unsafe or unrecoverable states.

---

### RQ5: How can autonomous systems remain robust under failures or adversarial conditions?

We integrate anomaly detection mechanisms, including innovation-based intrusion detection and integrity monitoring.

---

### RQ6: How should navigation adapt under resource constraints?

We investigate energy-aware and connectivity-aware planning, along with adaptive safe-mode mechanisms.

---

### RQ7: How can multiple robots coordinate under uncertainty?

We explore decentralized coordination strategies and risk allocation across agents.

---

### RQ8: Can navigation incorporate human preferences and trust?

We examine extensions involving language-based interaction and trust-aware decision-making.

---

## 3. Problem Formulation

We consider a robot operating in an unknown environment with:

- partial observability  
- uncertain state estimation  
- dynamic environmental changes  

The objective is to compute navigation strategies that optimize:

- path efficiency  
- safety  
- robustness  
- resource usage  

while explicitly accounting for uncertainty and risk.

---

## 4. Methodology

The framework combines:

- Classical planning algorithms (A*, graph search)
- Probabilistic reasoning (belief representation, uncertainty)
- Risk-aware cost modeling
- Learning-based heuristic estimation
- Constraint-aware decision-making

The system is structured into modular components, each corresponding to a research question.

---

## 5. Research Modules

### 5.1 Learned Heuristics for A*

- Code:
```

contributions/01_learned_astar/

````

- Run:
```bash
python contributions/01_learned_astar/experiments/eval_astar_learned.py
````

* Addresses: RQ1

---

### 5.2 Risk-Aware and Belief-Space Planning

* Code:

  ```
  contributions/03_belief_risk_planning/
  ```

* Addresses: RQ2, RQ3

---

### 5.3 Irreversibility-Aware Navigation

* Code:

  ```
  contributions/04_irreversibility_returnability/
  ```

* Addresses: RQ4

---

### 5.4 Safe-Mode and Adaptive Navigation

* Code:

  ```
  contributions/05_safe_mode_navigation/
  ```

* Addresses: RQ3, RQ6

---

### 5.5 Energy and Connectivity-Aware Planning

* Code:

  ```
  contributions/06_energy_connectivity/
  ```

* Addresses: RQ6

---

### 5.6 Security-Aware Estimation

* Code:

  ```
  contributions/08_security_ids/
  ```

* Addresses: RQ5

---

### 5.7 Multi-Robot Coordination

* Code:

  ```
  contributions/09_multi_robot/
  ```

* Addresses: RQ7

---

### 5.8 Human-Aware and Trust-Based Navigation

* Code:

  ```
  contributions/10_human_language_ethics/
  ```

* Addresses: RQ8

---

## 6. Experimental Methodology

The framework follows a structured experimental approach:

* parameter sweeps
* multi-seed evaluation
* ablation studies
* comparative analysis

All experiments generate:

* CSV logs
* quantitative metrics
* visualizations

Outputs are stored under:

```
contributions/*/results/
```

---

## 7. Results

The system demonstrates:

* reduced node expansions using learned heuristics
* improved safety through risk-aware planning
* robustness under uncertainty
* detection of anomalous behavior

Detailed results are available per module.

---

## 8. Discussion

The results indicate that:

* uncertainty must be explicitly modeled
* risk-aware planning improves safety
* learned components enhance efficiency
* modular design enables systematic evaluation

---

## 9. Conclusion

This work presents a unified framework for navigation under uncertainty, integrating planning, learning, and safety-aware decision-making.

It provides a foundation for future research in:

* safe autonomy
* uncertainty-aware control
* multi-agent systems

---

## 10. Reproducibility

* structured experiments
* modular implementation
* reproducible pipelines

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
python contributions/01_learned_astar/experiments/eval_astar_learned.py
```

---

## Author

Panagiota Grosdouli
Electrical and Computer Engineering
Democritus University of Thrace

---

## License

Apache License 2.0

```
