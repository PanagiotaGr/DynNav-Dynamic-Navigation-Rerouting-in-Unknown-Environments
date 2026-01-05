# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

This repository presents a **research-oriented framework for autonomous robotic navigation in unknown and evolving environments under uncertainty**.  
The work focuses on **dynamic replanning**, **uncertainty-aware exploration**, and **risk-sensitive decision-making**, combining **classical planning**, **probabilistic estimation**, and **learning-based heuristics**.

Developed as an **individual research project** at the  
**School of Electrical and Computer Engineering, Democritus University of Thrace (D.U.Th.)**.

---

## Abstract

Autonomous navigation in unknown environments is fundamentally constrained by:

- Partial observability and incrementally built maps  
- Visual odometry drift and feature sparsity  
- Dynamic obstacles requiring frequent replanning  
- Trade-offs between optimality, safety, coverage, and runtime  

This project introduces a **unified navigation pipeline** that explicitly models uncertainty and risk, integrates **learned heuristics** into classical planners, and supports **adaptive behavior under distributional shift (OOD)**.  
The framework is accompanied by **extensive quantitative evaluation**, ablations, and reproducible experiments.

---

## Key Research Contributions

### 1. Uncertainty-Aware Dynamic Navigation (ROS 2)
- Risk-adjusted global planning on evolving occupancy maps  
- Online replanning with uncertainty-weighted A*  
- ROS 2 / Gazebo integration with TurtleBot3  

### 2. Coverage Planning & Exploration Under Uncertainty
- AOI coverage planning inspired by photogrammetry  
- Missing-cell detection and uncertainty-aware replanning  
- Information Gain (IG) and Next-Best-View (NBV) strategies under VO drift  

### 3. Learned A* Heuristics (Optimality-Preserving)
- Neural heuristic accelerating A* while preserving optimal path cost  
- Curriculum training and online self-improvement  
- Node-expansion reduction with formal benchmarking  

### 4. Belief–Risk Planning & Self-Trust Adaptation
- Planning objective combining geometric cost and uncertainty exposure  
- Adaptive risk weighting via self-trust indices  
- OOD-aware safe modes with threshold ablations  

### 5. Irreversibility & Returnability-Aware Navigation
- Planning with bottleneck exposure and irreversibility penalties  
- Frontier/NBV selection constrained by returnability feasibility  
- Synthetic benchmarks and parameter sweeps  

### 6. Security-Aware Estimation & Planning Hooks
- Innovation-based IDS for UKF sensor fusion  
- TF integrity monitoring with CUSUM / EWMA variants  
- Attack-aware adaptive trust weighting  

### 7. Human, Language & Ethics-Aware Extensions
- Human preference–aware risk navigation  
- Language-driven safety logic and self-healing behavior  
- Ethical constraint layers for decision-making  

---

## System Architecture

1. Visual Odometry / SLAM-style state estimation  
2. Probabilistic uncertainty modeling (EKF / UKF)  
3. Risk, entropy, and feature-density maps  
4. Coverage deficit and priority field generation  
5. Dynamic replanning (risk-adjusted A*)  
6. IG / NBV exploration under drift constraints  
7. Self-trust and OOD detection  
8. Multi-seed evaluation and statistical analysis  

---

## Repository Structure

### Core Directories
```
dynamic_nav/              ROS 2 navigation and replanning core
modules/                  Planning, coverage, exploration utilities
photogrammetry_module/    AOI coverage & missing-cell replanning
neural_uncertainty/       Learned uncertainty & calibration models
ig_explorer/, NBV/        IG and Next-Best-View planners
research_experiments/     Reproducible experiment scripts
cpp_extension/            C++ extensions and ROS 2 integration
```

### Results & Artifacts
```
research_results/
results/
figures/
data/plots/
logs_*
```

---

## Installation

### Python Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -e .
```

### ROS 2.

---

## Running Key Experiments

### Learned A* Heuristic
```bash
python train_heuristic.py
python eval_astar_learned.py
```

### IG / NBV Exploration
```bash
python ig_planners_demo.py
python drift_aware_nbv_experiment.py
python drift_aware_nbv_multistep.py
```

### Belief–Risk Planning
```bash
python sweep_lambda_fused_risk.py
python plot_belief_risk_lambda_sweep.py
```

### Irreversibility-Aware Planning
```bash
python run_irreversibility_demo.py
python run_irreversibility_tau_sweep.py
python plot_irreversibility_tau_sweep.py
```

### Safe Mode Experiments
```bash
python analyze_safe_mode_results.py
python safe_mode_threshold_ablation.py
```

### IDS & Security Monitoring
```bash
python eval_ids_sweep.py
python calibrate_tf_cusum.py
```

---

## Reproducibility & Evaluation

This repository includes:
- Multi-seed experiment pipelines  
- Ablation studies  
- Statistical validation artifacts (tables, plots, t-tests)  

Results are stored under `research_results/`, `results/`, and `figures/`.

---

## Citation

See `CITATION.cff`.

---

## License

Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International  
See `LICENSE`.

---

## Disclaimer

This repository is intended **for research and educational use only**.  
The provided models and policies must not be deployed in safety-critical systems without additional validation.

---
##  Author
 Panagiota Grosdouli
